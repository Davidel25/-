"""Automated earnings feed: vendor adapter, cache/TTL/staleness, fail-closed behaviour, layering, service wiring."""
import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.alerts.notifiers import AlertDispatcher
from src.broker.mock_client import MockBroker
from src.config import Settings
from src.dashboard.app import create_app
from src.risk.earnings import EarningsCheck, EarningsInfo, EarningsStatus, JsonFile, classify_earnings
from src.risk.earnings_feed import (
    EarningsFeedError,
    EarningsRecord,
    FeedEarningsProvider,
    FinnhubSource,
    LayeredEarningsProvider,
    RequestsFetcher,
    build_earnings_provider,
)
from src.service import ScanService
from src.strategies.credit_spreads import ScannerConfig

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
TODAY = NOW.date()
KEY = "sk-secret-key-123456"


class FakeFetcher:
    """Stands in for the network. ``plan`` maps symbol -> payload or an exception to raise."""

    def __init__(self, plan=None):
        self.plan, self.calls = plan or {}, []

    def get_json(self, url, params, headers, timeout):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers), "timeout": timeout})
        out = self.plan.get(params["symbol"], {"earningsCalendar": []})
        if isinstance(out, Exception):
            raise out
        return out


def cal(sym, *dates, hour="amc"):
    return {"earningsCalendar": [{"symbol": sym, "date": d, "hour": hour, "quarter": 3, "year": 2026} for d in dates]}


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def make(plan=None, clock=None, **kw):
    fetcher = FakeFetcher(plan)
    clock = clock or Clock()
    return FeedEarningsProvider(FinnhubSource(KEY, fetcher), clock=clock, **kw), fetcher, clock


# --- Finnhub adapter ------------------------------------------------------------------------------------------------
def test_adapter_builds_the_request_and_parses_rows():
    f = FakeFetcher({"TJX": cal("TJX", "2026-11-18", "2027-02-25")})
    rows = FinnhubSource(KEY, f).fetch("TJX", TODAY, 120)
    assert rows == [EarningsRecord(date(2026, 11, 18), "amc"), EarningsRecord(date(2027, 2, 25), "amc")]
    call = f.calls[0]
    assert call["url"].endswith("/calendar/earnings") and call["params"]["symbol"] == "TJX"
    assert call["params"]["from"] == "2026-11-14" and call["params"]["to"] == "2027-03-14"
    assert call["headers"] == {"X-Finnhub-Token": KEY} and "token" not in call["params"]      # header auth by default


def test_adapter_query_auth_mode_and_validation():
    f = FakeFetcher()
    FinnhubSource(KEY, f, auth="query").fetch("TJX", TODAY, 30)
    assert f.calls[0]["params"]["token"] == KEY and f.calls[0]["headers"] == {}
    with pytest.raises(ValueError, match="API key"):
        FinnhubSource("")
    with pytest.raises(ValueError, match="header"):
        FinnhubSource(KEY, auth="cookie")


def test_adapter_is_tolerant_of_odd_rows_but_strict_about_the_shape():
    junk = {"earningsCalendar": [None, {"symbol": "OTHER", "date": "2026-11-20"}, {"symbol": "TJX"},
                                 {"symbol": "tjx", "date": "garbage"},
                                 {"symbol": "tjx", "date": "2026-11-19T00:00:00"}]}
    got = FinnhubSource(KEY, FakeFetcher({"TJX": junk})).fetch("TJX", TODAY, 120)
    assert got == [EarningsRecord(date(2026, 11, 19))]
    for bad in ({}, [], {"earningsCalendar": "x"}, "oops", None):
        with pytest.raises(EarningsFeedError, match="unexpected response shape"):
            FinnhubSource(KEY, FakeFetcher({"TJX": bad})).fetch("TJX", TODAY, 120)


def test_the_api_key_never_appears_in_repr_or_settings_repr():
    assert KEY not in repr(FinnhubSource(KEY)) and "***" in repr(FinnhubSource(KEY))
    s = Settings(earnings_api_key=KEY, telegram_token="tg-secret-token", webhook_url="https://hook.example/secret")
    assert KEY not in repr(s) and "tg-secret-token" not in repr(s) and "hook.example" not in repr(s)


# --- RequestsFetcher error sanitising (no network) ------------------------------------------------------------------
class Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self._p, self._bad = status, payload, bad_json

    def json(self):
        if self._bad:
            raise ValueError("boom")
        return self._p


@pytest.mark.parametrize("status,needle", [(401, "rejected the credentials"), (403, "rejected the credentials"),
                                           (429, "rate limit"), (500, "HTTP 500")])
def test_requests_fetcher_maps_http_errors_without_leaking_secrets(monkeypatch, status, needle):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(status))
    with pytest.raises(EarningsFeedError, match=needle) as ei:
        RequestsFetcher().get_json("https://x.example/?token=" + KEY, {"token": KEY}, {"X-Finnhub-Token": KEY}, 5)
    assert KEY not in str(ei.value) and "x.example" not in str(ei.value)


def test_requests_fetcher_handles_transport_failures_and_success(monkeypatch):
    import requests

    def raise_(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(requests, "get", raise_(requests.Timeout("slow " + KEY)))
    with pytest.raises(EarningsFeedError, match="timed out") as e1:
        RequestsFetcher().get_json("u", {}, {}, 1)
    monkeypatch.setattr(requests, "get", raise_(requests.ConnectionError("dns " + KEY)))
    with pytest.raises(EarningsFeedError, match="network error") as e2:
        RequestsFetcher().get_json("u", {}, {}, 1)
    assert KEY not in str(e1.value) and KEY not in str(e2.value)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, bad_json=True))
    with pytest.raises(EarningsFeedError, match="invalid JSON"):
        RequestsFetcher().get_json("u", {}, {}, 1)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, {"ok": 1}))
    assert RequestsFetcher().get_json("u", {}, {}, 1) == {"ok": 1}


# --- provider: interpretation ---------------------------------------------------------------------------------------
def test_vendor_dates_are_projected_never_confirmed_and_take_the_next_upcoming_one():
    p, _, _ = make({"TJX": cal("TJX", "2026-08-20", "2026-11-18", "2027-02-25")})
    p.refresh(["tjx"], TODAY)
    info = p.get("TJX")
    assert info == EarningsInfo("TJX", date(2026, 11, 18), EarningsStatus.PROJECTED)      # the past date is ignored
    assert classify_earnings(info, date(2026, 12, 18), TODAY) is EarningsCheck.RISK       # inside the trade window
    assert classify_earnings(info, date(2026, 11, 16), TODAY) is EarningsCheck.UNCONFIRMED  # just after expiry: review


def test_missing_empty_or_implausible_vendor_answers_fail_closed():
    p, _, _ = make({"NONE": cal("NONE"), "FAR": cal("FAR", "2027-03-30"), "BOTH": cal("BOTH", "2026-01-05")})
    p.refresh(["NONE", "FAR", "BOTH", "NEVER"], TODAY)
    assert p.get("NONE").status is EarningsStatus.UNKNOWN                  # empty answer is NOT "no earnings"
    assert p.get("FAR").status is EarningsStatus.UNKNOWN and p.get("FAR").date is None   # 136 days: implausible
    assert p.get("BOTH").status is EarningsStatus.UNKNOWN                  # only past dates
    assert p.get("NEVER").status is EarningsStatus.UNKNOWN                 # fetched nothing for it at all
    assert p.get("ZZZ").status is EarningsStatus.UNKNOWN                   # never requested


def test_etfs_never_touch_the_network():
    p, f, _ = make()
    p.refresh(["SPY", "QQQ", "IWM"], TODAY)
    assert f.calls == [] and p.get("SPY").status is EarningsStatus.NOT_APPLICABLE
    assert classify_earnings(p.get("QQQ"), date(2026, 12, 18), TODAY) is EarningsCheck.CLEAR


# --- provider: TTL, failures, staleness -----------------------------------------------------------------------------
def test_ttl_limits_how_often_a_symbol_is_fetched():
    p, f, clock = make({"TJX": cal("TJX", "2026-11-18")}, ttl_hours=12)
    p.refresh(["TJX"], TODAY)
    p.refresh(["TJX"], TODAY)
    clock.advance(hours=11)
    p.refresh(["TJX"], TODAY)
    assert len(f.calls) == 1
    clock.advance(hours=2)
    p.refresh(["TJX"], TODAY)
    assert len(f.calls) == 2


def test_failed_refresh_keeps_the_old_value_until_it_goes_stale_then_blocks():
    p, f, clock = make({"TJX": cal("TJX", "2026-11-18")}, ttl_hours=12, max_stale_hours=72)
    p.refresh(["TJX"], TODAY)
    f.plan["TJX"] = EarningsFeedError("earnings feed timed out")
    clock.advance(hours=13)
    p.refresh(["TJX"], TODAY)
    assert p.get("TJX").status is EarningsStatus.PROJECTED                    # still trusted: 13 h old
    h = p.health()
    assert not h["ok"] and h["failed"] == {"TJX": "earnings feed timed out"} and "timed out" in h["last_error"]
    clock.advance(hours=70)                                                   # now 83 h old
    p.refresh(["TJX"], TODAY)
    assert p.get("TJX").status is EarningsStatus.UNKNOWN                      # expired -> fail closed
    assert "TJX" in p.health()["stale"]


def test_recovery_clears_the_failure():
    p, f, clock = make({"TJX": EarningsFeedError("HTTP 500")})
    p.refresh(["TJX"], TODAY)
    assert not p.health()["ok"] and p.get("TJX").status is EarningsStatus.UNKNOWN
    f.plan["TJX"] = cal("TJX", "2026-11-18")
    p.refresh(["TJX"], TODAY)
    assert p.health()["ok"] and p.health()["failed"] == {} and p.get("TJX").date == date(2026, 11, 18)


def test_unexpected_exceptions_are_contained_and_do_not_leak_details():
    p, _, _ = make({"TJX": RuntimeError("secret " + KEY)})
    p.refresh(["TJX"], TODAY)                                                  # must not raise
    assert p.get("TJX").status is EarningsStatus.UNKNOWN
    assert KEY not in json.dumps(p.health()) and "unexpected error (RuntimeError)" in p.health()["failed"]["TJX"]


def test_call_limit_and_pacing():
    sleeps = []
    plan = {s: cal(s, "2026-11-20") for s in "ABCDE"}
    p, f, _ = make(plan, max_calls=3, pace_sec=0.5)
    p._sleep = sleeps.append
    p.refresh(list("ABCDE"), TODAY)
    assert len(f.calls) == 3 and sleeps == [0.5, 0.5]
    assert "call limit" in p.health()["last_error"]
    assert p.get("E").status is EarningsStatus.UNKNOWN                         # the unrefreshed ones stay blocked


def test_health_before_any_refresh_is_not_ok():
    assert make()[0].health()["ok"] is False


# --- provider: cache persistence ------------------------------------------------------------------------------------
def test_cache_survives_a_restart_and_respects_staleness(tmp_path):
    path = tmp_path / "sub" / "earn.json"
    clock = Clock()
    p1 = FeedEarningsProvider(FinnhubSource(KEY, FakeFetcher({"TJX": cal("TJX", "2026-11-18")})), path, clock=clock)
    p1.refresh(["TJX"], TODAY)
    assert path.exists() and not list(path.parent.glob("*.tmp"))                # atomic write left no temp file
    f2 = FakeFetcher()
    p2 = FeedEarningsProvider(FinnhubSource(KEY, f2), path, clock=clock)
    assert p2.get("TJX").date == date(2026, 11, 18)                             # loaded from disk
    p2.refresh(["TJX"], TODAY)
    assert f2.calls == []                                                       # still within the TTL: no refetch
    clock.advance(hours=100)
    assert p2.get("TJX").status is EarningsStatus.UNKNOWN                       # too old to trust


@pytest.mark.parametrize("content", ["not json", "[]", '{"TJX": {"status": "NOPE", "fetched_at": "x"}}',
                                     '{"TJX": {"date": "2026-11-18"}}'])
def test_corrupt_cache_means_unknown_not_a_guess(tmp_path, content):
    path = tmp_path / "earn.json"
    path.write_text(content)
    p = FeedEarningsProvider(FinnhubSource(KEY, FakeFetcher()), path, clock=Clock())
    assert p.get("TJX").status is EarningsStatus.UNKNOWN


def test_unwritable_cache_does_not_break_refresh(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    p = FeedEarningsProvider(FinnhubSource(KEY, FakeFetcher({"TJX": cal("TJX", "2026-11-18")})), blocker / "c.json",
                             clock=Clock())
    p.refresh(["TJX"], TODAY)                                                   # directory creation fails: logged only
    assert p.get("TJX").date == date(2026, 11, 18)


# --- layering and construction ----------------------------------------------------------------------------------------
def test_manual_calendar_wins_and_can_confirm(tmp_path):
    manual = tmp_path / "m.json"
    manual.write_text(json.dumps({"TJX": {"date": "2026-11-18", "confirmed": True}}))
    feed, f, _ = make({"TJX": cal("TJX", "2026-11-25"), "DELL": cal("DELL", "2026-11-24")})
    layered = LayeredEarningsProvider(JsonFile(manual), feed)
    layered.refresh(["TJX", "DELL", "SPY"], TODAY)
    assert layered.get("TJX") == EarningsInfo("TJX", date(2026, 11, 18), EarningsStatus.CONFIRMED)   # manual wins
    assert layered.get("DELL").status is EarningsStatus.PROJECTED and layered.get("DELL").date == date(2026, 11, 24)
    assert layered.get("SPY").status is EarningsStatus.NOT_APPLICABLE
    assert [c["params"]["symbol"] for c in f.calls] == ["DELL"]                # no feed call for manually known TJX
    assert layered.health()["provider"] == "finnhub" and layered.age_hours("DELL") == 0


def test_build_provider_from_settings():
    assert build_earnings_provider(Settings()).get("AAPL").status is EarningsStatus.UNKNOWN   # no calendar
    s = Settings(earnings_provider="finnhub", earnings_api_key=KEY, earnings_auth="query")
    prov = build_earnings_provider(s, FakeFetcher({"TJX": cal("TJX", "2026-11-18")}), Clock())
    prov.refresh(["TJX"], TODAY)
    assert prov.get("TJX").date == date(2026, 11, 18)
    with pytest.raises(ValueError, match="API key"):
        build_earnings_provider(Settings(earnings_provider="finnhub"))
    with pytest.raises(ValueError, match="unknown EARNINGS_PROVIDER"):
        build_earnings_provider(Settings(earnings_provider="magic"))


def test_environment_configuration(monkeypatch):
    for k, v in {"EARNINGS_PROVIDER": "Finnhub", "EARNINGS_API_KEY": KEY, "EARNINGS_AUTH": "QUERY",
                 "EARNINGS_TTL_HOURS": "6", "EARNINGS_MAX_STALE_HOURS": "48", "EARNINGS_CACHE_FILE": "x.json"}.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert (s.earnings_provider, s.earnings_auth, s.earnings_ttl_hours, s.earnings_max_stale_hours,
            s.earnings_cache_file) == ("finnhub", "query", 6.0, 48.0, "x.json") and s.earnings_api_key == KEY


# --- service wiring ---------------------------------------------------------------------------------------------------
REAL = date.today()


def service(plan, symbols=("AAPL",), **kw):
    settings = Settings(broker="mock", watchlist=list(symbols), scan_enabled=False, earnings_provider="finnhub",
                        earnings_api_key=KEY)
    fetcher = FakeFetcher(plan)
    prov = build_earnings_provider(settings, fetcher)
    svc = ScanService(settings, MockBroker(today=REAL), AlertDispatcher([]), earnings=prov,
                      scanner_cfg=ScannerConfig(top_n_per_symbol=3), **kw)
    return svc, fetcher


def test_service_refreshes_the_feed_and_the_gate_uses_it():
    soon = (REAL + timedelta(days=10)).isoformat()
    svc, f = service({"AAPL": cal("AAPL", soon)})
    asyncio.run(svc.run_once())
    assert [c["params"]["symbol"] for c in f.calls] == ["AAPL"]
    assert svc.state.signals == [] and "earnings check RISK" in svc.state.blocked[0]["reasons"][-1]
    assert svc.state.earnings_feed["provider"] == "finnhub" and svc.state.earnings_feed["ok"] is True
    asyncio.run(svc.run_once())
    assert len(f.calls) == 1                                                     # second scan served from the cache


def test_service_clears_a_stock_whose_next_report_is_after_expiry():
    later = (REAL + timedelta(days=100)).isoformat()
    svc, _ = service({"AAPL": cal("AAPL", later)})
    asyncio.run(svc.run_once())
    assert svc.state.signals and svc.state.blocked == []


def test_feed_outage_blocks_signals_and_is_reported():
    svc, _ = service({"AAPL": EarningsFeedError("earnings feed timed out")})
    asyncio.run(svc.run_once())
    assert svc.state.signals == [] and "earnings check UNKNOWN" in svc.state.blocked[0]["reasons"][-1]
    feed = svc.state.earnings_feed
    assert feed["ok"] is False and feed["failed"] == {"AAPL": "earnings feed timed out"}


def test_a_crashing_provider_never_breaks_the_scan():
    class Boom:
        def get(self, symbol):
            return EarningsInfo(symbol, None, EarningsStatus.UNKNOWN)

        def refresh(self, symbols):
            raise RuntimeError("kaboom " + KEY)

    settings = Settings(broker="mock", watchlist=["AAPL"], scan_enabled=False)
    svc = ScanService(settings, MockBroker(today=REAL), AlertDispatcher([]), earnings=Boom(),
                      scanner_cfg=ScannerConfig(top_n_per_symbol=3))
    asyncio.run(svc.run_once())
    assert svc.state.signals == [] and svc.state.last_error is None and svc.state.scans == 1


def test_refresh_failures_are_logged_without_the_key(caplog):
    svc, _ = service({"AAPL": RuntimeError("secret " + KEY)})
    with caplog.at_level(logging.DEBUG):
        asyncio.run(svc.run_once())
    assert KEY not in caplog.text


def test_earnings_endpoint_shows_what_the_gate_believes():
    soon = (REAL + timedelta(days=10)).isoformat()
    svc, _ = service({"AAPL": cal("AAPL", soon)}, symbols=("AAPL", "SPY", "MSFT"))
    asyncio.run(svc.run_once())
    with TestClient(create_app(svc)) as c:
        body = c.get("/api/earnings").json()
    rows = {r["symbol"]: r for r in body["rows"]}
    assert rows["AAPL"]["status"] == "PROJECTED" and rows["AAPL"]["days_to"] == 10 and rows["AAPL"]["age_hours"] < 1
    assert rows["SPY"]["status"] == "NOT_APPLICABLE" and rows["MSFT"]["status"] == "UNKNOWN"   # MSFT: feed had nothing
    assert body["feed"]["provider"] == "finnhub"


def test_manual_only_setup_still_works_and_reports_no_feed():
    svc = ScanService(Settings(broker="mock", watchlist=["SPY"], scan_enabled=False), MockBroker(today=REAL),
                      AlertDispatcher([]), scanner_cfg=ScannerConfig(top_n_per_symbol=2))
    asyncio.run(svc.run_once())
    assert svc.state.earnings_feed == {} and svc.state.signals
    with TestClient(create_app(svc)) as c:
        assert c.get("/api/earnings").json()["feed"] == {}
        html = c.get("/").text
    for marker in ('id="earn"', 'id="earnfeed"', '"earnings"', "BLOCKS"):
        assert marker in html
