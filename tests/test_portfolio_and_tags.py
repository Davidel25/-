"""Beta-weighted delta cap (math, RiskManager gate/alerts, service wiring) and trade tagging."""
import asyncio
import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.alerts.notifiers import AlertDispatcher, Notifier
from src.broker.mock_client import MockBroker
from src.config import Settings
from src.dashboard.app import create_app
from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType
from src.risk.manager import AlertType, RiskLimits, RiskManager, Severity
from src.risk.portfolio import (
    DEFAULT_BETAS,
    Exposure,
    PortfolioSnapshot,
    build_snapshot,
    net_delta_shares,
    parse_betas,
    position_exposure,
    signal_dollar_delta,
)
from src.service import ScanService
from src.storage.trade_log import ALLOWED_TAGS, TradeLog
from src.strategies.credit_spreads import ScannerConfig

TODAY = date.today()
EXP = TODAY + timedelta(days=30)
FRIDAY = TODAY + timedelta(days=(4 - TODAY.weekday()) % 7 or 7) + timedelta(weeks=4)   # an expiry the mock lists
BETAS = {"SPY": 1.0, "QQQ": 1.15}


def vpos(st=SpreadType.BULL_PUT, sym="SPY", price=500.0, short=0.20, long=0.10, contracts=1):
    return SpreadPosition(st, sym, EXP, 480, 475, contracts, 1.0, 0.5, price, short, None, None, long)


def vsig(st=SpreadType.BULL_PUT, sym="SPY", price=500.0, short=0.20, long=0.10, max_loss=300.0):
    return SpreadSignal(st, sym, EXP, 30, price, 480, 475, 5, 1.0, 100, max_loss, 0.3, short, 0.8, 479, 0.04,
                        long_delta=long)


# --- exposure math ----------------------------------------------------------------------------------------------
def test_parse_betas_ignores_junk():
    assert parse_betas("qqq=1.15, IWM = 1.2,bad,x=-1,y=abc,=2") == {"QQQ": 1.15, "IWM": 1.2}
    assert parse_betas("") == {}


def test_net_delta_of_a_vertical_is_signed_by_strategy():
    assert net_delta_shares(SpreadType.BULL_PUT, 0.20, 0.10, 1) == (pytest.approx(10.0), True)    # long the market
    assert net_delta_shares(SpreadType.BEAR_CALL, 0.20, 0.10, 1) == (pytest.approx(-10.0), True)  # short the market
    shares, exact = net_delta_shares(SpreadType.BULL_PUT, -0.20, None, 2)                         # signed input is fine
    assert shares == pytest.approx(40.0) and exact is False                    # long leg unknown -> conservative
    assert net_delta_shares(SpreadType.BULL_PUT, 0.20, 0.10, 3)[0] == pytest.approx(30.0)


def test_position_exposure_beta_weights_and_flags_assumptions():
    e = position_exposure(vpos(price=500, short=0.20, long=0.10), BETAS)
    assert e.net_delta_shares == pytest.approx(10) and e.dollar_delta == pytest.approx(10 * 500 * 1.0)
    q = position_exposure(vpos(sym="QQQ", price=430), BETAS)
    assert q.dollar_delta == pytest.approx(10 * 430 * 1.15) and q.beta_known
    unknown = position_exposure(vpos(sym="XYZ", price=100), BETAS)
    assert unknown.beta == 1.0 and unknown.beta_known is False
    assert position_exposure(vpos(short=None), BETAS) is None                           # no delta -> not computable
    bare = SpreadPosition(SpreadType.BULL_PUT, "SPY", EXP, 480, 475, 1, 1.0, 0.5)       # no price, no delta
    assert position_exposure(bare, BETAS) is None


def test_bull_put_and_bear_call_offset():
    snap = build_snapshot([vpos(), vpos(SpreadType.BEAR_CALL)], BETAS, 500.0, 20_000, 1.0)
    assert snap.net_dollar_delta == pytest.approx(0.0) and snap.complete
    assert build_snapshot([vpos(), vpos()], BETAS, 500.0, 20_000, 1.0).net_dollar_delta == pytest.approx(10_000)


def test_signal_dollar_delta_matches_equivalent_position():
    assert signal_dollar_delta(vsig(), BETAS) == pytest.approx(5_000)
    assert signal_dollar_delta(vsig(SpreadType.BEAR_CALL), BETAS) == pytest.approx(-5_000)
    assert signal_dollar_delta(vsig(), BETAS, contracts=2) == pytest.approx(10_000)
    assert signal_dollar_delta(vsig(long=None), BETAS) == pytest.approx(10_000)        # no long delta: conservative


def test_snapshot_metrics_and_completeness():
    snap = build_snapshot([vpos(), vpos(sym="QQQ", price=430)], BETAS, 500.0, 20_000, 0.5)
    assert snap.net_dollar_delta == pytest.approx(5_000 + 10 * 430 * 1.15)
    assert snap.spy_equivalent_shares == pytest.approx(snap.net_dollar_delta / 500)
    assert snap.pct_of_equity == pytest.approx(snap.net_dollar_delta / 20_000)
    assert snap.utilization == pytest.approx(snap.pct_of_equity / 0.5)
    d = snap.to_dict()
    assert d["complete"] and len(d["positions"]) == 2 and d["positions"][0]["beta_known"] is True
    assert not build_snapshot([vpos(short=None)], BETAS, 500.0, 20_000, 1.0).complete        # missing delta
    assert not build_snapshot([vpos()], BETAS, None, 20_000, 1.0).complete                  # no SPY quote
    assert not build_snapshot([vpos()], BETAS, 500.0, 0, 1.0).complete                      # no equity
    assert build_snapshot([], BETAS, 500.0, 20_000, 1.0).complete                           # empty book is fine
    assert "SPY" in DEFAULT_BETAS


# --- RiskManager gate and alerts --------------------------------------------------------------------------------
def snap(net, equity=20_000.0, spy=500.0, missing=()):
    ex = [Exposure("k", "SPY", net / 500, 1.0, True, True, net)] if net else []
    return PortfolioSnapshot(ex, list(missing), spy, equity, 1.0)


def test_gate_blocks_candidates_that_push_exposure_past_the_cap():
    rm = RiskManager()
    ok = rm.check_signal(vsig(), [], 20_000, portfolio=snap(15_000), signal_dollar_delta=3_000)          # 90%
    assert ok.approved
    bad = rm.check_signal(vsig(), [], 20_000, portfolio=snap(15_000), signal_dollar_delta=6_000)         # 105%
    assert not bad.approved and "beta-weighted delta +105%" in bad.reasons[-1]
    short_side = rm.check_signal(vsig(SpreadType.BEAR_CALL), [], 20_000, portfolio=snap(-15_000),
                                 signal_dollar_delta=-6_000)
    assert not short_side.approved                                                                         # -105%


def test_gate_allows_trades_that_reduce_exposure_even_when_over_the_cap():
    rm = RiskManager()
    over = snap(25_000)                                                                                   # 125%
    assert rm.check_signal(vsig(SpreadType.BEAR_CALL), [], 20_000, portfolio=over, signal_dollar_delta=-5_000).approved
    assert not rm.check_signal(vsig(), [], 20_000, portfolio=over, signal_dollar_delta=1_000).approved
    flip = rm.check_signal(vsig(SpreadType.BEAR_CALL), [], 20_000, portfolio=snap(5_000), signal_dollar_delta=-8_000)
    assert flip.approved                                                                  # ends net -15%: still fine


def test_gate_fails_closed_on_incomplete_data_and_is_skipped_without_a_snapshot():
    rm = RiskManager()
    res = rm.check_signal(vsig(), [], 20_000, portfolio=snap(1_000, missing=["SPY:BULL_PUT:x"]))
    assert not res.approved and "unavailable" in res.reasons[-1] and "SPY:BULL_PUT:x" in res.reasons[-1]
    assert not rm.check_signal(vsig(), [], 20_000, portfolio=snap(1_000, spy=None)).approved
    assert rm.check_signal(vsig(), [], 20_000).approved                                   # no snapshot given


def test_cap_is_configurable():
    tight = RiskManager(RiskLimits(max_net_beta_delta_pct=0.25))
    assert not tight.check_signal(vsig(), [], 20_000, portfolio=snap(4_000), signal_dollar_delta=2_000).approved
    assert RiskManager(RiskLimits(max_net_beta_delta_pct=0.5)).check_signal(
        vsig(), [], 20_000, portfolio=snap(4_000), signal_dollar_delta=2_000).approved


def test_portfolio_alerts_warn_near_the_cap_and_go_high_over_it():
    rm = RiskManager()
    assert rm.evaluate_portfolio(snap(10_000)) == []                                                      # 50%
    assert rm.evaluate_portfolio(snap(0)) == []                                                           # nothing open
    (warn,) = rm.evaluate_portfolio(snap(17_000))                                                         # 85%
    assert warn.type is AlertType.BETA_DELTA_LIMIT and warn.severity is Severity.WARN and "approaching" in warn.message
    (high,) = rm.evaluate_portfolio(snap(-22_000))                                                        # 110% short
    assert high.severity is Severity.HIGH and "short" in high.message and "over the cap" in high.message
    (inc,) = rm.evaluate_portfolio(snap(1_000, missing=["k"]))
    assert inc.severity is Severity.WARN and "incomplete" in inc.message


# --- service wiring ----------------------------------------------------------------------------------------------
class Rec(Notifier):
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)


def spy_put_legs(qty=1):
    return [PositionLeg("SPY", FRIDAY, 480, "P", -qty, 2.00, 0.95),
            PositionLeg("SPY", FRIDAY, 475, "P", qty, 1.00, 0.45)]


def build(broker=None, risk=None, settings=None, watch=("SPY",)):
    rec = Rec()
    settings = settings or Settings(broker="mock", watchlist=list(watch), scan_enabled=False)
    svc = ScanService(settings, broker or MockBroker(today=TODAY, positions=spy_put_legs()), AlertDispatcher([rec]),
                      risk=risk, scanner_cfg=ScannerConfig(top_n_per_symbol=50))
    return svc, rec


def test_service_publishes_portfolio_snapshot_and_per_position_dollars():
    svc, _ = build()
    asyncio.run(svc.run_once())
    p = svc.state.portfolio
    assert p["complete"] and p["net_beta_delta_dollars"] > 0 and p["spy_price"] == 500.0
    pos = svc.state.positions[0]
    assert pos["beta_delta_dollars"] == pytest.approx(p["net_beta_delta_dollars"]) and pos["long_delta"] is not None
    with TestClient(create_app(svc)) as c:
        assert c.get("/api/portfolio").json()["complete"] is True


def test_cap_blocks_added_long_exposure_but_not_exposure_reducing_signals():
    svc, _ = build(risk=RiskManager(RiskLimits(max_net_beta_delta_pct=0.001)))
    asyncio.run(svc.run_once())
    kinds = {s["spread_type"] for s in svc.state.signals}
    assert "BEAR_CALL" in kinds and "BULL_PUT" not in kinds                       # only the hedging side survives
    assert any("beta-weighted delta" in " ".join(b["reasons"]) for b in svc.state.blocked)


def test_breach_raises_a_high_portfolio_alert_and_notification():
    svc, rec = build(risk=RiskManager(RiskLimits(max_net_beta_delta_pct=0.001)))
    asyncio.run(svc.run_once())
    (a,) = [x for x in svc.state.alerts if x["position"] == "PORTFOLIO"]
    assert a["type"] == "BETA_DELTA_LIMIT" and a["severity"] == "HIGH"
    assert any(m.startswith("ALERT [HIGH] BETA_DELTA_LIMIT") for m in rec.sent)


def test_missing_spy_quote_makes_the_snapshot_incomplete_and_blocks_signals():
    broker = MockBroker(prices={"QQQ": 430.0}, today=TODAY)
    svc, _ = build(broker=broker, watch=("QQQ",))
    asyncio.run(svc.run_once())
    assert svc.state.portfolio["spy_price"] is None and svc.state.portfolio["complete"] is False
    assert svc.state.signals == [] and any("beta-weighted delta unavailable" in " ".join(b["reasons"])
                                           for b in svc.state.blocked)


def test_custom_betas_scale_the_exposure():
    base, _ = build()
    asyncio.run(base.run_once())
    s2 = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, betas={"SPY": 2.0})
    hi, _ = build(settings=s2)
    asyncio.run(hi.run_once())
    one, two = base.state.portfolio["net_beta_delta_dollars"], hi.state.portfolio["net_beta_delta_dollars"]
    assert two == pytest.approx(2 * one)


def test_betas_and_cap_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("BETAS", "AAPL=1.3,QQQ=1.25")
    monkeypatch.setenv("MAX_NET_BETA_DELTA_PCT", "0.6")
    s = Settings.from_env()
    assert (s.betas["AAPL"], s.betas["QQQ"], s.betas["SPY"]) == (1.3, 1.25, 1.0)    # file values override defaults
    assert s.max_net_beta_delta_pct == 0.6
    svc, _ = build(settings=s)
    assert svc.risk.limits.max_net_beta_delta_pct == 0.6


# --- trade tagging -------------------------------------------------------------------------------------------------
WED = datetime(2026, 1, 7, 15, 0, tzinfo=UTC)


def tpos(debit=0.5, sym="SPY"):
    return SpreadPosition(SpreadType.BULL_PUT, sym, date(2026, 2, 16), 480, 475, 1, 1.0, debit, 500.0, 0.18)


def closed_log():
    log = TradeLog()
    for sym, debit in (("SPY", 0.5), ("QQQ", 3.0), ("IWM", 0.4)):
        log.observe_position(tpos(debit, sym), "HOLD", WED)
    log.close_missing(set(), WED + timedelta(days=10), min_missing=1)
    return log


def test_set_tags_normalises_validates_and_replaces():
    log = closed_log()
    key = tpos().key
    t = log.set_tags(key, [" Stop_Hit ", "earnings_gap", "stop_hit", ""], "  gapped through on earnings  ")
    assert t["tags"] == ["stop_hit", "earnings_gap"] and t["notes"] == "gapped through on earnings"
    assert log.closed_trades()[0]["tags"] in ([], ["stop_hit", "earnings_gap"])
    assert log.set_tags(key, ["scratch"], None)["tags"] == ["scratch"] and log.set_tags(key, [], "")["notes"] is None
    with pytest.raises(ValueError, match="unknown tag"):
        log.set_tags(key, ["lol"])
    with pytest.raises(ValueError, match="limited to"):
        log.set_tags(key, [], "x" * 501)
    with pytest.raises(KeyError):
        log.set_tags("nope", ["scratch"])
    assert any(e["type"] == "TRADE_TAGGED" for e in log.events())


def test_untagged_trades_default_to_an_empty_list():
    assert all(t["tags"] == [] and t["notes"] is None for t in closed_log().closed_trades())


def test_tag_breakdown_counts_multi_tag_trades_under_each_tag_and_groups_untagged():
    log = closed_log()
    log.set_tags(tpos(3.0, "QQQ").key, ["stop_hit", "macro_selloff"], "panic")
    log.set_tags(tpos(0.5, "SPY").key, ["profit_target"])
    by = {r["group"]: r for r in log.breakdown("tag")}
    assert set(by) == {"stop_hit", "macro_selloff", "profit_target", "untagged"}
    assert by["stop_hit"]["net_pnl"] == by["macro_selloff"]["net_pnl"] < 0 and by["untagged"]["trades"] == 1
    assert by["profit_target"]["wins"] == 1


def test_allowed_tag_vocabulary_is_small_and_fixed():
    assert "earnings_gap" in ALLOWED_TAGS and "macro_selloff" in ALLOWED_TAGS and len(ALLOWED_TAGS) <= 15


def test_old_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE positions (key TEXT PRIMARY KEY, symbol TEXT, spread_type TEXT, expiry TEXT, short_strike REAL,
        long_strike REAL, contracts INTEGER, first_seen TEXT, last_seen TEXT, entry_credit REAL, entry_underlying REAL,
        entry_short_delta REAL, entry_delta_source TEXT, last_debit REAL, last_action TEXT,
        status TEXT NOT NULL DEFAULT 'OPEN', closed_at TEXT, exit_debit REAL, exit_reason TEXT, commission REAL,
        gross_pnl REAL, net_pnl REAL, pnl_is_estimate INTEGER NOT NULL DEFAULT 1);
      CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, scan_date TEXT NOT NULL,
        created_at TEXT NOT NULL, symbol TEXT, spread_type TEXT, expiry TEXT, short_strike REAL, long_strike REAL,
        credit REAL, max_loss REAL, roc REAL, short_delta REAL, underlying_price REAL, UNIQUE (key, scan_date));
      INSERT INTO positions (key, symbol, status, closed_at, net_pnl, contracts, entry_credit)
        VALUES ('SPY:BULL_PUT:2025-12-19:480/475', 'SPY', 'CLOSED', '2025-12-01T10:00:00+00:00', 42.0, 1, 1.0);
    """)
    db.commit()
    db.close()
    log = TradeLog(path)                                                          # must not raise
    (t,) = log.closed_trades()
    assert t["tags"] == [] and t["net_pnl"] == 42.0
    assert log.set_tags(t["key"], ["scratch"], "legacy row")["tags"] == ["scratch"]
    assert log.record_signal(SpreadSignal(SpreadType.BULL_PUT, "SPY", EXP, 30, 500, 480, 475, 5, 1.0, 100, 400,
                                          0.25, 0.2, 0.8, 479, 0.04), WED)
    log.close()


# --- tagging API -------------------------------------------------------------------------------------------------
def test_tag_endpoints():
    log = closed_log()
    svc, _ = build(broker=MockBroker(today=TODAY), settings=Settings(broker="mock", watchlist=["SPY"],
                                                                    scan_enabled=False))
    svc.trade_log = log
    key = tpos().key
    with TestClient(create_app(svc)) as c:
        assert c.get("/api/tags").json()["tags"] == list(ALLOWED_TAGS)
        ok = c.post("/api/trades/tag", json={"key": key, "tags": ["Stop_Hit", "macro_selloff"], "notes": "VIX spike"})
        assert ok.status_code == 200 and ok.json()["tags"] == ["stop_hit", "macro_selloff"]
        shown = {t["key"]: t for t in c.get("/api/trades").json()}
        assert shown[key]["notes"] == "VIX spike"
        assert c.post("/api/trades/tag", json={"key": key, "tags": ["bogus"]}).status_code == 422
        assert c.post("/api/trades/tag", json={"key": "missing", "tags": []}).status_code == 404
        assert c.post("/api/trades/tag", json={"key": key, "tags": [], "notes": "x" * 501}).status_code == 422
        assert c.post("/api/trades/tag", json={"tags": []}).status_code == 422
        assert {r["group"] for r in c.get("/api/breakdown?by=tag").json()} >= {"stop_hit", "untagged"}


def test_dashboard_html_has_portfolio_and_tagging_controls():
    svc, _ = build()
    with TestClient(create_app(svc)) as c:
        html = c.get("/").text
    for marker in ('id="pf"', 'id="tagdlg"', 'id="tag"', "/api/trades/tag", '"portfolio","tags"', "data-tag=", "β-Δ $"):
        assert marker in html
