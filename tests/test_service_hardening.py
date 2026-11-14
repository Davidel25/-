"""ScanService integration: earnings gate, market-hours and stale-data guards, automatic entry-delta capture,
trade logging, scale-out and the new dashboard endpoints."""
import asyncio
from datetime import UTC, date, datetime, timedelta

from fastapi.testclient import TestClient

from src.alerts.notifiers import AlertDispatcher, Notifier
from src.broker.mock_client import MockBroker
from src.config import Settings
from src.dashboard.app import create_app
from src.models import PositionLeg
from src.risk.earnings import EarningsInfo, EarningsStatus, StaticEarningsProvider
from src.risk.manager import RiskLimits, RiskManager
from src.service import ScanService
from src.storage.trade_log import TradeLog
from src.strategies.credit_spreads import ScannerConfig

TODAY = date.today()
FIRST_FRIDAY = TODAY + timedelta(days=(4 - TODAY.weekday()) % 7 or 7)
EXPIRY = FIRST_FRIDAY + timedelta(weeks=4)                         # a Friday the mock chain lists, ~28-34 DTE


class Rec(Notifier):
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)


def legs(qty=1, short_mark=0.95, long_mark=0.45, symbol="SPY", short=480, long=475):
    return [PositionLeg(symbol, EXPIRY, short, "P", -qty, 2.00, short_mark),
            PositionLeg(symbol, EXPIRY, long, "P", qty, 1.00, long_mark)]


def make(watchlist=("SPY",), broker=None, settings=None, **kw):
    rec = Rec()
    settings = settings or Settings(broker="mock", watchlist=list(watchlist), scan_enabled=False)
    broker = broker or MockBroker(today=TODAY)
    kw.setdefault("scanner_cfg", ScannerConfig(top_n_per_symbol=3))
    return ScanService(settings, broker, AlertDispatcher([rec]), **kw), rec


def scan(svc, times=1):
    for _ in range(times):
        asyncio.run(svc.run_once())


# --- earnings gate -----------------------------------------------------------------------------------------------
def test_unknown_earnings_block_every_candidate_for_a_stock():
    svc, rec = make(("AAPL",))                                      # AAPL is not in the (default) calendar
    scan(svc)
    assert svc.state.signals == [] and svc.state.rejected > 0 and not rec.sent
    assert all("earnings check UNKNOWN" in " ".join(b["reasons"]) for b in svc.state.blocked)


def test_in_window_earnings_block_and_clear_earnings_allow():
    soon = StaticEarningsProvider({"AAPL": EarningsInfo("AAPL", TODAY + timedelta(days=10), EarningsStatus.CONFIRMED)})
    svc, _ = make(("AAPL",), earnings=soon)
    scan(svc)
    assert svc.state.signals == [] and "earnings check RISK" in svc.state.blocked[0]["reasons"][-1]
    far = EarningsInfo("AAPL", TODAY + timedelta(days=200), EarningsStatus.CONFIRMED)
    later = StaticEarningsProvider({"AAPL": far})
    svc2, rec2 = make(("AAPL",), earnings=later)
    scan(svc2)
    assert svc2.state.signals and rec2.sent and svc2.state.blocked == []
    assert svc2.trade_log.signal_count() == len(svc2.state.signals)             # approved signals are logged


def test_earnings_gate_can_be_disabled_and_etfs_are_clear():
    s = Settings(broker="mock", watchlist=["AAPL"], scan_enabled=False, require_earnings_clear=False)
    svc, _ = make(settings=s)
    scan(svc)
    assert svc.state.signals                                         # gate off -> unknown earnings tolerated
    etf, _ = make(("SPY",))
    scan(etf)
    assert etf.state.signals                                         # ETFs have no earnings event


# --- market hours + stale data -----------------------------------------------------------------------------------
def test_scan_is_skipped_outside_market_hours():
    sat = datetime(2026, 11, 14, 15, 0, tzinfo=UTC)
    s = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, enforce_market_hours=True)
    svc, rec = make(settings=s, clock=lambda: sat)
    scan(svc)
    assert svc.state.skip_reason == "market closed" and svc.state.market_open is False
    assert svc.state.signals == [] and not rec.sent and "connect" not in svc.broker.calls
    assert svc.state.scans == 1 and svc.state.last_scan


def test_scan_runs_during_the_session_and_market_hours_can_be_off():
    wed = datetime(2026, 11, 18, 15, 0, tzinfo=UTC)                  # 10:00 ET
    s = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, enforce_market_hours=True)
    svc, _ = make(settings=s, clock=lambda: wed, broker=MockBroker(today=TODAY, clock=lambda: wed))
    scan(svc)
    assert svc.state.skip_reason is None and svc.state.market_open is True


def test_stale_quotes_are_skipped_not_scanned():
    now = datetime.now(UTC)
    old = MockBroker(today=TODAY, clock=lambda: now - timedelta(hours=1))
    s = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, ib_market_data_type=1)   # live: 120 s limit
    svc, rec = make(settings=s, broker=old)
    scan(svc)
    assert svc.state.signals == [] and not rec.sent
    assert "stale data" in svc.state.skipped[0]["reason"]
    fresh, _ = make(settings=s, broker=MockBroker(today=TODAY))
    scan(fresh)
    assert fresh.state.signals and fresh.state.skipped == []


def test_quote_age_limit_follows_market_data_type():
    assert Settings(ib_market_data_type=1).quote_age_limit() == 120
    assert Settings(ib_market_data_type=3).quote_age_limit() == 1200            # delayed data is ~15 min old by design
    assert Settings(ib_market_data_type=3, max_quote_age_sec=45).quote_age_limit() == 45


def test_weekday_filter_skips_the_scan_with_a_reason():
    cfg = ScannerConfig(allowed_weekdays=tuple(d for d in range(7) if d != date.today().weekday()))
    svc, rec = make(scanner_cfg=cfg)
    scan(svc)
    assert svc.state.signals == [] and "entries allowed on" in svc.state.skipped[0]["reason"] and not rec.sent


def test_edge_filters_fail_closed_when_history_is_unavailable():
    class NoHistory(MockBroker):
        async def get_daily_closes(self, symbol, days=90):
            from src.broker.base import BrokerError
            raise BrokerError("no history")
    svc, _ = make(broker=NoHistory(today=TODAY), scanner_cfg=ScannerConfig(min_iv_hv_ratio=1.1, top_n_per_symbol=3))
    scan(svc)
    assert svc.state.signals == []
    assert any("IV/HV unavailable" in k["reason"] for k in svc.state.skipped)


def test_disciplined_service_scan_uses_term_data_window_and_filters():
    s = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, disciplined_scanner=True)
    svc, _ = make(settings=s, scanner_cfg=None)
    assert svc.cfg.min_term_slope == 0.01 and (svc.data.min_dte, svc.data.max_dte) == (12, 70)   # wider data window
    scan(svc)                                                        # mock chain is flat -> blocked, with a reason
    assert svc.state.signals == [] and any("term structure" in k["reason"] for k in svc.state.skipped)


# --- entry delta capture, trade log, scale-out ------------------------------------------------------------------
def test_entry_delta_is_captured_automatically_and_position_close_is_logged():
    broker = MockBroker(today=TODAY, positions=legs(qty=2, short_mark=0.95, long_mark=0.45))
    svc, rec = make(broker=broker)
    scan(svc)
    pos = svc.state.positions[0]
    assert pos["entry_short_delta"] is not None and pos["short_delta"] is not None
    assert not any("no entry delta" in n for n in pos["notes"]) and pos["delta_ratio"] == 1.0
    key = pos["key"]
    assert svc.trade_log.entry_deltas()[key] == pos["entry_short_delta"]
    assert svc.trade_log.initial_contracts(key) == 2
    broker.positions = []                                            # position disappears (closed at the broker)
    scan(svc)
    assert svc.trade_log.closed_trades() == []                       # one missing scan is not enough
    scan(svc)
    (t,) = svc.trade_log.closed_trades()
    assert t["key"] == key and t["pnl_is_estimate"] == 1 and t["exit_reason"] in {"TAKE_PROFIT", "HOLD"}
    assert t["commission"] == 0.65 * 2 * 2 * 2 and svc.trade_log.performance().closed == 1


def test_manual_journal_overrides_the_automatic_entry_delta():
    broker = MockBroker(today=TODAY, positions=legs())
    probe, _ = make(broker=broker)
    scan(probe)
    key = probe.state.positions[0]["key"]
    svc, _ = make(broker=MockBroker(today=TODAY, positions=legs()), entry_deltas={key: 0.01})
    scan(svc)
    assert svc.state.positions[0]["entry_short_delta"] == 0.01
    assert svc.state.positions[0]["delta_ratio"] > 2                 # live delta far above the journaled entry


def test_scale_out_recommendation_flows_to_state_and_notification_once_only():
    broker = MockBroker(today=TODAY, positions=legs(qty=2, short_mark=1.10, long_mark=0.50))   # debit 0.60 = 40%
    svc, rec = make(broker=broker, risk=RiskManager(RiskLimits(scale_out_profit_pct=0.30)))
    scan(svc)
    pos = svc.state.positions[0]
    assert pos["action"] == "SCALE_OUT" and pos["scale_out_contracts"] == 1 and pos["status"] == "ACTION"
    assert any(m.startswith("SCALE OUT 1 of 2 contracts") for m in rec.sent)
    broker.positions = legs(qty=1, short_mark=1.10, long_mark=0.50)  # user closed one contract
    scan(svc)
    assert svc.state.positions[0]["action"] == "HOLD" and svc.state.positions[0]["scale_out_contracts"] == 0


def test_settings_enable_scale_out_via_env(monkeypatch):
    monkeypatch.setenv("SCALE_OUT_PROFIT_PCT", "0.4")
    monkeypatch.setenv("ALLOWED_WEEKDAYS", "1,2,3, 9, x")
    s = Settings.from_env()
    assert s.scale_out_profit_pct == 0.4 and s.allowed_weekdays == (1, 2, 3)
    svc, _ = make(settings=s, scanner_cfg=None)
    assert svc.risk.limits.scale_out_profit_pct == 0.4 and svc.cfg.allowed_weekdays == (1, 2, 3)
    monkeypatch.delenv("SCALE_OUT_PROFIT_PCT")
    monkeypatch.delenv("ALLOWED_WEEKDAYS")
    off = Settings.from_env()
    assert off.scale_out_profit_pct == 0 and off.allowed_weekdays is None and off.enforce_market_hours is False


# --- dashboard endpoints ------------------------------------------------------------------------------------------
def test_dashboard_trade_pnl_and_breakdown_endpoints():
    log = TradeLog()
    from tests.test_hardening import WED, spos
    log.observe_position(spos(), "TAKE_PROFIT", WED)
    log.close_missing(set(), datetime(2026, 1, 20, 15, 0, tzinfo=UTC), min_missing=1)
    log.record_snapshot(date(2026, 1, 20), 12.5, -3.0, 2)
    svc, _ = make(trade_log=log)
    scan(svc)
    with TestClient(create_app(svc)) as c:
        assert len(c.get("/api/trades").json()) == 1
        assert c.get("/api/trades?start=2026-01-21").json() == []
        assert len(c.get("/api/trades?end=2026-01-20").json()) == 1
        assert c.get("/api/trades?start=yesterday").status_code == 422        # malformed dates are rejected
        pnl = c.get("/api/pnl?start=2026-01-01&end=2026-01-31").json()
        assert pnl["snapshots"][0]["unrealized_bull_put"] == 12.5 and pnl["performance"]["closed"] == 1
        assert c.get("/api/breakdown?by=dte_bucket").json()[0]["group"] == "35-45"
        assert c.get("/api/breakdown?by=weekday").json()[0]["group"] == 2
        assert c.get("/api/breakdown?by=bogus").status_code == 422
        st = c.get("/api/status").json()
        assert st["performance"]["closed"] == 1 and st["market_open"] in (True, False) and "blocked" in st
        assert c.get("/api/diagnostics").json().keys() == {"skipped", "blocked"}


def test_dashboard_html_has_the_interactive_controls():
    svc, _ = make()
    with TestClient(create_app(svc)) as c:
        html = c.get("/").text
    for marker in ('id="auto"', 'id="interval"', 'id="q"', 'id="strat"', 'id="perf"', 'id="presets"', 'id="start"',
                   'id="end"', 'id="exp-pos"', 'id="chart"', 'id="toasts"', 'credit_spreads_export_', "csvCell"):
        assert marker in html
