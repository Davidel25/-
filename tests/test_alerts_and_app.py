import asyncio
from datetime import date, datetime, timedelta

from fastapi.testclient import TestClient

from src.alerts.notifiers import AlertDispatcher, Notifier, format_signal
from src.broker.mock_client import MockBroker
from src.config import Settings
from src.dashboard.app import create_app
from src.models import PositionLeg, SpreadSignal, SpreadType
from src.service import ScanService
from src.strategies.credit_spreads import ScannerConfig

TODAY = date.today()  # service uses real "today" for DTE, so the mock must too


class Recorder(Notifier):
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


class Broken(Notifier):
    async def send(self, message):
        raise RuntimeError("boom")


def test_dispatch_cooldown_and_failure_isolation():
    async def go():
        rec = Recorder()
        d = AlertDispatcher([Broken(), rec], cooldown_sec=60)
        assert await d.dispatch("k", "hello") is True
        assert await d.dispatch("k", "hello") is False       # cooled down
        assert await d.dispatch("k2", "other") is True
        return rec.sent
    assert asyncio.run(go()) == ["hello", "other"]


def make_service(positions=None, **kw):
    rec = Recorder()
    kw.setdefault("profit_target_pct", 0.5)          # these tests are written around the 50% rule
    settings = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, **kw)
    broker = MockBroker(today=TODAY, positions=positions or [])
    return ScanService(settings, broker, AlertDispatcher([rec]),
                       scanner_cfg=ScannerConfig(top_n_per_symbol=3)), rec


def test_scan_service_produces_alerts():
    svc, rec = make_service()
    asyncio.run(svc.run_once())
    assert svc.state.broker_connected and svc.state.scans == 1 and svc.state.last_error is None
    assert svc.state.signals and len(rec.sent) == len(svc.state.signals)
    assert "ROC" in rec.sent[0]


def test_scan_service_flags_profit_target():
    exp = TODAY + timedelta(days=30)
    legs = [PositionLeg("SPY", exp, 480, "P", -1, 2.00, 0.40), PositionLeg("SPY", exp, 475, "P", 1, 1.00, 0.0)]
    svc, rec = make_service(positions=legs)
    asyncio.run(svc.run_once())
    assert svc.state.positions[0]["action"] == "TAKE_PROFIT"
    assert any(m.startswith("CLOSE TAKE_PROFIT") for m in rec.sent)


def test_scan_survives_broker_failure():
    svc, _ = make_service()
    svc.broker.fail_connect = True
    asyncio.run(svc.run_once())
    assert svc.state.last_error and not svc.state.broker_connected


def test_dashboard_endpoints():
    svc, _ = make_service()
    asyncio.run(svc.run_once())
    with TestClient(create_app(svc)) as client:
        assert client.get("/api/health").json() == {"ok": True}
        st = client.get("/api/status").json()
        assert st["broker"] == "mock" and st["signals"] > 0
        assert client.get("/api/signals").json()[0]["symbol"] == "SPY"
        assert client.get("/api/positions").json() == []
        assert "Credit Spread Monitor" in client.get("/").text


def test_format_signal():
    svc, _ = make_service()
    asyncio.run(svc.run_once())
    s = svc.state.signals[0]
    text = format_signal(_rebuild(s))
    assert s["symbol"] in text and f"{s['short_strike']:g}/{s['long_strike']:g}" in text


def _rebuild(s):
    d = {k: v for k, v in s.items() if k != "key"}
    d.update(spread_type=SpreadType(d["spread_type"]), expiry=date.fromisoformat(d["expiry"]),
             created_at=datetime.fromisoformat(d["created_at"]))
    return SpreadSignal(**d)


# --- Alerts wired into ScanService and the dashboard -------------------------------------------------------------

FRIDAY = TODAY + timedelta(days=(4 - TODAY.weekday()) % 7 or 7) + timedelta(weeks=4)  # an expiry the mock lists


def aapl_legs(short=190, long=185):
    exp = FRIDAY
    return [PositionLeg("AAPL", exp, short, "P", -1, 2.00, 1.80), PositionLeg("AAPL", exp, long, "P", 1, 1.00, 0.90)]


def svc_with(positions, earnings=None, entry_deltas=None):
    from src.risk.earnings import StaticEarningsProvider
    rec = Recorder()
    settings = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False)
    svc = ScanService(settings, MockBroker(today=TODAY, positions=positions), AlertDispatcher([rec]),
                      scanner_cfg=ScannerConfig(top_n_per_symbol=1),
                      earnings=earnings or StaticEarningsProvider(), entry_deltas=entry_deltas or {})
    return svc, rec


def test_breached_strike_and_unknown_earnings_are_alerted_and_published():
    svc, rec = svc_with(aapl_legs(short=190, long=185))              # AAPL spot 190 == short put strike
    asyncio.run(svc.run_once())
    pos = svc.state.positions[0]
    types = {a["type"] for a in pos["alerts"]}
    assert types == {"STRIKE_PROXIMITY", "EARNINGS_UNKNOWN"}         # AAPL not in the calendar -> fail safe
    assert pos["status"] == "ACTION" and pos["distance_to_short_pct"] == 0 and pos["underlying_price"] == 190.0
    assert {a["severity"] for a in svc.state.alerts} == {"HIGH", "WARN"}
    assert any(m.startswith("ALERT [HIGH] STRIKE_PROXIMITY") for m in rec.sent)
    assert svc.position_counts() == {"ACTION": 1, "REVIEW": 0, "OK": 0}


def test_earnings_inside_window_dispatches_earnings_risk_alert():
    from src.risk.earnings import EarningsInfo, EarningsStatus, StaticEarningsProvider
    cal = StaticEarningsProvider({"AAPL": EarningsInfo("AAPL", TODAY + timedelta(days=10), EarningsStatus.CONFIRMED)})
    svc, rec = svc_with(aapl_legs(short=170, long=165), earnings=cal)     # strikes far OTM: only earnings fires
    asyncio.run(svc.run_once())
    assert [a["type"] for a in svc.state.alerts] == ["EARNINGS_RISK_ALERT"]
    assert any("EARNINGS_RISK_ALERT" in m for m in rec.sent)


def test_alert_dispatch_respects_cooldown_across_scans():
    svc, rec = svc_with(aapl_legs(short=190, long=185))
    asyncio.run(svc.run_once())
    asyncio.run(svc.run_once())                               # second scan within the cooldown window
    assert sum(m.startswith("ALERT [HIGH] STRIKE_PROXIMITY") for m in rec.sent) == 1


def test_delta_alert_uses_entry_delta_journal():
    exp_legs = aapl_legs(short=170, long=165)
    key = f"AAPL:BULL_PUT:{FRIDAY}:170/165"
    from src.risk.earnings import EarningsInfo, EarningsStatus, StaticEarningsProvider
    cal = StaticEarningsProvider({"AAPL": EarningsInfo("AAPL", TODAY + timedelta(days=400), EarningsStatus.CONFIRMED)})
    svc, _ = svc_with(exp_legs, earnings=cal, entry_deltas={key: 0.001})   # live delta will be >= 2x this tiny entry
    asyncio.run(svc.run_once())
    pos = svc.state.positions[0]
    if pos["short_delta"] is not None:                       # mock lists Friday expiries only
        assert "DELTA_ALERT" in {a["type"] for a in pos["alerts"]}
    else:
        assert any("delta unavailable" in n for n in pos["notes"]) and pos["status"] == "REVIEW"


def test_dashboard_exposes_alerts_and_status_counts():
    svc, _ = svc_with(aapl_legs(short=190, long=185))
    asyncio.run(svc.run_once())
    with TestClient(create_app(svc)) as client:
        al = client.get("/api/alerts").json()
        assert {a["type"] for a in al} == {"STRIKE_PROXIMITY", "EARNINGS_UNKNOWN"}
        st = client.get("/api/status").json()
        assert st["alerts"] == 2 and st["position_status"]["ACTION"] == 1
        pos = client.get("/api/positions").json()[0]
        assert pos["status"] == "ACTION" and pos["alerts"] and "recommendation" in pos
        html = client.get("/").text
        assert "Risk alerts" in html and '"alerts"' in html


def test_entry_delta_journal_is_read_from_settings_path(tmp_path):
    f = tmp_path / "deltas.json"
    f.write_text('{"X:BULL_PUT:2026-01-01:1/0": 0.2}')
    svc = ScanService(Settings(broker="mock", entry_deltas_file=str(f), scan_enabled=False), MockBroker(today=TODAY),
                      AlertDispatcher([]))
    assert svc._journal_now() == {"X:BULL_PUT:2026-01-01:1/0": 0.2}
    f.write_text("not json")
    assert svc._journal_now() == {}                      # bad journal never crashes a scan
