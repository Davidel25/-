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
        assert "Credit Spread Scanner" in client.get("/").text


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
