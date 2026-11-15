import asyncio
import json
from datetime import date, timedelta

from src.broker.mock_client import MockBroker
from src.models import PositionLeg, SpreadPosition, SpreadType
from src.risk.earnings import EarningsInfo, EarningsStatus, StaticEarningsProvider
from src.risk.manager import AlertType, ExitAction, RiskManager
from src.risk.report import (
    PositionReport,
    Status,
    build_reports,
    evaluate_position,
    format_report,
    load_entry_deltas,
    main,
    run,
)

TODAY = date(2026, 1, 5)            # Monday; the mock broker lists Friday expiries
EXP = date(2026, 2, 6)              # Friday, 32 DTE


def legs(symbol, short, long, right, credit_short, credit_long, mark_short, mark_long, exp=EXP):
    return [PositionLeg(symbol, exp, short, right, -1, credit_short, mark_short),
            PositionLeg(symbol, exp, long, right, 1, credit_long, mark_long)]


def mock(positions, prices=None):
    return MockBroker(prices=prices or {"SPY": 500.0, "QQQ": 430.0}, positions=positions, today=TODAY)


def reports_for(broker, deltas=None, earnings=None):
    async def go():
        async with broker:
            return await build_reports(broker, RiskManager(), TODAY, earnings, deltas or {})
    return asyncio.run(go())


def test_report_enriches_price_delta_and_runs_every_rule():
    broker = mock(legs("SPY", 480, 475, "P", 2.0, 1.0, 0.9, 0.4))              # debit 0.5 of 1.0 credit -> 50%
    key = "SPY:BULL_PUT:2026-02-06:480/475"
    [r] = reports_for(broker, {key: 0.05}, StaticEarningsProvider())
    p = r.position
    assert p.underlying_price == 500.0 and p.short_delta is not None and p.entry_short_delta == 0.05
    assert r.action is ExitAction.TAKE_PROFIT and r.status is Status.ACTION
    assert AlertType.DELTA_ALERT in {a.type for a in r.alerts}          # live ~0.1+ delta vs 0.05 entry


def test_missing_entry_delta_and_unknown_earnings_force_review_not_ok():
    broker = mock(legs("QQQ", 400, 395, "P", 1.0, 0.5, 0.9, 0.45))
    [r] = reports_for(broker, {}, StaticEarningsProvider())              # QQQ is a known ETF -> earnings clear
    assert r.status is Status.REVIEW and any("no entry delta" in n for n in r.notes)
    assert r.action is ExitAction.HOLD

    p = SpreadPosition(SpreadType.BULL_PUT, "AAPL", EXP, 180, 175, 1, 1.0, 0.9, 190.0, 0.12, 0.12)
    r2 = evaluate_position(RiskManager(), p, TODAY, StaticEarningsProvider())     # AAPL not in the calendar
    assert AlertType.EARNINGS_UNKNOWN in {a.type for a in r2.alerts} and r2.status is Status.REVIEW


def test_earnings_inside_window_makes_the_position_an_action_item():
    p = SpreadPosition(SpreadType.BULL_PUT, "AAPL", EXP, 180, 175, 1, 1.0, 0.9, 190.0, 0.12, 0.12)
    cal = StaticEarningsProvider({"AAPL": EarningsInfo("AAPL", date(2026, 1, 29), EarningsStatus.CONFIRMED)})
    r = evaluate_position(RiskManager(), p, TODAY, cal)
    assert [a.type for a in r.alerts] == [AlertType.EARNINGS_RISK_ALERT]
    assert r.status is Status.ACTION and "Act now" in r.recommendation
    after = StaticEarningsProvider({"AAPL": EarningsInfo("AAPL", EXP + timedelta(days=30), EarningsStatus.CONFIRMED)})
    assert evaluate_position(RiskManager(), p, TODAY, after).status is Status.OK


def test_clean_position_is_ok_and_earnings_check_can_be_disabled():
    p = SpreadPosition(SpreadType.BULL_PUT, "SPY", EXP, 480, 475, 1, 1.0, 0.9, 520.0, 0.10, 0.10)
    assert evaluate_position(RiskManager(), p, TODAY, StaticEarningsProvider()).status is Status.OK
    r = evaluate_position(RiskManager(), p, TODAY, None)
    assert r.status is Status.OK and r.alerts == []


def test_stop_loss_and_dte_are_reported():
    stop = SpreadPosition(SpreadType.BEAR_CALL, "QQQ", EXP, 432, 437, 1, 1.0, 3.0, 436.0, 0.45, 0.15)
    r = evaluate_position(RiskManager(), stop, TODAY, StaticEarningsProvider())
    assert r.action is ExitAction.STOP_LOSS and r.status is Status.ACTION
    assert AlertType.STRIKE_PROXIMITY in {a.type for a in r.alerts}     # 436 is through the 432 short call
    dte = SpreadPosition(SpreadType.BULL_PUT, "SPY", TODAY + timedelta(days=10), 480, 475, 1, 1.0, 0.9, 520.0)
    assert evaluate_position(RiskManager(), dte, TODAY, None).action is ExitAction.CLOSE_DTE


def test_format_report_lists_alerts_and_summary():
    p = SpreadPosition(SpreadType.BEAR_CALL, "QQQ", EXP, 432, 437, 1, 1.0, 3.0, 436.0, 0.45, 0.15)
    text = format_report([evaluate_position(RiskManager(), p, TODAY, StaticEarningsProvider())], TODAY)
    assert "READ-ONLY" in text and "STOP_LOSS" in text and "BREACHED" in text and "ACTION 1" in text


def test_run_exit_codes_and_disconnected_broker(capsys):
    quiet = MockBroker(today=TODAY)
    assert asyncio.run(run(quiet, today=TODAY)) == 0                    # no positions
    assert "No open credit spreads" in capsys.readouterr().out
    busy = mock(legs("SPY", 480, 475, "P", 2.0, 1.0, 0.9, 0.4))
    assert asyncio.run(run(busy, today=TODAY, earnings=StaticEarningsProvider())) == 1
    down = MockBroker(today=TODAY, fail_connect=True)
    assert asyncio.run(run(down, today=TODAY)) == 2
    assert "BROKER ERROR" in capsys.readouterr().err


def test_market_data_failure_is_a_review_note_not_a_crash():
    broker = mock(legs("SPY", 480, 475, "P", 2.0, 1.0, 1.9, 0.9), prices={"QQQ": 430.0})   # no SPY quote
    [r] = reports_for(broker, {}, StaticEarningsProvider())
    assert r.position.underlying_price is None and r.status is Status.REVIEW
    assert any("market data unavailable" in n for n in r.notes)


def test_load_entry_deltas(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"SPY:BULL_PUT:2026-02-06:480/475": 0.15}))
    assert load_entry_deltas(str(f)) == {"SPY:BULL_PUT:2026-02-06:480/475": 0.15}
    assert load_entry_deltas(None) == {}
    try:
        load_entry_deltas(str(tmp_path / "nope.json"))
    except ValueError as exc:
        assert "entry-delta journal" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_demo_and_config_errors(capsys, tmp_path):
    assert main(["--demo"]) == 1                                         # demo contains positions needing attention
    out = capsys.readouterr().out
    assert "SUMMARY" in out and "no orders are placed" in out
    assert main(["--demo", "--earnings-file", str(tmp_path / "missing.json")]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err


def test_report_module_never_references_order_apis():
    import inspect

    import src.risk.report as mod
    src = inspect.getsource(mod).lower()
    assert "placeorder" not in src and "place_order" not in src and "submit_order" not in src
    assert PositionReport  # imported for type coverage
