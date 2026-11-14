from datetime import date, timedelta

from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType
from src.risk.manager import ExitAction, RiskLimits, RiskManager, pair_legs

TODAY = date(2026, 1, 5)
EXP = TODAY + timedelta(days=30)


def sig(max_loss=300.0, symbol="SPY"):
    return SpreadSignal(SpreadType.BULL_PUT, symbol, EXP, 30, 500, 480, 475, 5, 2.0, 200, max_loss,
                        0.66, 0.15, 0.85, 478, 0.04)


def pos(credit=1.0, debit=0.5, dte=30, symbol="SPY", contracts=1):
    return SpreadPosition(SpreadType.BULL_PUT, symbol, TODAY + timedelta(days=dte), 480, 475,
                          contracts, credit, debit)


def test_profit_target_at_50_percent():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(1.0, 0.50), TODAY) is ExitAction.TAKE_PROFIT
    assert rm.evaluate_exit(pos(1.0, 0.51), TODAY) is ExitAction.HOLD
    assert pos(1.0, 0.5).profit_pct == 0.5


def test_stop_loss_and_dte_exit():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(1.0, 3.0), TODAY) is ExitAction.STOP_LOSS      # lost 2x credit
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=10), TODAY) is ExitAction.CLOSE_DTE


def test_signal_limits():
    rm = RiskManager(RiskLimits(max_risk_per_trade_pct=0.02, max_total_risk_pct=0.05))
    assert rm.check_signal(sig(300), [], 50_000).approved
    d = rm.check_signal(sig(1500), [], 50_000)
    assert not d.approved and "trade risk" in d.reasons[0]
    # open risk 400/spread * 6 = 2400 -> adding 300 exceeds 5% of 50k (2500)
    assert not rm.check_signal(sig(300), [pos()] * 6, 50_000).approved
    assert not rm.check_signal(sig(), [], 0).approved


def test_symbol_and_position_count_limits():
    rm = RiskManager(RiskLimits(max_positions_per_symbol=2, max_open_positions=3, max_total_risk_pct=1))
    assert not rm.check_signal(sig(), [pos(), pos()], 1e6).approved
    assert rm.check_signal(sig(symbol="QQQ"), [pos(), pos()], 1e6).approved
    assert not rm.check_signal(sig(symbol="QQQ"), [pos(), pos(), pos()], 1e6).approved


def test_pair_legs_bull_put_and_bear_call():
    legs = [
        PositionLeg("SPY", EXP, 480, "P", -2, 2.00, 1.00), PositionLeg("SPY", EXP, 475, "P", 2, 1.00, 0.50),
        PositionLeg("QQQ", EXP, 450, "C", -1, 1.50, 1.20), PositionLeg("QQQ", EXP, 455, "C", 1, 0.70, 0.60),
    ]
    p = {x.symbol: x for x in pair_legs(legs)}
    assert p["SPY"].spread_type is SpreadType.BULL_PUT and p["SPY"].contracts == 2
    assert p["SPY"].entry_credit == 1.0 and p["SPY"].current_debit == 0.5
    assert p["QQQ"].spread_type is SpreadType.BEAR_CALL and p["QQQ"].long_strike == 455


def test_naked_short_is_not_paired():
    assert pair_legs([PositionLeg("SPY", EXP, 480, "P", -1, 2.0, 1.0)]) == []
    # a long put ABOVE the short put is not a credit spread
    assert pair_legs([PositionLeg("SPY", EXP, 480, "P", -1, 2.0, 1.0),
                      PositionLeg("SPY", EXP, 485, "P", 1, 3.0, 2.0)]) == []
