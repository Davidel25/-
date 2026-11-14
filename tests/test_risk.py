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


# --- Exit-rule playbook: boundaries, precedence and spread-type parity -------------------------------------------

def call_pos(credit=1.0, debit=0.5, dte=30, contracts=1):
    return SpreadPosition(SpreadType.BEAR_CALL, "SPY", TODAY + timedelta(days=dte), 480, 485,
                          contracts, credit, debit)


def test_take_profit_boundary_is_inclusive_at_50_percent():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(2.0, 1.0), TODAY) is ExitAction.TAKE_PROFIT     # exactly 50% captured
    assert rm.evaluate_exit(pos(2.0, 1.01), TODAY) is ExitAction.HOLD           # just short of 50%
    assert rm.evaluate_exit(pos(2.0, 0.2), TODAY) is ExitAction.TAKE_PROFIT     # well past target


def test_stop_loss_boundary_at_two_times_credit():
    rm = RiskManager()
    # loss == 2x credit means a close debit of 3x credit
    assert rm.evaluate_exit(pos(1.0, 3.0), TODAY) is ExitAction.STOP_LOSS
    assert rm.evaluate_exit(pos(1.0, 2.99), TODAY) is ExitAction.HOLD
    assert rm.evaluate_exit(pos(1.0, 2.0), TODAY) is ExitAction.HOLD            # loss == 1x credit: no exit yet
    assert rm.evaluate_exit(pos(1.0, 5.0), TODAY) is ExitAction.STOP_LOSS       # debit at full width (max loss)


def test_dte_exit_boundary_is_inclusive_at_14_days():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=14), TODAY) is ExitAction.CLOSE_DTE
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=15), TODAY) is ExitAction.HOLD
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=0), TODAY) is ExitAction.CLOSE_DTE


def test_exit_precedence_profit_then_stop_then_dte():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(1.0, 0.4, dte=5), TODAY) is ExitAction.TAKE_PROFIT   # profit beats DTE
    assert rm.evaluate_exit(pos(1.0, 3.5, dte=5), TODAY) is ExitAction.STOP_LOSS     # stop beats DTE
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=5), TODAY) is ExitAction.CLOSE_DTE     # DTE only when nothing else fires


def test_exit_rules_identical_for_bear_call_spreads():
    rm = RiskManager()
    assert rm.evaluate_exit(call_pos(2.0, 1.0), TODAY) is ExitAction.TAKE_PROFIT
    assert rm.evaluate_exit(call_pos(1.0, 3.0), TODAY) is ExitAction.STOP_LOSS
    assert rm.evaluate_exit(call_pos(1.0, 0.9, dte=14), TODAY) is ExitAction.CLOSE_DTE
    assert rm.evaluate_exit(call_pos(1.0, 0.9, dte=30), TODAY) is ExitAction.HOLD


def test_exit_rules_ignore_contract_count():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(2.0, 1.0, contracts=5), TODAY) is ExitAction.TAKE_PROFIT
    assert rm.evaluate_exit(pos(1.0, 3.0, contracts=5), TODAY) is ExitAction.STOP_LOSS


def test_zero_credit_position_never_triggers_profit_or_stop():
    rm = RiskManager()
    assert rm.evaluate_exit(pos(0.0, 0.0), TODAY) is ExitAction.HOLD
    assert rm.evaluate_exit(pos(0.0, 2.0), TODAY) is ExitAction.HOLD
    assert rm.evaluate_exit(pos(0.0, 2.0, dte=10), TODAY) is ExitAction.CLOSE_DTE


def test_custom_exit_limits_are_respected():
    rm = RiskManager(RiskLimits(profit_target_pct=0.25, stop_loss_multiple=1.0, close_at_dte=7))
    assert rm.evaluate_exit(pos(1.0, 0.75), TODAY) is ExitAction.TAKE_PROFIT    # 25% captured
    assert rm.evaluate_exit(pos(1.0, 2.0), TODAY) is ExitAction.STOP_LOSS       # lost 1x credit
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=7), TODAY) is ExitAction.CLOSE_DTE
    assert rm.evaluate_exit(pos(1.0, 0.9, dte=8), TODAY) is ExitAction.HOLD


def test_default_exit_limits_match_playbook():
    lim = RiskLimits()
    assert (lim.profit_target_pct, lim.stop_loss_multiple, lim.close_at_dte) == (0.50, 2.0, 14)
    # playbook example: 0.95 credit, $5 width -> $95 max profit, $405 max loss per lot
    p = SpreadPosition(SpreadType.BEAR_CALL, "SPY", EXP, 780, 785, 1, 0.95, 0.95)
    assert round(p.max_loss) == 405
