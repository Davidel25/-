from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType
from src.risk.manager import AlertType, ExitAction, RiskLimits, RiskManager, Severity, pair_legs

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


# --- Delta alert and strike-proximity alert ----------------------------------------------------------------------

def dpos(spread_type=SpreadType.BULL_PUT, short=480, long=475, und=None, delta=None, entry=None):
    return SpreadPosition(spread_type, "SPY", EXP, short, long, 1, 1.0, 0.5, und, delta, entry)


def kinds(alerts):
    return {a.type for a in alerts}


def test_new_inputs_default_to_unknown_and_stay_backward_compatible():
    p = pos()
    assert (p.underlying_price, p.short_delta, p.entry_short_delta, p.distance_to_short_pct) == (None, None, None, None)
    assert RiskManager().evaluate_alerts(p, TODAY) == []          # unknown inputs never alert
    assert p.to_dict()["distance_to_short_pct"] is None


def test_delta_alert_fires_at_exactly_2x_entry_delta():
    rm = RiskManager()
    assert kinds(rm.evaluate_alerts(dpos(delta=0.30, entry=0.15), TODAY)) == {AlertType.DELTA_ALERT}
    assert kinds(rm.evaluate_alerts(dpos(delta=0.45, entry=0.15), TODAY)) == {AlertType.DELTA_ALERT}
    assert rm.evaluate_alerts(dpos(delta=0.29, entry=0.15), TODAY) == []
    assert rm.evaluate_alerts(dpos(delta=0.15, entry=0.15), TODAY) == []


def test_delta_alert_uses_absolute_values_and_skips_missing_inputs():
    rm = RiskManager()
    assert kinds(rm.evaluate_alerts(dpos(delta=-0.32, entry=-0.15), TODAY)) == {AlertType.DELTA_ALERT}  # signed ok
    assert rm.evaluate_alerts(dpos(delta=0.40, entry=None), TODAY) == []
    assert rm.evaluate_alerts(dpos(delta=0.40, entry=0.0), TODAY) == []     # zero entry delta: no divide-by-zero
    assert rm.evaluate_alerts(dpos(delta=None, entry=0.15), TODAY) == []


def test_delta_alert_multiple_is_configurable():
    rm = RiskManager(RiskLimits(delta_alert_multiple=1.5))
    assert kinds(rm.evaluate_alerts(dpos(delta=0.23, entry=0.15), TODAY)) == {AlertType.DELTA_ALERT}


def test_proximity_alert_bull_put_within_1_percent():
    rm = RiskManager()
    near = rm.evaluate_alerts(dpos(und=484.0), TODAY)                   # 0.83% above the 480 short put
    assert kinds(near) == {AlertType.STRIKE_PROXIMITY} and near[0].severity is Severity.WARN
    assert rm.evaluate_alerts(dpos(und=484.8), TODAY) != []             # exactly 1.0% -> alert (inclusive)
    assert rm.evaluate_alerts(dpos(und=485.0), TODAY) == []             # 1.04% -> safe
    assert rm.evaluate_alerts(dpos(und=520.0), TODAY) == []


def test_proximity_alert_bear_call_within_1_percent():
    rm = RiskManager()
    call = dict(spread_type=SpreadType.BEAR_CALL, short=500, long=505)
    near = rm.evaluate_alerts(dpos(**call, und=496.0), TODAY)           # 0.8% below the 500 short call
    assert kinds(near) == {AlertType.STRIKE_PROXIMITY} and near[0].severity is Severity.WARN
    assert rm.evaluate_alerts(dpos(**call, und=495.0), TODAY) != []     # exactly 1.0% below: alert is inclusive
    assert rm.evaluate_alerts(dpos(**call, und=494.9), TODAY) == []     # 1.02% below: safe


def test_proximity_alert_is_high_severity_once_strike_is_breached():
    rm = RiskManager()
    for p in (dpos(und=480.0), dpos(und=470.0),                                           # put at / through short
              dpos(SpreadType.BEAR_CALL, 500, 505, und=500.0), dpos(SpreadType.BEAR_CALL, 500, 505, und=512.0)):
        a = rm.evaluate_alerts(p, TODAY)
        assert kinds(a) == {AlertType.STRIKE_PROXIMITY} and a[0].severity is Severity.HIGH
        assert "BREACHED" in a[0].message


def test_proximity_threshold_is_configurable():
    rm = RiskManager(RiskLimits(strike_proximity_pct=0.03))
    assert kinds(rm.evaluate_alerts(dpos(und=493.0), TODAY)) == {AlertType.STRIKE_PROXIMITY}   # 2.7% away
    assert rm.evaluate_alerts(dpos(und=495.0), TODAY) == []


def test_both_alerts_can_fire_together_and_do_not_change_exit_action():
    rm = RiskManager()
    p = dpos(und=482.0, delta=0.35, entry=0.15)
    assert kinds(rm.evaluate_alerts(p, TODAY)) == {AlertType.DELTA_ALERT, AlertType.STRIKE_PROXIMITY}
    assert rm.evaluate_exit(p, TODAY) is ExitAction.TAKE_PROFIT  # credit 1.0 / debit 0.5 -> exit logic untouched


# --- Partial scale-out (opt-in) ----------------------------------------------------------------------------------

def so_rm(pct=0.30, fraction=0.5):
    return RiskManager(RiskLimits(scale_out_profit_pct=pct, scale_out_fraction=fraction))


def test_scale_out_is_off_by_default():
    rm = RiskManager()
    p = pos(2.0, 1.2, contracts=4)                                   # 40% captured
    assert rm.evaluate_exit(p, TODAY) is ExitAction.HOLD and rm.scale_out_contracts(p) == 0


def test_scale_out_triggers_between_trigger_and_target_and_closes_half():
    rm = so_rm(0.30)
    p = pos(2.0, 1.2, contracts=4)                                   # 40% captured, 4 contracts
    assert rm.evaluate_exit(p, TODAY) is ExitAction.SCALE_OUT and rm.scale_out_contracts(p) == 2
    assert rm.evaluate_exit(pos(2.0, 1.41, contracts=4), TODAY) is ExitAction.HOLD     # 29.5%: below the trigger
    assert rm.evaluate_exit(pos(2.0, 1.4, contracts=4), TODAY) is ExitAction.SCALE_OUT  # 30% boundary is inclusive
    assert so_rm(0.40).evaluate_exit(pos(2.0, 1.3, contracts=4), TODAY) is ExitAction.HOLD   # 35% < 40% trigger


def test_scale_out_needs_two_contracts_and_always_leaves_one_open():
    rm = so_rm(0.30)
    assert rm.evaluate_exit(pos(2.0, 1.2, contracts=1), TODAY) is ExitAction.HOLD      # nothing to split
    sizes = {n: rm.scale_out_contracts(pos(2.0, 1.2, contracts=n)) for n in (2, 3, 5, 10)}
    assert sizes == {2: 1, 3: 1, 5: 2, 10: 5}
    assert so_rm(fraction=1.0).scale_out_contracts(pos(2.0, 1.2, contracts=4)) == 3     # never closes everything
    assert so_rm(fraction=0.1).scale_out_contracts(pos(2.0, 1.2, contracts=4)) == 1     # at least one


def test_scale_out_fires_once_per_position():
    rm = so_rm(0.30)
    full = replace(pos(2.0, 1.2, contracts=4), initial_contracts=4)
    reduced = replace(pos(2.0, 1.2, contracts=2), initial_contracts=4)                  # already scaled out earlier
    unknown = replace(pos(2.0, 1.2, contracts=4), initial_contracts=None)
    assert rm.evaluate_exit(full, TODAY) is ExitAction.SCALE_OUT
    assert rm.evaluate_exit(reduced, TODAY) is ExitAction.HOLD and rm.scale_out_contracts(reduced) == 0
    assert rm.evaluate_exit(unknown, TODAY) is ExitAction.SCALE_OUT


def test_scale_out_never_overrides_the_hard_exits():
    rm = so_rm(0.30)
    assert rm.evaluate_exit(pos(2.0, 0.9, contracts=4), TODAY) is ExitAction.TAKE_PROFIT        # >= 50% wins
    assert rm.evaluate_exit(pos(2.0, 1.2, contracts=4, dte=10), TODAY) is ExitAction.CLOSE_DTE  # DTE exit wins
    assert rm.evaluate_exit(pos(2.0, 6.0, contracts=4), TODAY) is ExitAction.STOP_LOSS


def test_scale_out_works_for_bear_calls_too():
    p = replace(call_pos(2.0, 1.2, contracts=2), initial_contracts=2)
    assert so_rm().evaluate_exit(p, TODAY) is ExitAction.SCALE_OUT


def test_scale_out_locks_in_only_a_small_amount_versus_remaining_risk():
    """Documents why this is not 'risk-free': the kept contracts still carry their full max loss."""
    p = pos(1.0, 0.6, contracts=2)                                   # 40% of a $1.00 credit, $5 wide
    locked = (p.entry_credit - p.current_debit) * 100 * 1            # one contract closed
    remaining_risk = (p.width - p.entry_credit) * 100 * 1
    assert locked == pytest.approx(40) and remaining_risk == pytest.approx(400) and locked < remaining_risk
