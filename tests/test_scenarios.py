"""What-if scenario presets: repricing math, clamps, beta scaling, fail-safe handling, service and API wiring."""
import asyncio
from dataclasses import replace
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from src.alerts.notifiers import AlertDispatcher
from src.broker.mock_client import MockBroker
from src.config import Settings
from src.dashboard.app import create_app
from src.models import PositionLeg, SpreadPosition, SpreadType
from src.risk.scenarios import PRESET_KEYS, PRESETS, ScenarioPreset, reprice, run_presets, run_scenario
from src.service import ScanService
from src.strategies.credit_spreads import ScannerConfig

TODAY = date(2026, 1, 5)
EXP = TODAY + timedelta(days=34)
BETAS = {"SPY": 1.0, "QQQ": 1.15}
BY = {p.key: p for p in PRESETS}


def bull(contracts=1, sym="SPY", spot=500.0, dte_days=34, short_iv=0.20, long_iv=0.21):
    return SpreadPosition(SpreadType.BULL_PUT, sym, TODAY + timedelta(days=dte_days), 480, 475, contracts, 1.0, 0.6,
                          spot, 0.15, 0.15, None, 0.10, short_iv, long_iv)


def bear(contracts=1, spot=500.0):
    return SpreadPosition(SpreadType.BEAR_CALL, "SPY", EXP, 520, 525, contracts, 1.0, 0.6, spot, 0.15, 0.15, None, 0.10,
                          0.18, 0.18)


def change(pos, preset, betas=BETAS):
    return reprice(pos, preset, betas, TODAY).pnl_change


def test_a_zero_shock_changes_nothing():
    none = ScenarioPreset("x", "x", 0.0, 0.0, 0)
    for p in (bull(), bear(), bull(3)):
        r = reprice(p, none, BETAS, TODAY)
        assert r.pnl_change == pytest.approx(0.0, abs=1e-9) and r.projected_pnl == pytest.approx(p.unrealized_pnl)


def test_bull_put_loses_on_a_drop_more_on_a_bigger_drop_and_gains_on_a_rally():
    d3, d5, up3 = change(bull(), BY["spy_down_3"]), change(bull(), BY["spy_down_5"]), change(bull(), BY["spy_up_3"])
    assert d5 < d3 < 0 < up3
    assert up3 <= bull().entry_credit * 100 - bull().unrealized_pnl + 1e-6        # cannot exceed max profit


def test_bear_call_is_the_mirror_image():
    assert change(bear(), BY["spy_down_3"]) > 0 > change(bear(), BY["spy_up_3"])


def test_volatility_shock_hurts_both_sides_and_time_decay_helps_both():
    assert change(bull(), BY["iv_up_10"]) < 0 and change(bear(), BY["iv_up_10"]) < 0
    assert change(bull(), BY["week_passes"]) > 0 and change(bear(), BY["week_passes"]) > 0


def test_a_sell_off_with_rising_vol_is_worse_than_the_same_drop_with_flat_vol():
    assert change(bull(), BY["spy_down_3_iv5"]) < change(bull(), BY["spy_down_3"])


def test_results_are_clamped_to_max_loss_and_max_profit():
    crash = ScenarioPreset("c", "c", -0.40, 0.0, 0)                                      # far through the short strike
    r = reprice(bull(), crash, BETAS, TODAY)
    assert r.at_max_loss and r.projected_pnl == pytest.approx(-bull().max_loss)         # -(5 - 1) * 100 = -$400
    assert r.pnl_change == pytest.approx(-bull().max_loss - bull().unrealized_pnl)
    rally = reprice(bull(), ScenarioPreset("r", "r", 0.30, 0.0, 0), BETAS, TODAY)
    assert rally.projected_pnl <= 100 + 1e-9 and not rally.at_max_loss                  # cannot beat the credit
    assert not reprice(bull(), BY["spy_up_3"], BETAS, TODAY).at_max_loss


@pytest.mark.parametrize("debit", [0.2, 0.6, 1.5, 3.0])
def test_extremes_are_exact_whatever_the_current_mark_is(debit):
    """Model and market rarely agree on the current value; a total wipe-out must still equal -max loss."""
    p = replace(bull(), current_debit=debit)
    wipe = reprice(p, ScenarioPreset("w", "w", -0.4, 0.0, 0), BETAS, TODAY)
    assert wipe.at_max_loss and wipe.projected_pnl == pytest.approx(-p.max_loss)
    win = reprice(p, ScenarioPreset("u", "u", 0.4, 0.0, 0), BETAS, TODAY)
    assert win.projected_pnl == pytest.approx(p.entry_credit * 100) and not win.at_max_loss
    assert reprice(p, ScenarioPreset("z", "z", 0, 0, 0), BETAS, TODAY).pnl_change == pytest.approx(0.0, abs=1e-9)


def test_contracts_scale_the_result_linearly():
    assert change(bull(3), BY["spy_down_3_iv5"]) == pytest.approx(3 * change(bull(1), BY["spy_down_3_iv5"]))


def test_betas_scale_each_underlying_move_and_unknown_betas_are_flagged():
    hi = change(bull(sym="QQQ"), BY["spy_down_3"], {"QQQ": 1.15})
    one = change(bull(sym="QQQ"), BY["spy_down_3"], {"QQQ": 1.0})
    assert hi < one < 0
    assert reprice(bull(sym="QQQ"), BY["spy_down_3"], {"QQQ": 1.15}, TODAY).beta_known is True
    unknown = reprice(bull(sym="XYZ"), BY["spy_down_3"], {}, TODAY)
    assert unknown.beta_known is False and unknown.pnl_change == pytest.approx(change(bull(), BY["spy_down_3"]))


def test_missing_inputs_are_excluded_and_flagged_not_guessed():
    no_iv = replace(bull(), short_iv=None)
    no_price = replace(bull(), underlying_price=None)
    down = BY["spy_down_3"]
    assert reprice(no_iv, down, BETAS, TODAY) is None and reprice(no_price, down, BETAS, TODAY) is None
    assert reprice(replace(bull(), long_iv=0.0), BY["spy_down_3"], BETAS, TODAY) is None
    res = run_scenario([bull(), no_iv], BY["spy_down_3"], BETAS, 20_000, TODAY)
    assert not res.complete and res.unavailable == [no_iv.key] and len(res.positions) == 1
    assert res.total_change == pytest.approx(change(bull(), BY["spy_down_3"]))          # unavailable one is excluded


def test_totals_worst_position_and_percent_of_equity():
    res = run_scenario([bull(), bear()], BY["spy_down_5"], BETAS, 20_000, TODAY)
    assert res.complete and res.total_change == pytest.approx(sum(p.pnl_change for p in res.positions))
    assert res.worst.spread_type == "BULL_PUT" and res.pct_of_equity == pytest.approx(res.total_change / 20_000)
    assert res.projected_total_pnl == pytest.approx(sum(p.projected_pnl for p in res.positions))
    assert run_scenario([], BY["spy_down_5"], BETAS, 20_000, TODAY).worst is None
    assert run_scenario([bull()], BY["spy_down_5"], BETAS, 0, TODAY).pct_of_equity is None


def test_presets_cover_the_requested_shocks():
    assert PRESET_KEYS == ("spy_down_3", "spy_down_3_iv5", "spy_down_5", "spy_up_3", "iv_up_10", "crash_10",
                           "week_passes")
    assert (BY["spy_down_3"].spot_move, BY["spy_down_5"].spot_move, BY["spy_up_3"].spot_move) == (-0.03, -0.05, 0.03)
    assert BY["iv_up_10"].iv_shift == 0.10 and BY["iv_up_10"].spot_move == 0 and BY["week_passes"].days == 5
    results = run_presets([bull(), bear()], BETAS, 20_000, TODAY)
    assert [r.preset.key for r in results] == list(PRESET_KEYS)
    by = {r.preset.key: r for r in results}
    assert by["crash_10"].total_change < by["spy_down_5"].total_change < by["spy_down_3"].total_change < 0
    bull_loss = next(p for p in by["crash_10"].positions if p.spread_type == "BULL_PUT").projected_pnl
    assert bull_loss < -0.5 * bull().max_loss                                            # most of the max loss is gone


def test_edge_cases_do_not_raise():
    near = bull(dte_days=2)                                                              # expires before 5 days pass
    r = reprice(near, BY["week_passes"], BETAS, TODAY)
    assert r.pnl_change >= 0                                                             # full decay of an OTM spread
    wild = reprice(bull(), ScenarioPreset("w", "w", -2.0, -5.0, 0), BETAS, TODAY)   # spot floors, vol floors
    assert wild.at_max_loss
    expired = reprice(bull(dte_days=-3), BY["spy_down_3"], BETAS, TODAY)
    assert expired is not None


def test_to_dict_shape():
    d = run_scenario([bull()], ScenarioPreset("wipe", "wipe", -0.4, 0.0, 0), BETAS, 20_000, TODAY).to_dict()
    assert {"key", "label", "total_change", "projected_total_pnl", "pct_of_equity", "at_max_loss", "complete",
            "unavailable", "worst", "positions"} <= d.keys()
    assert d["positions"][0]["at_max_loss"] is True and d["worst"]["key"] == bull().key


# --- service + API -------------------------------------------------------------------------------------------------
REAL = date.today()
FRIDAY = REAL + timedelta(days=(4 - REAL.weekday()) % 7 or 7) + timedelta(weeks=4)       # an expiry the mock lists


def legs(expiry):
    return [PositionLeg("SPY", expiry, 480, "P", -1, 2.00, 0.95), PositionLeg("SPY", expiry, 475, "P", 1, 1.00, 0.45)]


def service(expiry):
    settings = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False)
    broker = MockBroker(today=REAL, positions=legs(expiry))
    return ScanService(settings, broker, AlertDispatcher([]), scanner_cfg=ScannerConfig(top_n_per_symbol=3))


def test_service_computes_all_presets_with_leg_ivs_from_the_chain():
    svc = service(FRIDAY)
    asyncio.run(svc.run_once())
    pos = svc.state.positions[0]
    assert pos["short_iv"] and pos["long_iv"]                                            # captured from the quotes
    assert [s["key"] for s in svc.state.scenarios] == list(PRESET_KEYS)
    assert all(s["complete"] for s in svc.state.scenarios)
    by = {s["key"]: s for s in svc.state.scenarios}
    assert by["spy_down_5"]["total_change"] < by["spy_down_3"]["total_change"] < 0 < by["spy_up_3"]["total_change"]
    assert by["crash_10"]["total_change"] < by["spy_down_5"]["total_change"]
    with TestClient(create_app(svc)) as c:
        got = c.get("/api/scenarios").json()
        assert len(got) == 7 and got[0]["positions"][0]["key"] == pos["key"]


def test_position_without_a_chain_quote_makes_scenarios_incomplete_not_wrong():
    svc = service(REAL + timedelta(days=30))                                             # not a listed expiry -> no IV
    asyncio.run(svc.run_once())
    assert all(s["complete"] is False and s["positions"] == [] for s in svc.state.scenarios)
    assert svc.state.scenarios[0]["unavailable"] == [svc.state.positions[0]["key"]]


def test_no_positions_gives_empty_but_valid_scenarios():
    svc = ScanService(Settings(broker="mock", watchlist=["SPY"], scan_enabled=False), MockBroker(today=REAL),
                      AlertDispatcher([]), scanner_cfg=ScannerConfig(top_n_per_symbol=1))
    asyncio.run(svc.run_once())
    assert len(svc.state.scenarios) == 7 and all(s["total_change"] == 0 and s["complete"] for s in svc.state.scenarios)


def test_dashboard_html_has_the_scenario_controls():
    with TestClient(create_app(service(FRIDAY))) as c:
        html = c.get("/").text
    for marker in ('id="scpills"', 'id="scpos"', 'id="scall"', 'id="exp-scen"', '"scenarios"', "AT MAX LOSS",
                   "credit_spreads_scenarios_"):
        assert marker in html
