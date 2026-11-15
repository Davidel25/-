"""The parameter-sweep tooling (experiments/optimize.py): grid, config mapping, aggregation, main-effects regression."""
import importlib.util
import random
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "optimize.py"
spec = importlib.util.spec_from_file_location("optimize", SCRIPT)
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


def test_the_grid_is_the_full_factorial_with_unique_ids():
    grid = opt.full_grid()
    assert len(grid) == 5 * 3 * 4 * 3 * 2 * 4 == 1440
    assert len({opt.config_id(p) for p in grid}) == 1440
    p = {"delta": 0.12, "floors": "mid", "target": 0.65, "stop": 2.0, "window": "30-45", "exit_dte": 14}
    assert opt.config_id(p) == "d0.12|mid|t65|s2|30-45|x14"
    assert set(grid[0]) == set(opt.GRID_KEYS)


def test_a_grid_point_maps_onto_the_real_scanner_and_risk_limits():
    cfg = opt.build({"delta": 0.15, "floors": "high", "target": 0.70, "stop": 1.5, "window": "45-60", "exit_dte": 21})
    s, lim = cfg.scanner, cfg.limits
    assert (round(s.min_short_delta, 3), round(s.max_short_delta, 3)) == (0.135, 0.165)
    assert (s.min_credit, s.min_roc) == (0.60, 0.20) and (s.min_dte, s.max_dte, s.preferred_dte) == (45, 60, (50, 60))
    assert (lim.profit_target_pct, lim.stop_loss_multiple, lim.close_at_dte) == (0.70, 1.5, 21)
    assert lim.max_risk_per_trade_pct == 0.10 and cfg.equity == 20_000.0 and cfg.max_new_per_day == 2
    assert s.pricing == "natural" and s.min_iv_hv_ratio == 1.1          # the live edge filters stay on
    assert s.require_trend_alignment


def test_current_is_the_adopted_configuration_unchanged():
    cur = opt.build(None)
    assert (cur.scanner.min_short_delta, cur.scanner.max_short_delta) == (0.12, 0.15)
    got = (cur.scanner.min_credit, cur.scanner.min_roc, cur.scanner.min_dte, cur.scanner.max_dte)
    assert got == (0.45, 0.12, 30, 50)
    assert (cur.limits.profit_target_pct, cur.limits.stop_loss_multiple, cur.limits.close_at_dte) == (0.65, 2.0, 14)


def row(cid, params, vrp, seed, sharpe, trades=30, tpy=30.0, net=5.0, dd=500.0):
    return {"id": cid, "params": params, "vrp": vrp, "seed": seed, "sharpe": sharpe, "trades": trades,
            "trades_per_year": tpy, "mean_net": net, "mean_pct_risk": 0.01, "max_drawdown": dd, "win_rate": 0.75,
            "stop_rate": 0.15, "worst": -300.0, "annualized_return": 0.01}


P = {"delta": 0.12, "floors": "mid", "target": 0.65, "stop": 2.0, "window": "30-45", "exit_dte": 14}


def test_aggregation_averages_and_applies_the_eligibility_rules():
    rows = [row("a", P, 0.1, s, sh) for s, sh in enumerate((0.1, 0.3, 0.2, 0.4))]
    a = opt.aggregate(rows)["a"]
    assert a["sharpe"] == pytest.approx(0.25) and a["sharpe_se"] > 0 and a["eligible"] and a["n"] == 4
    thin = opt.aggregate([row("b", P, 0.1, s, 0.5, tpy=5.0) for s in range(4)])["b"]
    assert not thin["eligible"]                                           # too few trades per year
    undefined = opt.aggregate([row("c", P, 0.1, s, None if s else 0.5) for s in range(4)])["c"]
    assert not undefined["eligible"]                                      # Sharpe undefined in most datasets


def test_paired_difference_uses_the_same_datasets():
    rows = [row("CURRENT", None, v, s, 0.0) for v in (0.1, 0.2) for s in range(3)]
    rows += [row("x", P, v, s, 0.2 + 0.1 * s) for v in (0.1, 0.2) for s in range(3)]
    mean, se, better, n = opt.paired(rows, "x")
    assert mean == pytest.approx(0.3) and better == 6 and n == 6 and se > 0
    assert opt.paired(rows, "x", key="mean_net")[0] == 0.0 and opt.paired([], "x")[3] == 0


def synthetic_truth(noise=0.02, seed=1):
    """Rows whose Sharpe is a known additive function plus dataset luck, to test that the regression recovers it."""
    rng = random.Random(seed)
    truth = {"cell": {(0.12, "mid"): 0.30, (0.15, "mid"): 0.10, (0.12, "low"): 0.0, (0.15, "low"): -0.05},
             "target": {0.5: 0.0, 0.6: 0.05, 0.65: 0.20, 0.7: 0.10}, "stop": {1.5: -0.10, 2.0: 0.0, 2.5: 0.05},
             "window": {"30-45": 0.0, "45-60": -0.15}, "exit_dte": {7: -0.08, 10: 0.0, 14: 0.12, 21: 0.04}}
    luck = {(v, s): rng.gauss(0, 0.4) for v in (0.1, 0.2) for s in range(4)}               # big dataset-level noise
    rows = []
    for cell in truth["cell"]:
        for t in truth["target"]:
            for st in truth["stop"]:
                for w in truth["window"]:
                    for x in truth["exit_dte"]:
                        p = {"delta": cell[0], "floors": cell[1], "target": t, "stop": st, "window": w, "exit_dte": x}
                        base = (truth["cell"][cell] + truth["target"][t] + truth["stop"][st] + truth["window"][w]
                                + truth["exit_dte"][x])
                        for (v, s), lk in luck.items():
                            rows.append(row(opt.config_id(p), p, v, s, base + lk + rng.gauss(0, noise)))
    return rows, truth


def test_main_effects_recover_planted_effects_despite_large_dataset_noise():
    rows, truth = synthetic_truth()
    eff, n = opt.main_effects(rows)
    assert n == len(rows)
    for fac in ("target", "stop", "window", "exit_dte"):
        levels = list(truth[fac])
        ref = truth[fac][levels[0]]
        for lv in levels:
            assert eff[fac][lv][0] == pytest.approx(truth[fac][lv] - ref, abs=0.04), (fac, lv)
    cells = list(truth["cell"])
    for cell in cells:
        assert eff["cell"][cell][0] == pytest.approx(truth["cell"][cell] - truth["cell"][sorted(cells)[0]], abs=0.04)


def test_predicted_best_picks_the_planted_optimum_and_ignores_ineligible_cells():
    rows, truth = synthetic_truth()
    eff, _ = opt.main_effects(rows)
    agg = opt.aggregate(rows)
    score, best = opt.predicted_best(eff, agg)
    assert (best["delta"], best["floors"], best["target"], best["stop"], best["window"], best["exit_dte"]) == \
        (0.12, "mid", 0.65, 2.5, "30-45", 14)
    for a in agg.values():                                                              # make the best cell untradeable
        if (a["params"]["delta"], a["params"]["floors"]) == (0.12, "mid"):
            a["eligible"] = False
    assert opt.predicted_best(eff, agg)[1]["floors"] != "mid" or opt.predicted_best(eff, agg)[1]["delta"] != 0.12


def test_report_prints_ranked_tables_and_the_current_reference(tmp_path, capsys):
    rows, _ = synthetic_truth()
    cur = [row("CURRENT", None, v, s, 0.05) for v in (0.1, 0.2) for s in range(4)]
    import json
    (tmp_path / "t.json").write_text(json.dumps(rows + cur))
    (tmp_path / "v.json").write_text(json.dumps(rows + cur))
    opt.cmd_report(str(tmp_path / "t.json"), str(tmp_path / "v.json"))
    out = capsys.readouterr().out
    assert "Top 3 by validation Sharpe" in out and "Top 3 by validation net P&L per trade" in out
    assert "Top 3 by validation max drawdown" in out and "CURRENT (validation)" in out and "d0.12|mid|t65" in out
