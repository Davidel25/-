"""Parameter sweep on SYNTHETIC SPY+QQQ chains, designed to resist over-fitting.

A naive "best of N configurations" is biased upward (winner's curse) and, on synthetic data, finds what suits the simulator.
So this runs in three stages:

  search    random sample of the full grid, each config scored on the same synthetic datasets (4 seeds x 2 premium levels)
  analyze   main effects (OLS with dataset fixed effects: robust to noise) + training ranking + the predicted-best combination
  validate  finalists and the CURRENT configuration on FRESH seeds; only these numbers are used for the final ranking

    python experiments/optimize.py search   search.json   [n_configs=360]
    python experiments/optimize.py analyze  search.json   finalists.json
    python experiments/optimize.py validate finalists.json validation.json
    python experiments/optimize.py report   search.json   validation.json

Grid (1,440 combinations): short-delta band centre {0.10, 0.12, 0.15, 0.18, 0.20} +/- 0.015 | credit/ROC floors
{.30/8%, .45/12%, .60/20%} | profit target {50, 60, 65, 70}% | stop multiple {1.5, 2.0, 2.5}x credit |
entry DTE window {30-45, 45-60} | mandatory early exit at DTE {7, 10, 14, 21}.
Everything else is the live `ScannerConfig.disciplined()` / `RiskLimits.disciplined()` rule set on a $20k account, natural
fills, $0.65 commission per contract per leg."""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.compare import variant_stats  # noqa: E402
from src.backtest.data import from_frame  # noqa: E402
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.backtest.synthetic import SymbolProfile, generate_universe  # noqa: E402
from src.risk.manager import RiskLimits  # noqa: E402
from src.strategies.credit_spreads import ScannerConfig  # noqa: E402

EQUITY = 20_000.0
DAYS = 800
DELTAS = (0.10, 0.12, 0.15, 0.18, 0.20)
HALF = 0.015
FLOORS = {"low": (0.30, 0.08), "mid": (0.45, 0.12), "high": (0.60, 0.20)}
TARGETS = (0.50, 0.60, 0.65, 0.70)
STOPS = (1.5, 2.0, 2.5)
WINDOWS = {"30-45": (30, 45, (35, 45)), "45-60": (45, 60, (50, 60))}
EXIT_DTES = (7, 10, 14, 21)
GRID_KEYS = ("delta", "floors", "target", "stop", "window", "exit_dte")
MIN_TRADES_PER_YEAR = 12.0      # configs trading less than this are not eligible for ranking (tiny samples flatter drawdown)
TRAIN_SEEDS = (101, 102, 103, 104)
VALID_SEEDS = tuple(range(201, 213))
PREMIA = (0.10, 0.20)


def full_grid() -> list[dict]:
    return [dict(zip(GRID_KEYS, v, strict=True))
            for v in itertools.product(DELTAS, FLOORS, TARGETS, STOPS, WINDOWS, EXIT_DTES)]


def config_id(p: dict) -> str:
    if "_name" in p:                       # a named variant of CURRENT (see ONE_AT_A_TIME)
        return p["_name"]
    return f"d{p['delta']:.2f}|{p['floors']}|t{int(p['target'] * 100)}|s{p['stop']:g}|{p['window']}|x{p['exit_dte']}"


# CURRENT with exactly one thing changed (or the "let winners run" bundle), to read each parameter's effect directly.
ONE_AT_A_TIME = {
    "CURRENT+stop2.5": {"limits": {"stop_loss_multiple": 2.5}},
    "CURRENT+target70": {"limits": {"profit_target_pct": 0.70}},
    "CURRENT+exit7": {"limits": {"close_at_dte": 7}},
    "CURRENT+window45-60": {"scanner": {"min_dte": 45, "max_dte": 60, "preferred_dte": (50, 60)}},
    "CURRENT+stop2.5+target70+exit7": {"limits": {"stop_loss_multiple": 2.5, "profit_target_pct": 0.70, "close_at_dte": 7}},
}


def build(p: dict | None) -> BacktestConfig:
    """``None`` = the CURRENT adopted configuration (ScannerConfig.disciplined() / RiskLimits.disciplined())."""
    if p is None:
        return BacktestConfig(equity=EQUITY, max_new_per_day=2)
    if "_name" in p:
        base = build(None)
        return replace(base, scanner=replace(base.scanner, **p.get("scanner", {})),
                       limits=replace(base.limits, **p.get("limits", {})))
    lo, hi, pref = WINDOWS[p["window"]]
    credit, roc = FLOORS[p["floors"]]
    scanner = ScannerConfig.disciplined(min_dte=lo, max_dte=hi, preferred_dte=pref, min_short_delta=p["delta"] - HALF,
                                        max_short_delta=p["delta"] + HALF, min_credit=credit, min_roc=roc)
    limits = RiskLimits.disciplined(profit_target_pct=p["target"], stop_loss_multiple=p["stop"],
                                    close_at_dte=p["exit_dte"])
    return BacktestConfig(scanner=scanner, limits=limits, equity=EQUITY, max_new_per_day=2)


def dataset(vrp: float, seed: int):
    profs = {"SPY": SymbolProfile(real_vol=0.16, vrp=vrp, spot0=500, half_spread_pct=0.004),
             "QQQ": SymbolProfile(real_vol=0.22, vrp=vrp, spot0=430, half_spread_pct=0.005)}
    return from_frame(generate_universe(profs, DAYS, seed, max_dte=65)[0])


def run_one(store, p: dict | None) -> dict:
    return variant_stats("x", run_backtest(store, build(p))).flat()


def search_job(args):
    vrp, seed, params = args
    store = dataset(vrp, seed)
    rows = []
    for p in [None, *params]:
        rows.append({"id": "CURRENT" if p is None else config_id(p), "params": p, "vrp": vrp, "seed": seed,
                     **run_one(store, p)})
    return rows


def run_pool(jobs, fn, label):
    rows, t0 = [], time.time()
    with Pool(4) as pool:
        for part in pool.imap_unordered(fn, jobs):
            rows += part
            print(f"{label}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------------- analysis
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def se(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


def aggregate(rows):
    """Per-config means over datasets, with eligibility (enough trades and a defined Sharpe in most datasets)."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    out = {}
    for cid, rs in by.items():
        sh = [r["sharpe"] for r in rs if r["sharpe"] is not None]
        tpy = mean([r["trades_per_year"] for r in rs]) or 0.0
        out[cid] = {"id": cid, "params": rs[0]["params"], "n": len(rs), "trades_per_year": tpy,
                    "sharpe": mean(sh), "sharpe_se": se(sh), "net_trade": mean([r["mean_net"] for r in rs]),
                    "pct_risk": mean([r["mean_pct_risk"] for r in rs]), "max_dd": mean([r["max_drawdown"] for r in rs]),
                    "win": mean([r["win_rate"] for r in rs]), "stop": mean([r["stop_rate"] for r in rs]),
                    "worst": mean([r["worst"] for r in rs]), "annual": mean([(r["annualized_return"] or 0) * EQUITY for r in rs]),
                    "eligible": tpy >= MIN_TRADES_PER_YEAR and len(sh) >= 0.8 * len(rs)}
    return out


def main_effects(rows):
    """OLS of per-run Sharpe on dataset fixed effects + (delta x floors) cell + target + stop + window + exit DTE.
    Returns {factor: {level: (effect, se)}} relative to each factor's first level, plus the fitted model."""
    import numpy as np
    data = [r for r in rows if r["params"] and r["sharpe"] is not None and r["trades"] >= 8]
    cols: list[tuple[str, object]] = []
    levels = {"cell": sorted({(r["params"]["delta"], r["params"]["floors"]) for r in data}),
              "target": TARGETS, "stop": STOPS, "window": tuple(WINDOWS), "exit_dte": EXIT_DTES}
    for fac, lv in levels.items():
        cols += [(fac, x) for x in lv[1:]]
    datasets = sorted({(r["vrp"], r["seed"]) for r in data})
    cols += [("ds", d) for d in datasets[1:]]

    def key(r, fac):
        p = r["params"]
        return (p["delta"], p["floors"]) if fac == "cell" else p[fac]

    X = np.zeros((len(data), len(cols) + 1))
    X[:, 0] = 1.0
    for j, (fac, x) in enumerate(cols, start=1):
        for i, r in enumerate(data):
            X[i, j] = 1.0 if ((r["vrp"], r["seed"]) == x if fac == "ds" else key(r, fac) == x) else 0.0
    y = np.array([r["sharpe"] for r in data])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid / dof)
    eff: dict = {}
    for j, (fac, x) in enumerate(cols, start=1):
        if fac != "ds":
            eff.setdefault(fac, {})[x] = (float(beta[j]), float(math.sqrt(max(cov[j, j], 0))))
    for fac, lv in levels.items():
        eff.setdefault(fac, {})[lv[0]] = (0.0, 0.0)
    return eff, len(data)


def predicted_best(eff, agg):
    """Best combination under the additive model, among cells that actually traded enough in the search."""
    cells = {(a["params"]["delta"], a["params"]["floors"]) for a in agg.values() if a["params"] and a["eligible"]}
    best = None
    for cell in cells:
        for t, s, w, x in itertools.product(TARGETS, STOPS, WINDOWS, EXIT_DTES):
            score = (eff["cell"].get(cell, (0, 0))[0] + eff["target"][t][0] + eff["stop"][s][0]
                     + eff["window"][w][0] + eff["exit_dte"][x][0])
            if best is None or score > best[0]:
                best = (score, {"delta": cell[0], "floors": cell[1], "target": t, "stop": s, "window": w, "exit_dte": x})
    return best


def cmd_search(out: str, n: int):
    grid = full_grid()
    sample = random.Random(7).sample(grid, n)
    jobs = [(v, s, sample) for v in PREMIA for s in TRAIN_SEEDS]
    rows = run_pool(jobs, search_job, "search")
    json.dump(rows, open(out, "w"))
    print("DONE", len(rows))


def fmt(x, d=2, pct=False):
    return "n/a" if x is None else (f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}")


def cmd_analyze(path: str, out: str):
    rows = json.load(open(path))
    agg = aggregate(rows)
    elig = [a for a in agg.values() if a["params"] and a["eligible"]]
    print(f"{len(agg) - 1} configs searched, {len(elig)} eligible (>= {MIN_TRADES_PER_YEAR:g} trades/yr, Sharpe defined in 80%+ of datasets)")
    cur = agg["CURRENT"]
    print(f"CURRENT (training): Sharpe {fmt(cur['sharpe'])} +/- {fmt(cur['sharpe_se'])}, net/trade ${fmt(cur['net_trade'], 1)}, "
          f"maxDD ${fmt(cur['max_dd'], 0)}, trades/yr {fmt(cur['trades_per_year'], 1)}")
    eff, n = main_effects(rows)
    print(f"\nMain effects on per-run Sharpe (OLS, dataset fixed effects, {n} runs), vs first level of each factor:")
    for fac, lv in eff.items():
        line = ", ".join(f"{k}: {v[0]:+.2f}±{v[1]:.2f}" for k, v in lv.items())
        print(f"  {fac}: {line}")
    pb = predicted_best(eff, agg)
    print(f"\nAdditive-model best combination (predicted Sharpe gain {pb[0]:+.2f} over baseline cell): {config_id(pb[1])}")
    top = sorted(elig, key=lambda a: -a["sharpe"])[:8]
    print("\nTop 8 by TRAINING Sharpe (inflated by selection; validated next):")
    for a in top:
        print(f"  {a['id']:<38} Sharpe {fmt(a['sharpe'])}±{fmt(a['sharpe_se'])}  net/trade ${fmt(a['net_trade'], 1)}  maxDD ${fmt(a['max_dd'], 0)}  tr/yr {fmt(a['trades_per_year'], 1)}")
    fin: dict[str, dict] = {}
    for a in top + sorted(elig, key=lambda a: -(a["net_trade"] or -1e9))[:3] + sorted(elig, key=lambda a: a["max_dd"])[:3]:
        fin[a["id"]] = a["params"]
    fin[config_id(pb[1])] = pb[1]
    json.dump({"finalists": fin, "effects": {k: {str(l): v for l, v in lv.items()} for k, lv in eff.items()}}, open(out, "w"))
    print(f"\n{len(fin)} finalists (+CURRENT) written to {out}")


def validate_job(args):
    vrp, seed, params = args
    store = dataset(vrp, seed)
    return [{"id": "CURRENT" if p is None else config_id(p), "params": p, "vrp": vrp, "seed": seed, **run_one(store, p)}
            for p in params]


def cmd_validate(path: str, out: str):
    fin = json.load(open(path))["finalists"]
    params = [None, *fin.values(), *({"_name": n, **ov} for n, ov in ONE_AT_A_TIME.items())]
    jobs = [(v, s, params) for v in PREMIA for s in VALID_SEEDS]
    rows = run_pool(jobs, validate_job, "validate")
    json.dump(rows, open(out, "w"))
    print("DONE", len(rows))


def paired(rows, cid, base="CURRENT", key="sharpe", scale=1.0):
    b = {(r["vrp"], r["seed"]): r[key] for r in rows if r["id"] == base}
    d = [(r[key] - b[(r["vrp"], r["seed"])]) * scale for r in rows
         if r["id"] == cid and r[key] is not None and b.get((r["vrp"], r["seed"])) is not None]
    return mean(d), se(d), sum(x > 0 for x in d), len(d)


def cmd_report(search_path: str, valid_path: str):
    train = aggregate(json.load(open(search_path)))
    vrows = json.load(open(valid_path))
    val = aggregate(vrows)
    elig = [a for a in val.values() if a["eligible"] and a["id"] != "CURRENT"]

    def table(title, items):
        print(f"\n### {title}\n")
        print("| # | Config | Valid. Sharpe | Net/trade $ | Max DD $ | Trades/yr | Win | Stop-hit | Train Sharpe | Δ Sharpe vs CURRENT (seeds better) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for i, a in enumerate(items, 1):
            t = train.get(a["id"], {})
            dm, ds, nb, nn = paired(vrows, a["id"])
            print(f"| {i} | `{a['id']}` | {fmt(a['sharpe'])} ± {fmt(a['sharpe_se'])} | {fmt(a['net_trade'], 1)} | {fmt(a['max_dd'], 0)} | "
                  f"{fmt(a['trades_per_year'], 1)} | {fmt(a['win'], 0, True)} | {fmt(a['stop'], 0, True)} | {fmt(t.get('sharpe'))} | "
                  f"{dm:+.2f} ± {ds:.2f} ({nb}/{nn}) |")

    table("Top 3 by validation Sharpe", sorted(elig, key=lambda a: -a["sharpe"])[:3])
    table("Top 3 by validation net P&L per trade", sorted(elig, key=lambda a: -a["net_trade"])[:3])
    table("Top 3 by validation max drawdown (lowest)", sorted(elig, key=lambda a: a["max_dd"])[:3])
    cur = val["CURRENT"]
    print(f"\nCURRENT (validation): Sharpe {fmt(cur['sharpe'])} ± {fmt(cur['sharpe_se'])}, net/trade ${fmt(cur['net_trade'], 1)}, "
          f"max DD ${fmt(cur['max_dd'], 0)}, trades/yr {fmt(cur['trades_per_year'], 1)}")
    for vrp in PREMIA:
        sub = aggregate([r for r in vrows if r["vrp"] == vrp])
        best = max((a for a in sub.values() if a["eligible"] and a["id"] != "CURRENT"), key=lambda a: a["sharpe"])
        print(f"  premium {vrp:+.0%}: CURRENT Sharpe {fmt(sub['CURRENT']['sharpe'])}; best finalist {best['id']} {fmt(best['sharpe'])}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        cmd_search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 360)
    elif cmd == "analyze":
        cmd_analyze(sys.argv[2], sys.argv[3])
    elif cmd == "validate":
        cmd_validate(sys.argv[2], sys.argv[3])
    elif cmd == "report":
        cmd_report(sys.argv[2], sys.argv[3])
