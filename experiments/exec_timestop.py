"""Paired grid: optional time-stops and limit-order (mid-fill) execution, on top of the sweep's recommended set.

Every variant runs on the SAME fresh synthetic datasets (12 unseen seeds x 2 implied-vol premia, SPY+QQQ, 800 days),
so differences vs BASE are paired. SYNTHETIC evidence only; the fill-probability and adverse-selection inputs are
ASSUMPTIONS (daily data cannot calibrate them).

    python experiments/exec_timestop.py run  out.json      # ~10-20 min on 4 cores
    python experiments/exec_timestop.py report out.json     # markdown tables
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optimize as opt  # noqa: E402

from src.backtest.compare import variant_stats  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402

SEEDS = tuple(range(301, 313))          # unseen by the sweep (training 101-104, validation 201-212)
PREMIA = (0.10, 0.20)
RECO = {"delta": 0.15, "floors": "low", "target": 0.70, "stop": 2.5, "window": "45-60", "exit_dte": 7}


def variants() -> dict[str, dict]:
    v: dict[str, dict] = {"CURRENT": {"cur": True}, "BASE": {}}
    for frac in (0.4, 0.5, 0.6):
        for minp in (0.0, 0.10, 0.20, 0.30):
            v[f"TS f{frac:g} p{int(minp * 100)}"] = {"limits": {"time_stop_fraction": frac,
                                                                "time_stop_min_profit_pct": minp}}
    v["FILL entry.5"] = {"bt": {"entry_fill_improvement": 0.5}}
    v["FILL exit.5"] = {"bt": {"exit_fill_improvement": 0.5}}
    for f in (0.25, 0.5, 1.0):
        for prob in (1.0, 0.6):
            for hair in (0.0, 0.5):
                v[f"FILL f{f:g} prob{prob:g} adv{hair:g}"] = {"bt": {
                    "entry_fill_improvement": f, "exit_fill_improvement": f, "fill_prob_at_mid": prob,
                    "adverse_selection_pct": hair}}
    return v


def make(spec: dict):
    cfg = opt.build(None) if spec.get("cur") else opt.build(RECO)
    return replace(replace(cfg, limits=replace(cfg.limits, **spec.get("limits", {}))), **spec.get("bt", {}))


def job(args):
    vrp, seed = args
    store = opt.dataset(vrp, seed)
    return [{"id": name, "vrp": vrp, "seed": seed, **variant_stats(name, run_backtest(store, make(spec))).flat()}
            for name, spec in variants().items()]


def cmd_run(out: str) -> None:
    rows = []
    with Pool(4) as pool:
        for part in pool.imap_unordered(job, [(p, s) for p in PREMIA for s in SEEDS]):
            rows += part
            print(f"{len(rows)} rows", flush=True)
    json.dump(rows, open(out, "w"))
    print("DONE")


def _m(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def _diff(rows, vid, key, per=None):
    base = {(r["vrp"], r["seed"]): r for r in rows if r["id"] == "BASE"}
    d = []
    for r in rows:
        if r["id"] != vid or (per is not None and r["vrp"] != per):
            continue
        b = base[(r["vrp"], r["seed"])]
        if r.get(key) is not None and b.get(key) is not None:
            d.append(r[key] - b[key])
    if len(d) < 2:
        return None, None, 0, len(d)
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return m, se, sum(x > 0 for x in d), len(d)


def cmd_report(path: str) -> None:
    rows = json.load(open(path))
    ids = list(dict.fromkeys(r["id"] for r in rows))

    def col(vid, key):
        return _m([r[key] for r in rows if r["id"] == vid])

    def annual(vid):
        return _m([(r["annualized_return"] or 0) * opt.EQUITY for r in rows if r["id"] == vid])

    for r in rows:
        r["annual"] = (r["annualized_return"] or 0) * opt.EQUITY
    print("| variant | Sharpe | dSharpe vs BASE (+/-SE, better/n) | d Sharpe @+10% | d Sharpe @+20% | net $/trade | d annual $ "
          "| max DD $ | trades/yr | avg hold d |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for vid in ids:
        dm, ds, nb, n = _diff(rows, vid, "sharpe")
        d10 = _diff(rows, vid, "sharpe", 0.10)[0]
        d20 = _diff(rows, vid, "sharpe", 0.20)[0]
        da = _diff(rows, vid, "annual")[0]
        f = lambda x, k=2: "n/a" if x is None else f"{x:.{k}f}"  # noqa: E731
        cell = "-" if vid == "BASE" else f"{dm:+.2f} ± {ds:.2f} ({nb}/{n})" if dm is not None else "n/a"
        print(f"| {vid} | {f(col(vid, 'sharpe'))} | {cell} | {f(d10)} | {f(d20)} | {f(col(vid, 'mean_net'), 1)} "
              f"| {f(da, 0)} | {f(col(vid, 'max_drawdown'), 0)} | {f(col(vid, 'trades_per_year'), 1)} "
              f"| {f(col(vid, 'avg_hold_days'), 1)} |")
    bound = _diff(rows, "FILL f1 prob1 adv0", "annual")[0]
    if bound:
        print(f"\nFriction-recovery bound (entry+exit at the mid, always filled): {bound:+.0f} $/yr vs BASE. "
              "Share of it captured:")
        for vid in ids:
            if vid.startswith("FILL"):
                print(f"  {vid}: {(_diff(rows, vid, 'annual')[0] or 0) / bound:.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2])
    else:
        cmd_report(sys.argv[2])
