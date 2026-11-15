"""Paired-difference tables for results written by strategy_adjustments.py (delta/exit rows; universe via analyze_universe.py)."""
import json
import math
import sys
from collections import defaultdict
rows = json.load(open(sys.argv[1]))

def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None
def se(v):
    v = [x for x in v if x is not None]
    if len(v) < 2: return None
    m = sum(v) / len(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))
def f(x, d=2, pct=False):
    if x is None: return "n/a"
    return f"{x*100:.{d}f}%" if pct else f"{x:.{d}f}"
def pm(v, d=2, pct=False):
    m, s = mean(v), se(v)
    if m is None: return "n/a"
    return (f"{m*100:.{d}f}%" if pct else f"{m:.{d}f}") + (f" ±{(s*100 if pct else s):.{d}f}" if s is not None else "")

def table(group, variants, keys, base=None):
    # group: dict variant -> list of row dicts (by seed)
    hdr = ["variant"] + [k[0] for k in keys]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for v in variants:
        rs = sorted(group[v], key=lambda r: r["seed"])
        cells = [v]
        for name, key, d, pct in keys:
            cells.append(pm([r[key] for r in rs], d, pct))
        print("| " + " | ".join(cells) + " |")

def paired(group, base, variants, key, scale=1.0):
    out = {}
    b = {r["seed"]: r[key] for r in group[base]}
    for v in variants:
        diffs = [(r[key] - b[r["seed"]]) * scale for r in group[v] if r["seed"] in b and r[key] is not None and b[r["seed"]] is not None]
        better = sum(d > 0 for d in diffs)
        out[v] = (mean(diffs), se(diffs), better, len(diffs))
    return out

KEYS = [("trades/yr", "trades_per_year", 1, False), ("win rate", "win_rate", 1, True), ("stop-hit", "stop_rate", 1, True),
        ("target-hit", "target_rate", 1, True), ("credit/width", "credit_over_width", 1, True), ("avg |Δ|", "short_delta", 3, False),
        ("avg win $", "avg_win", 0, False), ("avg loss $", "avg_loss", 0, False), ("net/trade % of risk", "mean_pct_risk", 2, True),
        ("Sharpe", "sharpe", 2, False), ("max DD $", "max_drawdown", 0, False), ("5th pct trade $", "p05_trade", 0, False),
        ("worst $", "worst", 0, False), ("days held", "avg_hold_days", 1, False), ("PF", "profit_factor", 2, False)]

d = [r for r in rows if r["exp"] == "delta_exit"]
if d:
    for vrp in sorted({r["vrp"] for r in d}):
        g = defaultdict(list)
        for r in d:
            if r["vrp"] == vrp: g[r["variant"]].append(r)
        order = ["d20-25|t50 (current)", "d15-20|t50", "d12-15|t50", "d20-25|t30", "d12-15|t30", "d20-25|t40", "d20-25|t65",
                 "d20-25|t80", "d12-15|t65", "d20-25|t50|disciplined floors", "d12-15|t50|disciplined floors"]
        print(f"\n### Implied-vol premium over realized = {vrp:+.0%}  (SPY-like, mean ± s.e. over {len(g[order[0]])} seeds)\n")
        table(g, order, KEYS)
        print("\nPaired difference vs `d20-25|t50 (current)` (same price paths):\n")
        print("| variant | Δ Sharpe | seeds better | Δ net/trade % risk | Δ win rate | Δ stop-hit | Δ trades/yr |")
        print("|---|---|---|---|---|---|---|")
        base = order[0]
        for v in order[1:]:
            sh = paired(g, base, [v], "sharpe")[v]; np_ = paired(g, base, [v], "mean_pct_risk", 100)[v]
            wr = paired(g, base, [v], "win_rate", 100)[v]; st = paired(g, base, [v], "stop_rate", 100)[v]; tr = paired(g, base, [v], "trades_per_year")[v]
            fm = lambda t, dgt=2: "n/a" if t[0] is None else f"{t[0]:+.{dgt}f}" + (f" ±{t[1]:.{dgt}f}" if t[1] is not None else "")
            print(f"| {v} | {fm(sh)} | {sh[2]}/{sh[3]} | {fm(np_)} pp | {fm(wr,1)} pp | {fm(st,1)} pp | {fm(tr,1)} |")

u = [r for r in rows if r["exp"] == "universe"]
if u:
    g = defaultdict(list)
    for r in u: g[r["variant"]].append(r)
    order = ["SPY-like ETF", "QQQ-like ETF", "High-IV index (IWM/RUT-like)", "Stock, VRP 0% (earnings gate ON)", "Stock, VRP 0% (NO earnings gate)",
             "Stock, VRP +10% (earnings gate ON)", "Stock, VRP +10% (NO earnings gate)", "Stock, VRP +20% (earnings gate ON)", "Stock, VRP +20% (NO earnings gate)"]
    print(f"\n### Universe (delta 0.15-0.25, 50% target, mean ± s.e. over {len(g[order[0]])} seeds)\n")
    keys = [("trades/yr", "trades_per_year", 1, False), ("win rate", "win_rate", 1, True), ("stop-hit", "stop_rate", 1, True),
            ("credit/width", "credit_over_width", 1, True), ("net/trade % of risk", "mean_pct_risk", 2, True), ("Sharpe", "sharpe", 2, False),
            ("5th pct trade $", "p05_trade", 0, False), ("worst $", "worst", 0, False), ("cost share of gross", "cost_share", 2, True), ("PF", "profit_factor", 2, False)]
    table(g, order, keys)
    print("\nEarnings gate value (gate ON minus gate OFF, paired by seed):\n")
    print("| premium | Δ Sharpe | Δ net/trade % risk | Δ stop-hit | Δ worst trade $ | trades/yr ON vs OFF |"); print("|---|---|---|---|---|---|")
    for vrp in ("0%", "+10%", "+20%"):
        on, off = f"Stock, VRP {vrp} (earnings gate ON)", f"Stock, VRP {vrp} (NO earnings gate)"
        sh = paired(g, off, [on], "sharpe")[on]; np_ = paired(g, off, [on], "mean_pct_risk", 100)[on]; st = paired(g, off, [on], "stop_rate", 100)[on]
        wr = paired(g, off, [on], "worst")[on]
        fm = lambda t, dgt=2: "n/a" if t[0] is None else f"{t[0]:+.{dgt}f}" + (f" ±{t[1]:.{dgt}f}" if t[1] is not None else "")
        print(f"| {vrp} | {fm(sh)} ({sh[2]}/{sh[3]} better) | {fm(np_)} pp | {fm(st,1)} pp | {fm(wr,0)} | {mean([r['trades_per_year'] for r in g[on]]):.1f} vs {mean([r['trades_per_year'] for r in g[off]]):.1f} |")
