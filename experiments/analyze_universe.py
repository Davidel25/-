"""Universe / friction / check tables: python experiments/analyze_universe.py results_universe.json"""
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
def pm(v, d=2, pct=False):
    m, s = mean(v), se(v)
    if m is None: return "n/a"
    k = 100 if pct else 1
    return f"{m*k:.{d}f}{'%' if pct else ''}" + (f" ±{s*k:.{d}f}" if s is not None else "")
g = defaultdict(list)
for r in rows: g[r["variant"]].append(r)
order = list(dict.fromkeys(r["variant"] for r in sorted(rows, key=lambda r: (r["seed"],))))
keys = [("trades/yr", "trades_per_year", 1, False), ("win rate", "win_rate", 1, True), ("stop-hit", "stop_rate", 1, True), ("credit/width", "credit_over_width", 1, True),
        ("net/trade % risk", "mean_pct_risk", 2, True), ("Sharpe", "sharpe", 2, False), ("5th pct $", "p05_trade", 0, False), ("worst $", "worst", 0, False),
        ("max DD $", "max_drawdown", 0, False), ("cost/gross", "cost_share", 2, True), ("PF", "profit_factor", 2, False)]
print("| variant | " + " | ".join(k[0] for k in keys) + " |"); print("|" + "---|" * (len(keys) + 1))
for v in order:
    print(f"| {v} | " + " | ".join(pm([r[k] for r in g[v]], d, p) for _, k, d, p in keys) + " |")
def paired(a, b, key, scale=1.0):
    ba = {r["seed"]: r[key] for r in g[b]}
    d = [(r[key] - ba[r["seed"]]) * scale for r in g[a] if r["seed"] in ba and r[key] is not None and ba[r["seed"]] is not None]
    m, s = mean(d), se(d)
    return (f"{m:+.2f} ±{s:.2f}" if m is not None and s is not None else "n/a"), sum(x > 0 for x in d), len(d)
pairs = [("SPY-like ETF (vol 16%, VRP +10%)", "QQQ-like ETF (vol 22%, VRP +10%)"),
         ("High-vol index (vol 24%, SAME VRP +10%)", "SPY-like ETF (vol 16%, VRP +10%)"),
         ("High-vol index (vol 24%, HIGHER VRP +14%)", "High-vol index (vol 24%, SAME VRP +10%)"),
         ("Stock: 1.5% half-spread, VRP +10% (gate ON)", "Stock: 0.5% half-spread, VRP +10% (gate ON)"),
         ("Stock: 3.0% half-spread, VRP +10% (gate ON)", "Stock: 1.5% half-spread, VRP +10% (gate ON)"),
         ("Stock: 1.5% half-spread, VRP +10%, news gaps UNPRICED (gate ON)", "Stock: 1.5% half-spread, VRP +10% (gate ON)"),
         ("Stock: 1.5% half-spread, VRP +10% (gate ON)", "Stock: 1.5% half-spread, VRP +10% (NO earnings gate)"),
         ("Stock: 1.5% half-spread, VRP +10% (gate ON)", "Stock: 1.5% half-spread, VRP 0% (gate ON)"),
         ("Stock: 0.5% half-spread, VRP +20% (gate ON)", "Stock: 0.5% half-spread, VRP +10% (gate ON)")]
print("\nPaired differences (A minus B, same price paths where the profile matches; otherwise independent paths)\n")
print("| A | B | Δ Sharpe (A>B seeds) | Δ net/trade % risk | Δ worst trade $ |"); print("|---|---|---|---|---|")
for a, b in pairs:
    sh = paired(a, b, "sharpe"); np_ = paired(a, b, "mean_pct_risk", 100); w = paired(a, b, "worst")
    print(f"| {a} | {b} | {sh[0]} ({sh[1]}/{sh[2]}) | {np_[0]} pp | {w[0]} |")
