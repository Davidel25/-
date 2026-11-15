"""Verify the ADOPTED configuration end to end (old vs new, full filters, SPY+QQQ synthetic, $20k risk-cap check).
    python experiments/verify_adopted.py verify.json   # ~75 s on 4 cores"""
import json, sys, time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.compare import variant_stats
from src.backtest.data import from_frame
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.synthetic import SymbolProfile, generate_universe
from src.risk.manager import RiskLimits
from src.strategies.credit_spreads import ScannerConfig

OLD = ScannerConfig.disciplined(min_short_delta=0.10, max_short_delta=0.25, min_credit=0.60, min_roc=0.20)
NEW = ScannerConfig.disciplined()
BIG = 500_000.0

def bc(scanner, target, equity=BIG, risk_pct=0.02):
    return BacktestConfig(scanner=scanner, limits=RiskLimits.disciplined(profit_target_pct=target, max_risk_per_trade_pct=risk_pct),
                          equity=equity, max_new_per_day=2)

VARIANTS = {
    "OLD: 0.10-0.25, floors .60/20%, 50% target": bc(OLD, 0.50),
    "NEW: 0.12-0.15, floors .45/12%, 65% target": bc(NEW, 0.65),
    "NEW delta+floors, 50% target": bc(NEW, 0.50),
    "OLD delta+floors, 65% target": bc(OLD, 0.65),
    "NEW delta band, OLD floors (.60/20%), 65%": bc(replace(NEW, min_credit=0.60, min_roc=0.20), 0.65),
    "NEW @ $20k, 2% risk cap": bc(NEW, 0.65, 20_000.0, 0.02),
    "NEW @ $20k, 2.5% risk cap": bc(NEW, 0.65, 20_000.0, 0.025),
    "OLD @ $20k, 2% risk cap": bc(OLD, 0.50, 20_000.0, 0.02),
}

def job(args):
    vrp, seed, days = args
    profs = {"SPY": SymbolProfile(real_vol=0.16, vrp=vrp, spot0=500, half_spread_pct=0.004),
             "QQQ": SymbolProfile(real_vol=0.22, vrp=vrp, spot0=430, half_spread_pct=0.005)}
    df, _ = generate_universe(profs, days, seed)
    store = from_frame(df)
    out = []
    for n, c in VARIANTS.items():
        r = run_backtest(store, c)
        out.append({"vrp": vrp, "seed": seed, "variant": n, "skipped": dict(r.skipped), **variant_stats(n, r).flat()})
    return out

if __name__ == "__main__":
    t0 = time.time(); rows = []
    jobs = [(v, s, 900) for v in (0.10, 0.20) for s in range(1, 9)]
    with Pool(4) as p:
        for part in p.imap_unordered(job, jobs):
            rows += part; print("done", len(rows), round(time.time() - t0), "s", flush=True)
    json.dump(rows, open(sys.argv[1], "w")); print("DONE")
