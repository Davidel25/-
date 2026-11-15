"""Strategy-adjustment experiments on SYNTHETIC chains (assumption-driven; see REPORT.md for caveats).

    python experiments/strategy_adjustments.py full 900 8 results.json       # delta + exit grids and universe (~8 min, 4 cores)
    python experiments/strategy_adjustments.py universe 900 8 universe.json  # asset-class experiment only
    python experiments/strategy_adjustments.py friction 900 8 friction.json  # commission / bid-ask decomposition
    python experiments/strategy_adjustments.py check 900 8 check.json        # frictionless stock sanity + mid fills
    python experiments/analyze.py results.json                               # paired tables

Run with real option history by building a ChainStore with src.backtest.data.load_chains and reusing ``run_set``."""
import json
import sys
import time
from dataclasses import replace
from datetime import date
from multiprocessing import Pool

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.compare import variant_stats
from src.backtest.data import from_frame
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.synthetic import SymbolProfile, generate_universe
from src.risk.manager import RiskLimits
from src.strategies.credit_spreads import ScannerConfig

EQUITY = 200_000.0
FAR = [date(2100, 1, 1)]


def scanner(lo, hi, credit=0.20, roc=0.05):
    return ScannerConfig(min_dte=30, max_dte=50, min_short_delta=lo, max_short_delta=hi, widths=(1.0, 2.0, 5.0),
                         min_credit=credit, min_roc=roc, pricing="natural", max_bid_ask_pct=0.10, preferred_dte=(35, 45))


def cfg(lo, hi, target=0.50, credit=0.20, roc=0.05, earnings=None):
    return BacktestConfig(scanner=scanner(lo, hi, credit, roc), limits=RiskLimits(profit_target_pct=target),
                          equity=EQUITY, max_new_per_day=2, earnings=earnings)


def run_set(df, variants):
    store = from_frame(df)
    return {n: variant_stats(n, run_backtest(store, c)).flat() for n, c in variants.items()}


def task_delta_exit(args):
    vrp, seed, days = args
    df, _ = generate_universe({"SPY": SymbolProfile(real_vol=0.16, vrp=vrp, half_spread_pct=0.004)}, days, seed)
    v = {"d20-25|t50 (current)": cfg(.20, .25), "d15-20|t50": cfg(.15, .20), "d12-15|t50": cfg(.12, .15),
         "d20-25|t30": cfg(.20, .25, .30), "d12-15|t30": cfg(.12, .15, .30),
         "d20-25|t40": cfg(.20, .25, .40), "d20-25|t65": cfg(.20, .25, .65), "d20-25|t80": cfg(.20, .25, .80),
         "d12-15|t65": cfg(.12, .15, .65),
         "d20-25|t50|disciplined floors": cfg(.20, .25, .50, 0.60, 0.20),
         "d12-15|t50|disciplined floors": cfg(.12, .15, .50, 0.60, 0.20)}
    return [{"exp": "delta_exit", "vrp": vrp, "seed": seed, "variant": n, **m} for n, m in run_set(df, v).items()]


def stock(vrp=0.10, half=0.015, priced=True):
    return SymbolProfile(real_vol=0.30, vrp=vrp, spot0=150, half_spread_pct=half, min_half_spread=0.02,
                         jump_per_year=4, jump_sd=0.04, earnings_every_days=91, earnings_jump_sd=0.06,
                         price_jumps_in_iv=priced)


UNIVERSE = {   # name -> (symbol, profile, earnings gate on?)
    "SPY-like ETF (vol 16%, VRP +10%)": ("SPY", SymbolProfile(real_vol=0.16, vrp=0.10, spot0=500, half_spread_pct=0.004), None),
    "QQQ-like ETF (vol 22%, VRP +10%)": ("QQQ", SymbolProfile(real_vol=0.22, vrp=0.10, spot0=430, half_spread_pct=0.005), None),
    "High-vol index (vol 24%, SAME VRP +10%)": ("IWM", SymbolProfile(real_vol=0.24, vrp=0.10, spot0=210, half_spread_pct=0.007,
                                                                      jump_per_year=2, jump_sd=0.03, price_jumps_in_iv=True), None),
    "High-vol index (vol 24%, HIGHER VRP +14%)": ("IWM", SymbolProfile(real_vol=0.24, vrp=0.14, spot0=210, half_spread_pct=0.007,
                                                                         jump_per_year=2, jump_sd=0.03, price_jumps_in_iv=True), None),
    "Stock: 0.5% half-spread, VRP +10% (gate ON)": ("AAA", stock(0.10, 0.005), True),
    "Stock: 1.5% half-spread, VRP +10% (gate ON)": ("AAA", stock(0.10, 0.015), True),
    "Stock: 3.0% half-spread, VRP +10% (gate ON)": ("AAA", stock(0.10, 0.030), True),
    "Stock: 1.5% half-spread, VRP +10%, news gaps UNPRICED (gate ON)": ("AAA", stock(0.10, 0.015, False), True),
    "Stock: 1.5% half-spread, VRP +10% (NO earnings gate)": ("AAA", stock(0.10, 0.015), False),
    "Stock: 1.5% half-spread, VRP 0% (gate ON)": ("AAA", stock(0.0, 0.015), True),
    "Stock: 0.5% half-spread, VRP +20% (gate ON)": ("AAA", stock(0.20, 0.005), True),
}


def task_universe(args):
    seed, days = args
    out = []
    for name, (sym, prof, gate) in UNIVERSE.items():
        df, sched = generate_universe({sym: prof}, days, seed)
        earnings = None if gate is None else ({sym: sched[sym]} if gate else {sym: FAR})
        c = cfg(.15, .25, 0.50, credit=0.15, roc=0.08, earnings=earnings)
        out.append({"exp": "universe", "seed": seed, "variant": name,
                    **variant_stats(name, run_backtest(from_frame(df), c)).flat()})
    return out



def task_friction(args):
    seed, days = args
    spy = lambda vrp, half, mh: SymbolProfile(real_vol=0.16, vrp=vrp, spot0=500, half_spread_pct=half, min_half_spread=mh)  # noqa: E731
    out = []
    def go(name, sym, prof, c):
        df, sched = generate_universe({sym: prof}, days, seed)
        if c.earnings == "GATE":
            c = replace(c, earnings={sym: sched[sym]})
        out.append({"exp": "friction", "seed": seed, "variant": name, **variant_stats(name, run_backtest(from_frame(df), c)).flat()})
    base = cfg(.15, .25, 0.50, credit=0.15, roc=0.08)
    free = lambda c: replace(c, commission_per_contract=0.0)  # noqa: E731
    go("SPY-like FRICTIONLESS, VRP 0%", "SPY", spy(0.0, 0.0, 0.0), free(base))
    go("SPY-like FRICTIONLESS, VRP +10%", "SPY", spy(0.10, 0.0, 0.0), free(base))
    go("SPY-like FRICTIONLESS, VRP +20%", "SPY", spy(0.20, 0.0, 0.0), free(base))
    go("SPY-like, VRP +10%: commission only", "SPY", spy(0.10, 0.0, 0.0), base)
    go("SPY-like, VRP +10%: bid/ask only", "SPY", spy(0.10, 0.004, 0.01), free(base))
    go("SPY-like, VRP +10%: both (as before)", "SPY", spy(0.10, 0.004, 0.01), base)
    stk = stock(0.10, 0.015)
    gate = replace(cfg(.15, .25, 0.50, credit=0.15, roc=0.08), earnings="GATE")
    go("Stock 1.5%: as before (gate ON)", "AAA", stk, gate)
    go("Stock 1.5%: commission $0", "AAA", stk, free(gate))
    go("Stock 1.5%: 5-wide spreads only", "AAA", stk, replace(gate, scanner=replace(gate.scanner, widths=(5.0,))))
    go("Stock 0%: no commission, no bid/ask, VRP 0% (sanity)", "AAA", stock(0.0, 0.0), free(gate))
    return out


def task_check(args):
    seed, days = args
    out = []
    def go(name, sym, prof, c):
        df, sched = generate_universe({sym: prof}, days, seed)
        if c.earnings == "GATE":
            c = replace(c, earnings={sym: sched[sym]})
        out.append({"exp": "check", "seed": seed, "variant": name, **variant_stats(name, run_backtest(from_frame(df), c)).flat()})
    base = cfg(.15, .25, 0.50, credit=0.15, roc=0.08)
    free = lambda c: replace(c, commission_per_contract=0.0)  # noqa: E731
    gate = replace(base, earnings="GATE")
    nofric = lambda vrp: replace(stock(vrp, 0.0), min_half_spread=0.0)  # noqa: E731
    go("Stock TRULY frictionless, VRP 0%", "AAA", nofric(0.0), free(gate))
    go("Stock TRULY frictionless, VRP +10%", "AAA", nofric(0.10), free(gate))
    spy = SymbolProfile(real_vol=0.16, vrp=0.10, spot0=500, half_spread_pct=0.004)
    mid = replace(base, scanner=replace(base.scanner, pricing="mid"), exit_pricing="mid")
    go("SPY-like VRP +10%: natural fills (as before)", "SPY", spy, base)
    go("SPY-like VRP +10%: fills at MID, commission kept", "SPY", spy, mid)
    return out


def main():
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "smoke":
        rows = task_delta_exit((0.10, 1, 260)) + task_universe((1, 260))
        for r in rows[:4] + rows[-3:]:
            print(r["variant"], r["trades"], r["win_rate"], r["sharpe"])
        print("ok", round(time.time() - t0), "s")
        return
    days = int(sys.argv[2])
    seeds = range(1, int(sys.argv[3]) + 1)
    jobs1 = [(vrp, s, days) for vrp in (0.0, 0.10, 0.20) for s in seeds] if mode == "full" else []
    rows = []
    with Pool(4) as pool:
        for part in pool.imap_unordered(task_delta_exit, jobs1):
            rows += part
            print("delta_exit done", len(rows), round(time.time() - t0), "s", flush=True)
        task = {"friction": task_friction, "check": task_check}.get(mode, task_universe)
        for part in pool.imap_unordered(task, [(s, days) for s in seeds]):
            rows += part
            print("universe done", len(rows), round(time.time() - t0), "s", flush=True)
    json.dump(rows, open(sys.argv[4], "w"))
    print("DONE", len(rows), "rows", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
