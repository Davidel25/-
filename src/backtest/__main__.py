"""CLI:  python -m src.backtest --data chains.csv [--earnings earnings.csv] [--oos-from 2025-01-01] [--out report.md]
        python -m src.backtest --demo        (synthetic data: exercises the pipeline, says nothing about real edge)"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date

from src.backtest.data import from_frame, load_chains, load_earnings
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import summarize
from src.backtest.report import render_report
from src.backtest.synthetic import generate_frame
from src.risk.manager import RiskLimits
from src.strategies.credit_spreads import ScannerConfig

DEMO_BANNER = ("SYNTHETIC DATA: option prices were generated from Black-Scholes with an assumed implied-vol premium. "
               "This run only proves the pipeline works; it says NOTHING about whether the strategy has an edge.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay historical option chains through the scanner and risk rules.")
    ap.add_argument("--data", help="CSV/Parquet of daily option-chain snapshots (see src/backtest/data.py)")
    ap.add_argument("--earnings", help="CSV with columns symbol,date of historical earnings dates")
    ap.add_argument("--demo", action="store_true", help="use generated synthetic chains instead of --data")
    ap.add_argument("--demo-days", type=int, default=260)
    ap.add_argument("--demo-vrp", type=float, default=0.10, help="synthetic IV premium over realized vol")
    ap.add_argument("--demo-seed", type=int, default=1)
    ap.add_argument("--preset", choices=["disciplined", "base"], default="disciplined")
    ap.add_argument("--profit-target", type=float, default=None,
                    help="override the profit target as a fraction of the credit (default: the configured 0.65)")
    ap.add_argument("--equity", type=float, default=20_000.0)
    ap.add_argument("--commission", type=float, default=0.65, help="$ per contract per leg fill")
    ap.add_argument("--slippage", type=float, default=0.0, help="extra $ per share on each leg fill")
    ap.add_argument("--exit-pricing", choices=["natural", "mid"], default="natural")
    ap.add_argument("--max-new-per-day", type=int, default=1)
    ap.add_argument("--oos-from", help="YYYY-MM-DD: report in-sample vs out-of-sample from this entry date")
    ap.add_argument("--out", help="write the Markdown report to this file as well")
    ap.add_argument("--json", help="write summary + trades as JSON to this file")
    args = ap.parse_args(argv)
    if not args.demo and not args.data:
        print("ERROR: give --data PATH (or --demo)", file=sys.stderr)
        return 2
    try:
        oos = date.fromisoformat(args.oos_from) if args.oos_from else None
        store = from_frame(generate_frame(days=args.demo_days, seed=args.demo_seed, vrp=args.demo_vrp)) \
            if args.demo else load_chains(args.data)
        earnings = load_earnings(args.earnings) if args.earnings else None
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    scanner = ScannerConfig.disciplined() if args.preset == "disciplined" else ScannerConfig(pricing="natural")
    limits = RiskLimits.disciplined(**({"profit_target_pct": args.profit_target} if args.profit_target else {}))
    cfg = BacktestConfig(scanner=scanner, limits=limits, equity=args.equity,
                         commission_per_contract=args.commission, slippage_per_leg=args.slippage,
                         exit_pricing=args.exit_pricing, max_new_per_day=args.max_new_per_day, earnings=earnings)
    result = run_backtest(store, cfg)
    text = render_report(result, oos, DEMO_BANNER if args.demo else None)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    if args.json:
        s = summarize(result.trades, result.equity_curve, cfg.equity, result.first_date, result.last_date)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": {**asdict(s), "verdict": s.verdict}, "trades": [asdict(t) for t in result.trades]},
                      fh, default=str, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
