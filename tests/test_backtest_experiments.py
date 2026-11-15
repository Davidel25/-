"""Sharpe ratio, per-asset synthetic profiles and variant comparison."""
import math
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from src.backtest.compare import run_variants, variant_stats
from src.backtest.data import from_frame
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import daily_sharpe, summarize
from src.backtest.report import render_report
from src.backtest.synthetic import (
    SymbolProfile,
    earnings_schedule,
    generate_frame,
    generate_universe,
    trading_days,
)
from src.risk.manager import RiskLimits
from src.strategies.credit_spreads import ScannerConfig

D0 = date(2024, 1, 2)


def curve(values):
    return [(D0 + timedelta(i), v) for i, v in enumerate(values)]


# --- Sharpe ---
def test_sharpe_is_the_annualised_ratio_of_mean_to_std_of_daily_changes():
    pnl = [0.0]
    for i in range(60):
        pnl.append(pnl[-1] + (3.0 if i % 2 else 1.0))              # daily changes alternate 1 and 3
    diffs = [b - a for a, b in zip(pnl, pnl[1:], strict=False)]
    mean = sum(diffs) / len(diffs)
    std = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1))
    assert daily_sharpe(curve(pnl)) == pytest.approx(mean / std * math.sqrt(252))


def test_sharpe_is_scale_free_signed_and_guarded():
    base = [0, 1, 3, 2, 5, 4, 6, 9, 8, 10, 12, 11, 13, 15, 14, 16, 19, 18, 20, 22, 21, 24]
    assert daily_sharpe(curve(base)) == pytest.approx(daily_sharpe(curve([100 * x for x in base])))
    assert daily_sharpe(curve([-x for x in base])) == pytest.approx(-daily_sharpe(curve(base)))
    assert daily_sharpe(curve([0.0] * 40)) is None                          # no variance
    assert daily_sharpe(curve([0, 1, 2])) is None                           # too short
    assert daily_sharpe([]) is None


def test_summary_and_report_carry_the_sharpe():
    s = summarize([], curve([float(i % 5) for i in range(50)]))
    assert s.sharpe is not None
    assert summarize([]).sharpe is None


# --- synthetic profiles ---
def test_default_generation_is_unchanged_in_shape_and_reproducible():
    a, b = generate_frame(days=30, seed=9), generate_frame(days=30, seed=9)
    pd.testing.assert_frame_equal(a, b)
    wanted = {"date", "symbol", "expiry", "strike", "right", "bid", "ask", "underlying_price", "delta", "iv"}
    assert wanted <= set(a.columns)


def test_universe_is_deterministic_and_returns_earnings_dates():
    prof = {"X": SymbolProfile(spot0=150, earnings_every_days=91, earnings_jump_sd=0.06)}
    (a, ea), (b, eb) = generate_universe(prof, 40, 5), generate_universe(prof, 40, 5)
    pd.testing.assert_frame_equal(a, b)
    assert ea == eb and ea["X"] and all(d.weekday() < 5 for d in ea["X"])
    assert ea["X"][-1] >= trading_days(D0, 40)[-1]                               # runs past the end of the data
    assert generate_universe({"SPY": SymbolProfile()}, 5, 1)[1] == {"SPY": []}   # no earnings configured


def test_earnings_schedule_cadence_and_weekdays():
    p = SymbolProfile(earnings_every_days=91, earnings_jump_sd=0.05)
    dates = earnings_schedule(p, D0, 300)
    assert len(dates) >= 4 and all(d.weekday() < 5 for d in dates)
    assert all(60 < (b - a).days < 125 for a, b in zip(dates, dates[1:], strict=False))
    assert earnings_schedule(SymbolProfile(), D0, 300) == []
    assert earnings_schedule(p, D0, 300, offset=14)[0] > dates[0]


def returns(df, symbol="X"):
    px = df[df["symbol"] == symbol].drop_duplicates("date").set_index("date")["underlying_price"]
    return [math.log(b / a) for a, b in zip(px, px[1:], strict=False)]


def test_news_and_earnings_jumps_fatten_the_return_tails():
    calm = generate_universe({"X": SymbolProfile(real_vol=0.2, spot0=150)}, 500, 3)[0]
    wild = generate_universe({"X": SymbolProfile(real_vol=0.2, spot0=150, jump_per_year=12, jump_sd=0.05,
                                                  earnings_every_days=91, earnings_jump_sd=0.08)}, 500, 3)[0]
    assert max(abs(r) for r in returns(wild)) > 1.5 * max(abs(r) for r in returns(calm))


def test_wider_quotes_and_the_premium_show_up_in_the_chains():
    tight = generate_universe({"X": SymbolProfile(half_spread_pct=0.002)}, 5, 1)[0]
    wide = generate_universe({"X": SymbolProfile(half_spread_pct=0.03)}, 5, 1)[0]
    rel = lambda d: ((d["ask"] - d["bid"]) / ((d["ask"] + d["bid"]) / 2))[d["bid"] > 1].mean()    # noqa: E731
    assert rel(wide) > 5 * rel(tight)
    lo = generate_universe({"X": SymbolProfile(vrp=0.0)}, 5, 1)[0]
    hi = generate_universe({"X": SymbolProfile(vrp=0.3)}, 5, 1)[0]
    assert hi["iv"].mean() > 1.15 * lo["iv"].mean()


def test_pricing_news_gaps_into_implied_vol_raises_iv_only_when_asked():
    base = SymbolProfile(real_vol=0.2, vrp=0.1, jump_per_year=6, jump_sd=0.05)
    unpriced = generate_universe({"X": base}, 60, 4)[0]
    priced = generate_universe({"X": replace(base, price_jumps_in_iv=True)}, 60, 4)[0]
    assert priced["iv"].mean() > 1.03 * unpriced["iv"].mean()
    quiet = SymbolProfile(real_vol=0.2, vrp=0.1)                              # no jumps: the flag changes nothing
    pd.testing.assert_frame_equal(generate_universe({"X": quiet}, 20, 2)[0],
                                  generate_universe({"X": replace(quiet, price_jumps_in_iv=True)}, 20, 2)[0])


def test_options_spanning_earnings_are_priced_for_the_report():
    prof = {"X": SymbolProfile(spot0=150, earnings_every_days=91, earnings_jump_sd=0.07)}
    df, sched = generate_universe(prof, 30, 2)
    d0 = df["date"].iloc[0]
    day = df[(df["date"] == d0) & (df["right"] == "C") & ((df["strike"] - df["underlying_price"]).abs() < 3)]
    nxt = min(e for e in sched["X"] if e.isoformat() >= d0)
    after = day[pd.to_datetime(day["expiry"]).dt.date >= nxt]["iv"].mean()
    before = day[pd.to_datetime(day["expiry"]).dt.date < nxt]["iv"].mean()
    assert after > before


# --- variants ---
def cfgs():
    base = ScannerConfig(min_dte=30, max_dte=50, widths=(5.0,), min_credit=0.2, min_roc=0.05, pricing="natural")
    return {"t50": BacktestConfig(scanner=base, equity=200_000.0, max_new_per_day=2),
            "t30": BacktestConfig(scanner=base, limits=RiskLimits(profit_target_pct=0.30), equity=200_000.0,
                                  max_new_per_day=2)}


def test_variants_share_the_same_data_and_report_the_comparison_metrics():
    store = from_frame(generate_frame(days=140, seed=4))
    out = run_variants(store, cfgs())
    assert set(out) == {"t50", "t30"}
    for v in out.values():
        flat = v.flat()
        assert v.summary.trades > 0 and 0 <= v.stop_rate <= 1 and 0 <= v.target_rate <= 1
        assert 0 < v.avg_credit_over_width < 1 and 0.05 < v.avg_short_delta < 0.5
        assert v.p05_trade <= v.summary.best
        assert flat["trades"] == v.summary.trades and "sharpe" in flat and flat["trades_per_year"] > 0
    assert out["t30"].summary.avg_hold_days <= out["t50"].summary.avg_hold_days     # a lower target closes sooner
    assert out["t30"].target_rate >= out["t50"].target_rate


def test_variant_stats_handles_a_run_with_no_trades():
    store = from_frame(generate_frame(days=40, seed=1))
    c = replace(cfgs()["t50"], scanner=ScannerConfig(min_credit=99.0))
    v = variant_stats("none", run_backtest(store, c))
    assert v.summary.trades == 0 and v.stop_rate is None and v.avg_credit_over_width is None and v.p05_trade is None


def test_report_includes_the_sharpe_row():
    store = from_frame(generate_frame(days=60, seed=2))
    assert "Sharpe (daily P&L" in render_report(run_backtest(store, cfgs()["t50"]))
