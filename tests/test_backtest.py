"""Backtest harness: data loading, replay engine (exits, costs, gates, no look-ahead), statistics, report, CLI."""
import json
import math
import random
from datetime import date, timedelta

import pandas as pd
import pytest

from src.backtest.__main__ import main
from src.backtest.data import NO_OI, ChainStore, from_frame, load_chains, load_earnings
from src.backtest.engine import BacktestConfig, Trade, run_backtest
from src.backtest.metrics import (
    MIN_MONTHS,
    MIN_TRADES,
    breakdown,
    cluster_bootstrap_ci,
    iv_hv_bucket,
    max_drawdown,
    split_by_date,
    summarize,
)
from src.backtest.report import render_report
from src.backtest.synthetic import generate_frame, trading_days
from src.risk.manager import RiskLimits
from src.strategies.credit_spreads import ScannerConfig

D0 = date(2026, 1, 5)
EXP = D0 + timedelta(days=35)


# --- tiny hand-built datasets ---
def row(d, symbol, strike, right, bid, ask, delta, und=500.0, expiry=EXP, oi=100, iv=0.2):
    return {"date": d.isoformat(), "symbol": symbol, "expiry": expiry.isoformat(), "strike": strike, "right": right,
            "bid": bid, "ask": ask, "underlying_price": und, "delta": delta, "iv": iv, "open_interest": oi}


def day(d, short_q, long_q, symbol="SPY", und=500.0, expiry=EXP):
    """One day's two put legs: short 480 and long 475. ``*_q`` = (bid, ask), or None to omit the leg."""
    rows = []
    if short_q:
        rows.append(row(d, symbol, 480, "P", *short_q, -0.15, und, expiry))
    if long_q:
        rows.append(row(d, symbol, 475, "P", *long_q, -0.08, und, expiry))
    if not rows:   # keep the date (and its underlying price) in the dataset via a still-live far-OTM call
        rows.append(row(d, symbol, 600, "C", 0.05, 0.07, 0.01, und, expiry + timedelta(14)))
    return rows


ENTRY = day(D0, (2.00, 2.04), (1.00, 1.04))                # natural credit 2.00 - 1.04 = 0.96, 5-wide
CALM = ((0.85, 0.90), (0.50, 0.60))                          # next day: credit 0.25 -> no re-entry


def cfg(**kw):
    scanner = ScannerConfig(min_dte=30, max_dte=50, widths=(5.0,), min_credit=0.5, min_roc=0.1, min_open_interest=1,
                            max_bid_ask_pct=0.2, pricing="natural", top_n_per_symbol=5)
    kw.setdefault("equity", 50_000.0)                        # 2% = $1,000 per trade: a $404-risk spread passes
    kw.setdefault("limits", RiskLimits())                    # 50% target: the hand-built P&L below assumes it
    return BacktestConfig(scanner=scanner, **kw)


def run(rows, **kw):
    return run_backtest(from_frame(pd.DataFrame(rows)), cfg(**kw))


def trade_after(*following_days, **kw):
    r = run(ENTRY + [x for d in following_days for x in d], **kw)
    return r, (r.trades[0] if r.trades else None)


def test_take_profit_exit_and_cost_math():
    r, t = trade_after(day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55)))        # close debit 0.95 - 0.55 = 0.40
    assert (t.exit_reason, t.open_date, t.close_date) == ("TAKE_PROFIT", D0, D0 + timedelta(1))
    assert t.credit == pytest.approx(0.96) and t.exit_debit == pytest.approx(0.40)
    assert t.gross_pnl == pytest.approx(56.0) and t.costs == pytest.approx(2.6) and t.net_pnl == pytest.approx(53.4)
    assert t.dte_at_entry == 35 and t.dte_bucket == "35-45" and t.weekday == 0 and t.max_loss == pytest.approx(404.0)
    assert r.still_open == 0 and r.first_date == D0 and r.days == 2


def test_stop_loss_exit():
    _, t = trade_after(day(D0 + timedelta(1), (3.9, 3.9), (0.9, 0.9)))             # debit 3.00 >= 3 x credit
    assert t.exit_reason == "STOP_LOSS" and t.gross_pnl == pytest.approx((0.96 - 3.0) * 100)


def test_dte_exit_and_a_position_is_never_closed_on_its_entry_day():
    r, t = trade_after(day(D0 + timedelta(22), (0.9, 0.9), (0.2, 0.2)))           # DTE 13, debit 0.70 (27% profit)
    assert t.exit_reason == "CLOSE_DTE" and t.close_date > t.open_date
    assert all(x.close_date > x.open_date for x in r.trades)


def test_expiry_settles_at_intrinsic_value():
    hold = day(D0 + timedelta(34), (0.9, 0.9), (0.2, 0.2))                         # debit 0.70, DTE 1: nothing triggers
    # quotes on their own expiry day are dropped by the loader; only the underlying price remains
    otm = day(EXP, None, None, und=500.0)
    _, win = trade_after(hold, otm, limits=RiskLimits(close_at_dte=0))
    assert win.exit_reason == "EXPIRY" and win.exit_debit == 0 and win.net_pnl == pytest.approx(96.0 - 2.6)
    itm = day(EXP, None, None, und=470.0)
    _, loss = trade_after(hold, itm, limits=RiskLimits(close_at_dte=0))
    assert loss.exit_reason == "EXPIRY" and loss.exit_debit == 5.0       # (480-470) - (475-470) = 5 = the width
    assert loss.gross_pnl == pytest.approx((0.96 - 5.0) * 100)


def test_mid_exit_pricing_differs_from_natural():
    nxt = day(D0 + timedelta(1), (0.45, 0.65), (0.20, 0.30))        # natural 0.65-0.20 = 0.45; mid 0.55-0.25 = 0.30
    _, nat = trade_after(nxt)
    _, mid = trade_after(nxt, exit_pricing="mid")
    assert nat.exit_debit == pytest.approx(0.45) and mid.exit_debit == pytest.approx(0.30)


def test_slippage_is_charged_on_all_four_fills():
    _, t = trade_after(day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55)), slippage_per_leg=0.01, contracts=1)
    assert t.costs == pytest.approx(2.6 + 0.01 * 100 * 4)


def test_multiple_contracts_scale_pnl_and_costs():
    _, t = trade_after(day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55)), contracts=3, equity=500_000)
    assert t.gross_pnl == pytest.approx(168.0) and t.costs == pytest.approx(7.8) and t.max_loss == pytest.approx(1212.0)


def test_per_trade_risk_cap_blocks_entries_exactly_like_the_live_rules():
    r = run(ENTRY + day(D0 + timedelta(1), *CALM), equity=20_000)                  # 2% of $20k = $400 < $404 risk
    assert r.trades == [] and r.skipped["risk_limits"] == 1


def test_position_quote_gap_keeps_the_position_open_and_is_counted():
    gap = day(D0 + timedelta(1), None, None)
    r, t = trade_after(gap, day(D0 + timedelta(2), (0.95, 0.95), (0.55, 0.55)))
    assert r.skipped["position_quote_gap"] == 1 and t.close_date == D0 + timedelta(2)


def test_positions_open_at_the_end_are_reported_not_counted():
    r = run(ENTRY + day(D0 + timedelta(1), (1.5, 1.6), (0.7, 0.8)))                # debit 0.90: nothing triggers
    assert r.trades == [] and r.still_open == 1


def test_one_entry_per_symbol_and_direction_while_a_position_is_open():
    same = (2.00, 2.04), (1.00, 1.04)
    r = run(ENTRY + day(D0 + timedelta(1), *same) + day(D0 + timedelta(2), *same))
    assert r.still_open == 1 and r.skipped["already_holding"] >= 1


# --- earnings gate in the replay ---
def stock_rows():
    return day(D0, (2.00, 2.04), (1.00, 1.04), symbol="AAPL") + day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55),
                                                                       symbol="AAPL")


def test_stocks_are_not_traded_without_an_earnings_history():
    r = run(stock_rows())
    assert r.trades == [] and r.skipped["earnings"] >= 1 and any("NOT traded" in n for n in r.notes)


def test_earnings_inside_the_trade_window_block_and_after_expiry_allow():
    assert run(stock_rows(), earnings={"AAPL": [D0 + timedelta(10)]}).trades == []
    assert run(stock_rows(), earnings={"AAPL": [EXP]}).trades == []                # expiry day counts as inside
    ok = run(stock_rows(), earnings={"AAPL": [EXP + timedelta(5)]})
    assert len(ok.trades) == 1 and ok.trades[0].symbol == "AAPL"


def test_history_that_ends_before_the_trade_means_unknown_and_blocked():
    assert run(stock_rows(), earnings={"AAPL": [D0 - timedelta(30)]}).trades == []
    assert run(stock_rows(), earnings={"MSFT": [EXP + timedelta(5)]}).trades == []          # no data for this symbol


def test_etfs_need_no_earnings_data():
    assert len(run(ENTRY + day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55))).trades) == 1


# --- no look-ahead ---
def test_stats_use_only_data_up_to_the_signal_day():
    rows = [row(D0 + timedelta(i), "SPY", 480, "P", 1, 1.1, -0.15, und=100 + i) for i in range(10)]
    store = from_frame(pd.DataFrame(rows))
    assert store.closes("SPY", D0 + timedelta(3)) == [100, 101, 102, 103]
    assert store.closes("SPY", D0 - timedelta(1)) == [] and len(store.closes("SPY", D0 + timedelta(99))) == 10
    assert store.closes("NOPE", D0) == []


def test_trend_filter_sees_a_flat_history_not_future_prices():
    """With 60 flat days then a spike, a Bull Put needs price > SMA50 on the signal day itself, using past data only."""
    rows = []
    for i in range(60):
        d = D0 + timedelta(i)
        rows += [row(d, "SPY", 480, "P", 2.0, 2.04, -0.15, und=500.0, expiry=d + timedelta(35)),
                 row(d, "SPY", 475, "P", 1.0, 1.04, -0.08, und=500.0, expiry=d + timedelta(35))]
    scanner = ScannerConfig(min_dte=30, max_dte=50, widths=(5.0,), min_credit=0.5, min_roc=0.1, min_open_interest=1,
                            max_bid_ask_pct=0.2, pricing="natural", require_trend_alignment=True)
    flat = run_backtest(from_frame(pd.DataFrame(rows)), BacktestConfig(scanner=scanner, equity=50_000))
    assert flat.trades == [] and flat.still_open == 0           # price == SMA50 exactly: neither direction qualifies


# --- data loading ---
def frame(**over):
    base = {"date": [D0.isoformat()] * 2, "symbol": ["SPY"] * 2, "expiry": [EXP.isoformat()] * 2,
            "strike": [480, 475],
            "right": ["p", "P"], "bid": [2.0, 1.0], "ask": [2.04, 1.04], "underlying_price": [500, 500],
            "delta": [-0.15, -0.08], "iv": [0.2, 0.2], "open_interest": [100, 100]}
    base.update(over)
    return pd.DataFrame(base)


def test_loader_validates_input():
    with pytest.raises(ValueError, match="missing required column"):
        from_frame(frame().drop(columns=["bid"]))
    with pytest.raises(ValueError, match="empty"):
        from_frame(frame().iloc[0:0])
    with pytest.raises(ValueError, match="P or C"):
        from_frame(frame(right=["X", "P"]))
    with pytest.raises(ValueError, match="iv and/or delta"):
        from_frame(frame().drop(columns=["iv", "delta"]))
    with pytest.raises(ValueError, match="cannot parse"):
        from_frame(frame(date=["nope", "nope"]))
    with pytest.raises(ValueError, match="no usable"):
        from_frame(frame(bid=[3.0, 3.0]))                          # every quote crossed


def test_loader_drops_bad_rows_fills_delta_and_notes_missing_open_interest():
    f = frame(bid=[2.0, 5.0], ask=[2.04, 1.0])                      # second row crossed
    s = from_frame(f)
    assert s.dropped == 1 and any("dropped" in n for n in s.notes) and len(s.chain(D0, "SPY")) == 1
    no_delta = from_frame(frame().drop(columns=["delta"]))          # delta computed from iv by Black-Scholes
    assert all(-1 < q.delta < 0 for q in no_delta.chain(D0, "SPY"))
    no_oi = from_frame(frame().drop(columns=["open_interest"]))
    assert no_oi.chain(D0, "SPY")[0].open_interest == NO_OI and any("NOT applied" in n for n in no_oi.notes)
    assert from_frame(frame()).quote(D0, "SPY", EXP, "P", 480).bid == 2.0
    assert isinstance(from_frame(frame()), ChainStore) and from_frame(frame()).symbols == ["SPY"]


def test_loader_reads_csv_and_earnings_files(tmp_path):
    p = tmp_path / "c.csv"
    frame().to_csv(p, index=False)
    assert load_chains(p).dates == [D0]
    with pytest.raises(ValueError, match="cannot read"):
        load_chains(tmp_path / "missing.csv")
    e = tmp_path / "e.csv"
    e.write_text("symbol,date\naapl,2026-02-01\nAAPL,2026-05-01\nAAPL,2026-02-01\nmsft,2026-03-01\n")
    assert load_earnings(e) == {"AAPL": [date(2026, 2, 1), date(2026, 5, 1)], "MSFT": [date(2026, 3, 1)]}
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,when\nx,y\n")
    with pytest.raises(ValueError, match="cannot read earnings"):
        load_earnings(bad)


# --- statistics ---
def tr(net, open_date=D0, reason="TAKE_PROFIT", symbol="SPY", st="BULL_PUT", dte=35, iv_hv=1.2, held=10,
       max_loss=400.0):
    return Trade(symbol, st, 480, 475, open_date, open_date + timedelta(held), EXP, dte, 1.0, 0.5, reason, 1,
                 net + 2.6, 2.6, net, max_loss, 0.15, iv_hv)


def test_summary_numbers():
    trades = [tr(50), tr(50), tr(50), tr(-200)]
    s = summarize(trades, None, 20_000, D0, D0 + timedelta(365))
    assert (s.trades, s.wins, s.win_rate, s.net_pnl) == (4, 3, 0.75, -50.0)
    assert s.avg_win == 50 and s.avg_loss == -200 and s.profit_factor == pytest.approx(0.75)
    assert s.breakeven_win_rate == pytest.approx(0.8) and s.mean_net == -12.5
    assert s.mean_pct_of_risk == pytest.approx(-50 / 1600) and s.worst == -200 and s.best == 50
    assert s.annualized_return == pytest.approx(-50 / 20_000 / (365 / 365.25)) and s.avg_hold_days == 10
    assert s.costs == pytest.approx(10.4) and s.months == 1


def test_summary_edge_cases():
    empty = summarize([])
    assert empty.trades == 0 and empty.win_rate is None and empty.profit_factor is None and "NO TRADES" in empty.verdict
    assert summarize([tr(10), tr(20)]).profit_factor == math.inf and summarize([tr(-5)]).profit_factor == 0


def test_cluster_bootstrap_is_deterministic_and_needs_two_months():
    one_month = [tr(10, D0), tr(-5, D0 + timedelta(3))]
    assert cluster_bootstrap_ci(one_month) is None
    rng = random.Random(5)
    trades = [tr(rng.gauss(10, 50), date(2025, m, 10)) for m in range(1, 13) for _ in range(5)]
    lo, hi = cluster_bootstrap_ci(trades)
    assert (lo, hi) == cluster_bootstrap_ci(trades) and lo < sum(t.net_pnl for t in trades) / len(trades) < hi
    assert cluster_bootstrap_ci(trades, seed=99) != (lo, hi)                      # the seed matters; same seed repeats


def _many(per_month_values, months=14, seed=None):
    """``per_month_values`` repeated each month; with ``seed`` each value gets noise so months genuinely differ."""
    rng = random.Random(seed)
    jitter = (lambda: rng.gauss(0, 15)) if seed is not None else (lambda: 0.0)
    return [tr(v + jitter(), date(2024 + (m // 12), m % 12 + 1, 10)) for m in range(months) for v in per_month_values]


def test_verdict_demands_enough_trades_and_months_before_concluding():
    few = summarize(_many([50, 50, 50]))                           # 42 trades over 14 months: great numbers, too few
    assert "INCONCLUSIVE" in few.verdict and str(MIN_TRADES) in few.verdict
    short = summarize([tr(50, date(2024, 1, 10)) for _ in range(MIN_TRADES + 5)])
    assert "INCONCLUSIVE" in short.verdict and str(MIN_MONTHS) in short.verdict


def test_verdict_positive_negative_and_indistinguishable():
    pos = summarize(_many([40, 35, 45, 30, 38, 42, 36, 44], seed=1))   # 112 trades, 14 months, consistently positive
    assert pos.trades >= MIN_TRADES and "Positive expectancy supported" in pos.verdict and pos.ci_low > 0
    neg = summarize(_many([-40, -35, -45, -30, -38, -42, -36, -44], seed=2))
    assert "NEGATIVE" in neg.verdict and neg.ci_high < 0
    noise = summarize(_many([60, -55, 70, -65, 50, -48, 62, -64], seed=3))
    assert "NOT DISTINGUISHABLE" in noise.verdict and noise.ci_low < 0 < noise.ci_high


def test_drawdown_breakdowns_and_split():
    curve = [(D0, 10.0), (D0 + timedelta(1), 50.0), (D0 + timedelta(2), -30.0), (D0 + timedelta(3), 20.0)]
    assert max_drawdown(curve) == 80.0 and max_drawdown([]) == 0.0
    trades = [tr(50, D0, symbol="SPY"), tr(-100, D0 + timedelta(40), symbol="QQQ", st="BEAR_CALL")]
    by = {g["group"]: g for g in breakdown(trades, lambda t: t.spread_type)}
    assert by["BULL_PUT"]["win_rate"] == 1.0 and by["BEAR_CALL"]["net_pnl"] == -100
    buckets = [iv_hv_bucket(tr(1, iv_hv=v)) for v in (None, 1.0, 1.1, 1.3, 1.5)]
    assert buckets == ["n/a", "<1.1", "1.1-1.3", "1.1-1.3", ">1.3"]
    r = run(ENTRY + day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55)))
    r.trades.extend(trades)
    ins, oos = split_by_date(r, D0 + timedelta(30))
    assert len(ins) == 2 and len(oos) == 1


# --- synthetic data + end-to-end ---
def test_synthetic_data_is_reproducible_and_wellformed():
    a, b = generate_frame(days=40, seed=3), generate_frame(days=40, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_frame(days=40, seed=4))
    assert (a["bid"] <= a["ask"]).all() and (a["ask"] > 0).all() and set(a["right"]) == {"P", "C"}
    days = trading_days(date(2024, 1, 2), 10)
    assert len(days) == 10 and all(d.weekday() < 5 for d in days)
    assert generate_frame(("SPY", "QQQ"), days=5)["symbol"].nunique() == 2


def test_end_to_end_replay_on_synthetic_chains_obeys_the_invariants():
    store = from_frame(generate_frame(days=130, seed=2))
    r = run_backtest(store, BacktestConfig(scanner=ScannerConfig(pricing="natural"), equity=100_000.0))
    assert r.trades, "the permissive scanner should trade on 130 days of synthetic chains"
    for t in r.trades:
        assert t.close_date > t.open_date and 25 <= t.dte_at_entry <= 50
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs, abs=0.011)
        assert t.exit_reason in {"TAKE_PROFIT", "STOP_LOSS", "CLOSE_DTE", "EXPIRY"} and -t.max_loss - 1 <= t.gross_pnl
    assert len(r.equity_curve) == r.days == len(store.dates)


def test_report_states_limitations_verdict_and_oos_split():
    r = run(ENTRY + day(D0 + timedelta(1), (0.95, 0.95), (0.55, 0.55)))
    text = render_report(r, oos_from=D0 + timedelta(1), banner="SYNTHETIC DATA: test")
    for needle in ("# Credit spread backtest", "SYNTHETIC DATA: test", "INCONCLUSIVE",
                   "Limitations (read these first)", "End-of-day snapshots", "In-sample vs out-of-sample",
                   "By entry DTE bucket", "95% interval", "Trades (closed)"):
        assert needle in text
    assert "Candidates skipped" not in render_report(r)


# --- CLI ---
def test_cli_demo_prints_the_synthetic_banner(capsys):
    assert main(["--demo", "--demo-days", "100"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "Verdict" in out


def test_cli_reads_a_csv_and_writes_report_and_json(tmp_path, capsys):
    data = tmp_path / "chains.csv"
    generate_frame(days=100, seed=5).to_csv(data, index=False)
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    args = ["--data", str(data), "--preset", "base", "--equity", "100000", "--out", str(out_md),
            "--json", str(out_json), "--oos-from", "2024-04-01", "--max-new-per-day", "2", "--exit-pricing", "mid"]
    assert main(args) == 0
    assert "Credit spread backtest" in out_md.read_text() and "SYNTHETIC" not in out_md.read_text()
    payload = json.loads(out_json.read_text())
    assert "verdict" in payload["summary"] and isinstance(payload["trades"], list)
    capsys.readouterr()


def test_cli_errors_exit_with_code_2(tmp_path, capsys):
    assert main([]) == 2
    assert main(["--data", str(tmp_path / "nope.csv")]) == 2
    assert main(["--demo", "--oos-from", "not-a-date"]) == 2
    assert main(["--demo", "--demo-days", "30", "--earnings", str(tmp_path / "nope.csv")]) == 2
    err = capsys.readouterr().err
    assert err.count("ERROR") == 4


# --- time-stop in the replay (off by default) ---
def test_time_stop_closes_a_stale_position_only_when_enabled():
    stale = day(D0 + timedelta(18), (1.05, 1.05), (0.20, 0.20))        # debit 0.85 = 11% profit, 51% of life, DTE 17
    r_off, none = trade_after(stale)
    assert none is None and r_off.still_open == 1
    r_on, t = trade_after(stale, limits=RiskLimits(time_stop_fraction=0.5))
    assert t.exit_reason == "TIME_STOP" and t.close_date == D0 + timedelta(18) and r_on.still_open == 0
    early = day(D0 + timedelta(10), (1.05, 1.05), (0.20, 0.20))
    assert trade_after(early, limits=RiskLimits(time_stop_fraction=0.5))[1] is None


# --- execution model (off by default) ---
def test_fill_improvement_defaults_are_a_no_op():
    nxt = day(D0 + timedelta(1), (0.45, 0.65), (0.20, 0.30))
    base = trade_after(nxt)[1]
    zero = trade_after(nxt, entry_fill_improvement=0.0, exit_fill_improvement=0.0)[1]
    assert (base.credit, base.exit_debit, base.net_pnl) == (zero.credit, zero.exit_debit, zero.net_pnl)


def test_entry_fill_moves_toward_the_mid_and_adverse_selection_gives_part_back():
    nxt = day(D0 + timedelta(1), (0.45, 0.65), (0.20, 0.30))           # entry: natural credit 0.96, mid credit 1.00
    assert trade_after(nxt, entry_fill_improvement=0.5)[1].credit == pytest.approx(0.98)
    assert trade_after(nxt, entry_fill_improvement=1.0)[1].credit == pytest.approx(1.00)
    assert trade_after(nxt, entry_fill_improvement=1.0, adverse_selection_pct=0.5)[1].credit == pytest.approx(0.98)


def test_exit_fill_moves_toward_the_mid_but_a_stop_always_crosses():
    nxt = day(D0 + timedelta(1), (0.45, 0.65), (0.20, 0.30))           # close debit: natural 0.45, mid 0.30
    assert trade_after(nxt, exit_fill_improvement=0.5)[1].exit_debit == pytest.approx(0.375)
    stop = day(D0 + timedelta(1), (3.9, 3.9), (0.9, 0.9))
    t = trade_after(stop, exit_fill_improvement=1.0, fill_prob_at_mid=0.0)[1]
    assert t.exit_reason == "STOP_LOSS" and t.exit_debit == pytest.approx(3.0)


def test_unfilled_limit_orders_skip_entries_and_retry_exits():
    r = run(ENTRY, entry_fill_improvement=1.0, fill_prob_at_mid=0.0)
    assert r.trades == [] and r.still_open == 0 and r.skipped["entry_not_filled"] >= 1
    nxt = day(D0 + timedelta(1), (0.45, 0.65), (0.20, 0.30))
    r2 = run(ENTRY + nxt, exit_fill_improvement=1.0, fill_prob_at_mid=0.0)
    assert r2.trades == [] and r2.still_open == 1 and r2.skipped["exit_not_filled"] == 1


def test_fill_model_is_deterministic_per_seed_and_validated():
    a = run(ENTRY, entry_fill_improvement=0.5, fill_prob_at_mid=0.5, fill_seed=3)
    b = run(ENTRY, entry_fill_improvement=0.5, fill_prob_at_mid=0.5, fill_seed=3)
    assert (a.still_open, a.skipped) == (b.still_open, b.skipped)
    for bad in ({"entry_fill_improvement": 1.5}, {"fill_prob_at_mid": -0.1}, {"adverse_selection_pct": 2.0}):
        with pytest.raises(ValueError):
            BacktestConfig(**bad)
