"""Backtest statistics with honest uncertainty.

The headline question is "is the mean net P&L per trade distinguishable from zero?". A plain t-test would overstate
confidence because trades overlap and cluster (same market regime, same month), so the interval here comes from a
CLUSTER bootstrap: whole entry-months are resampled together. With few months the interval is wide, which is correct.

``verdict`` is deliberately conservative: fewer than ``MIN_TRADES`` trades or ``MIN_MONTHS`` distinct entry months is
reported as inconclusive no matter how good the numbers look."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from src.backtest.engine import BacktestResult, Trade

MIN_TRADES = 100
MIN_MONTHS = 12
RESAMPLES = 2000


@dataclass(frozen=True)
class Summary:
    trades: int
    wins: int
    net_pnl: float
    gross_pnl: float
    costs: float
    mean_net: float
    mean_pct_of_risk: float | None
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    worst: float | None
    best: float | None
    ci_low: float | None
    ci_high: float | None
    months: int
    max_drawdown: float
    annualized_return: float | None
    avg_hold_days: float | None
    breakeven_win_rate: float | None
    sharpe: float | None = None

    @property
    def verdict(self) -> str:
        if self.trades == 0:
            return "NO TRADES: nothing to evaluate (check filters, data coverage and earnings file)."
        if self.trades < MIN_TRADES or self.months < MIN_MONTHS:
            return (f"INCONCLUSIVE: {self.trades} trades over {self.months} entry month(s); need at least {MIN_TRADES} "
                    f"trades across {MIN_MONTHS}+ months before any conclusion.")
        if self.ci_low is not None and self.ci_low > 0:
            return ("Positive expectancy supported IN THIS SAMPLE (95% cluster-bootstrap interval above zero). "
                    "Not proof out of sample.")
        if self.ci_high is not None and self.ci_high < 0:
            return "NEGATIVE expectancy in this sample (95% interval below zero)."
        return "NOT DISTINGUISHABLE FROM ZERO: the 95% interval for mean net P&L per trade includes zero."


def _month(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def cluster_bootstrap_ci(trades: list[Trade], seed: int = 7, resamples: int = RESAMPLES) -> tuple[float, float] | None:
    """95% interval for the mean net P&L per trade, resampling whole entry-months. None if fewer than 2 months."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        groups[_month(t.open_date)].append(t.net_pnl)
    clusters = list(groups.values())
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [rng.choice(clusters) for _ in clusters]
        n = sum(len(c) for c in draw)
        means.append(sum(sum(c) for c in draw) / n)
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def daily_sharpe(curve: list[tuple[date, float]], min_days: int = 20) -> float | None:
    """Annualised Sharpe of the daily change in the (realized + open mark-to-market) P&L curve, sqrt(252) scaling, no
    risk-free subtraction. The ratio is scale-free, so it does not depend on the account size used.

    Read it with care for short-premium strategies: they earn small steady gains and occasional large losses
    (negative skew), which a Sharpe ratio understates until a loss arrives. Always look at max drawdown and the worst
    trade too."""
    if len(curve) < min_days + 1:
        return None
    diffs = [b[1] - a[1] for a, b in zip(curve, curve[1:], strict=False)]
    mean = sum(diffs) / len(diffs)
    var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    return None if var <= 1e-12 else mean / math.sqrt(var) * math.sqrt(252)


def max_drawdown(curve: list[tuple[date, float]]) -> float:
    """Largest peak-to-trough fall of the realized + open mark-to-market curve, in dollars (a positive number)."""
    peak, worst = 0.0, 0.0
    for _, v in curve:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst


def summarize(trades: list[Trade], curve: list[tuple[date, float]] | None = None, equity: float = 20_000.0,
              first: date | None = None, last: date | None = None, seed: int = 7) -> Summary:
    n = len(trades)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    net = sum(t.net_pnl for t in trades)
    gw, gl = sum(t.net_pnl for t in wins), -sum(t.net_pnl for t in losses)
    ci = cluster_bootstrap_ci(trades, seed) if n else None
    years = ((last - first).days / 365.25) if first and last and last > first else None
    risk = sum(t.max_loss for t in trades)
    avg_win = gw / len(wins) if wins else None
    avg_loss = -gl / len(losses) if losses else None
    be = (abs(avg_loss) / (avg_win + abs(avg_loss))) if avg_win and avg_loss else None
    return Summary(
        trades=n, wins=len(wins), net_pnl=round(net, 2), gross_pnl=round(sum(t.gross_pnl for t in trades), 2),
        costs=round(sum(t.costs for t in trades), 2), mean_net=round(net / n, 2) if n else 0.0,
        mean_pct_of_risk=(net / risk) if risk else None, win_rate=len(wins) / n if n else None,
        avg_win=avg_win, avg_loss=avg_loss, profit_factor=(gw / gl if gl else (math.inf if gw else None)),
        worst=min((t.net_pnl for t in trades), default=None), best=max((t.net_pnl for t in trades), default=None),
        ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None, months=len({_month(t.open_date) for t in trades}),
        max_drawdown=round(max_drawdown(curve or []), 2),
        annualized_return=(net / equity / years) if years and equity else None,
        avg_hold_days=sum(t.held_days for t in trades) / n if n else None, breakeven_win_rate=be,
        sharpe=daily_sharpe(curve) if curve else None)


def breakdown(trades: list[Trade], key) -> list[dict]:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[str(key(t))].append(t)
    out = []
    for g in sorted(groups):
        v = groups[g]
        net = sum(t.net_pnl for t in v)
        out.append({"group": g, "trades": len(v), "win_rate": sum(t.net_pnl > 0 for t in v) / len(v),
                    "net_pnl": round(net, 2), "mean_net": round(net / len(v), 2)})
    return out


def iv_hv_bucket(t: Trade) -> str:
    if t.iv_hv is None:
        return "n/a"
    return "<1.1" if t.iv_hv < 1.1 else "1.1-1.3" if t.iv_hv <= 1.3 else ">1.3"


def split_by_date(result: BacktestResult, oos_from: date) -> tuple[list[Trade], list[Trade]]:
    """In-sample (opened before ``oos_from``) and out-of-sample trades."""
    return ([t for t in result.trades if t.open_date < oos_from], [t for t in result.trades if t.open_date >= oos_from])
