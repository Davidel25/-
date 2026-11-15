"""Run several rule sets against the SAME data and summarise each with the metrics that matter for a short-premium
strategy: win rate, payoff, tail frequency, Sharpe, drawdown, friction. Because every variant sees the identical price
path, differences between variants are far less noisy than differences between separate datasets."""
from __future__ import annotations

from dataclasses import dataclass

from src.backtest.data import ChainStore
from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtest.metrics import Summary, summarize


@dataclass(frozen=True)
class VariantStats:
    name: str
    summary: Summary
    trades_per_year: float | None
    stop_rate: float | None            # share of trades closed by the stop-loss
    target_rate: float | None          # share closed by the profit target
    avg_credit_over_width: float | None
    avg_short_delta: float | None
    p05_trade: float | None            # 5th-percentile trade P&L: the typical "bad day"
    cost_share: float | None           # costs as a share of gross P&L (friction), when gross > 0

    def flat(self) -> dict:
        s = self.summary
        return {
            "trades": s.trades, "trades_per_year": self.trades_per_year, "win_rate": s.win_rate,
            "mean_net": s.mean_net, "mean_pct_risk": s.mean_pct_of_risk, "avg_win": s.avg_win,
            "avg_loss": s.avg_loss, "profit_factor": s.profit_factor, "sharpe": s.sharpe,
            "max_drawdown": s.max_drawdown, "annualized_return": s.annualized_return, "stop_rate": self.stop_rate,
            "target_rate": self.target_rate, "credit_over_width": self.avg_credit_over_width,
            "short_delta": self.avg_short_delta, "p05_trade": self.p05_trade, "avg_hold_days": s.avg_hold_days,
            "worst": s.worst, "cost_share": self.cost_share}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(q * len(v))))]


def variant_stats(name: str, result: BacktestResult) -> VariantStats:
    t, cfg = result.trades, result.config
    s = summarize(t, result.equity_curve, cfg.equity, result.first_date, result.last_date)
    years = (result.last_date - result.first_date).days / 365.25
    n = len(t)
    gross = sum(x.gross_pnl for x in t)
    return VariantStats(
        name=name, summary=s, trades_per_year=(n / years) if years > 0 else None,
        stop_rate=(sum(x.exit_reason == "STOP_LOSS" for x in t) / n) if n else None,
        target_rate=(sum(x.exit_reason == "TAKE_PROFIT" for x in t) / n) if n else None,
        avg_credit_over_width=(sum(x.credit / (x.max_loss / x.contracts / 100 + x.credit) for x in t) / n)
        if n else None,
        avg_short_delta=(sum(x.entry_short_delta for x in t) / n) if n else None,
        p05_trade=_percentile([x.net_pnl for x in t], 0.05),
        cost_share=(sum(x.costs for x in t) / gross) if gross > 0 else None)


def run_variants(store: ChainStore, variants: dict[str, BacktestConfig]) -> dict[str, VariantStats]:
    return {name: variant_stats(name, run_backtest(store, cfg)) for name, cfg in variants.items()}
