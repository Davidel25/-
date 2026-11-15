"""Markdown report for a backtest, with the limitations stated up front rather than buried."""
from __future__ import annotations

from datetime import date

from src.backtest.engine import BacktestResult
from src.backtest.metrics import (
    MIN_MONTHS,
    MIN_TRADES,
    Summary,
    breakdown,
    iv_hv_bucket,
    split_by_date,
    summarize,
)

LIMITS = (
    "End-of-day snapshots: signals and fills share the same EOD quotes; real fills come later and usually worse.",
    "Entries and exits cross the spread on every leg (natural pricing); no queue position, partial fills or impact.",
    "No early assignment, dividends or pin risk; positions that reach expiry settle at intrinsic value.",
    "Earnings dates are the ones you supplied, known with hindsight; without them single stocks are not traded.",
    "Trades overlap and cluster by regime, so the effective sample is smaller than the trade count "
    "(the interval below resamples whole months to account for that).",
    "The filters were chosen after reading about this strategy: in-sample results flatter it. Prefer the "
    "out-of-sample split, and treat anything under "
    f"{MIN_TRADES} trades or {MIN_MONTHS} months as inconclusive.",
)


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"{'-' if v < 0 else ''}${abs(v):,.2f}"


def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def _summary_rows(s: Summary) -> list[tuple[str, str]]:
    pf = "n/a" if s.profit_factor is None else ("inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}")
    ci = "n/a (needs 2+ entry months)" if s.ci_low is None else f"{_money(s.ci_low)} to {_money(s.ci_high)}"
    return [
        ("Trades (closed)", str(s.trades)), ("Entry months", str(s.months)),
        ("Win rate", _pct(s.win_rate)), ("Break-even win rate (from avg win/loss)", _pct(s.breakeven_win_rate)),
        ("Net P&L", _money(s.net_pnl)), ("Gross P&L / costs", f"{_money(s.gross_pnl)} / {_money(s.costs)}"),
        ("Mean net per trade", _money(s.mean_net)), ("Mean net as % of capital at risk", _pct(s.mean_pct_of_risk, 2)),
        ("95% interval for mean net per trade", ci), ("Profit factor", pf),
        ("Average win / average loss", f"{_money(s.avg_win)} / {_money(s.avg_loss)}"),
        ("Best / worst trade", f"{_money(s.best)} / {_money(s.worst)}"), ("Max drawdown", _money(s.max_drawdown)),
        ("Annualized return on account", _pct(s.annualized_return, 2)),
        ("Sharpe (daily P&L, annualised, no risk-free)", "n/a" if s.sharpe is None else f"{s.sharpe:.2f}"),
        ("Average days held", "n/a" if s.avg_hold_days is None else f"{s.avg_hold_days:.1f}"),
    ]


def _table(rows: list[tuple[str, str]]) -> str:
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def _breakdown_table(title: str, rows: list[dict]) -> str:
    if not rows:
        return ""
    body = "\n".join(f"| {r['group']} | {r['trades']} | {_pct(r['win_rate'], 0)} | {_money(r['net_pnl'])} | "
                     f"{_money(r['mean_net'])} |" for r in rows)
    return f"### {title}\n\n| Group | Trades | Win rate | Net P&L | Mean net |\n|---|---|---|---|---|\n{body}\n"


def render_report(result: BacktestResult, oos_from: date | None = None, banner: str | None = None) -> str:
    cfg, trades = result.config, result.trades
    s = summarize(trades, result.equity_curve, cfg.equity, result.first_date, result.last_date)
    sc = cfg.scanner
    out = ["# Credit spread backtest", ""]
    if banner:
        out += [f"> **{banner}**", ""]
    out += [f"**Verdict:** {s.verdict}", "",
            f"**Data:** {result.first_date} to {result.last_date} ({result.days} snapshot days); "
            f"{result.still_open} position(s) still open at the end are excluded.", ""]
    out += ["## Limitations (read these first)", ""] + [f"- {x}" for x in LIMITS]
    out += [f"- {n}" for n in result.notes] + [""]
    out += ["## Settings", "",
            f"Equity ${cfg.equity:,.0f}, {cfg.contracts} contract(s) per trade, commission "
            f"${cfg.commission_per_contract:.2f}/contract/leg, extra slippage ${cfg.slippage_per_leg:.3f}/share/fill, "
            f"exit pricing `{cfg.exit_pricing}`, at most {cfg.max_new_per_day} new trade(s) per day.", "",
            f"Scanner: DTE {sc.min_dte}-{sc.max_dte}, |short delta| {sc.min_short_delta}-{sc.max_short_delta}, "
            f"credit >= {sc.min_credit}, ROC >= {sc.min_roc:.0%}, pricing `{sc.pricing}`, "
            f"IV/HV >= {sc.min_iv_hv_ratio}, trend alignment {sc.require_trend_alignment}, "
            f"term slope >= {sc.min_term_slope}, put skew <= {sc.max_put_skew}, "
            f"weekdays {sc.allowed_weekdays}, preferred DTE {sc.preferred_dte}.", "",
            f"Risk: {cfg.limits.max_risk_per_trade_pct:.0%} of equity per trade, "
            f"{cfg.limits.max_total_risk_pct:.0%} total, max {cfg.limits.max_open_positions} positions; "
            f"exits at {cfg.limits.profit_target_pct:.0%} profit, {cfg.limits.stop_loss_multiple:g}x credit loss, "
            f"DTE <= {cfg.limits.close_at_dte}.", ""]
    out += ["## Results", "", _table(_summary_rows(s)), ""]
    if oos_from:
        ins, oos = split_by_date(result, oos_from)
        si, so = (summarize(x, None, cfg.equity) for x in (ins, oos))
        out += [f"## In-sample vs out-of-sample (split at {oos_from})", "",
                "| | In-sample | Out-of-sample |\n|---|---|---|",
                f"| Trades | {si.trades} | {so.trades} |", f"| Win rate | {_pct(si.win_rate)} | {_pct(so.win_rate)} |",
                f"| Net P&L | {_money(si.net_pnl)} | {_money(so.net_pnl)} |",
                f"| Mean net per trade | {_money(si.mean_net)} | {_money(so.mean_net)} |",
                f"| 95% interval | {_money(si.ci_low)} to {_money(si.ci_high)} | "
                f"{_money(so.ci_low)} to {_money(so.ci_high)} |",
                "", f"Out-of-sample verdict: {so.verdict}", ""]
    out += ["## Breakdowns", ""]
    for title, key in (("By strategy", lambda t: t.spread_type), ("By entry DTE bucket", lambda t: t.dte_bucket),
                       ("By entry weekday (0 = Monday)", lambda t: t.weekday), ("By IV/HV at entry", iv_hv_bucket),
                       ("By exit reason", lambda t: t.exit_reason), ("By symbol", lambda t: t.symbol)):
        out.append(_breakdown_table(title, breakdown(trades, key)))
    if result.skipped:
        out += ["## Candidates skipped", "", "| Reason | Count |\n|---|---|"] + [
            f"| {k} | {v} |" for k, v in sorted(result.skipped.items())] + [""]
    return "\n".join(out)
