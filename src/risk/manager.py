"""Risk rules: pre-trade exposure limits and post-trade exit rules (50% profit target)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType
from src.risk.earnings import EarningsCheck, EarningsInfo, classify_earnings


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 0.02   # of equity
    max_total_risk_pct: float = 0.20       # all open spreads' max loss, of equity
    max_open_positions: int = 10
    max_positions_per_symbol: int = 2
    profit_target_pct: float = 0.50        # close at 50% of credit captured
    stop_loss_multiple: float = 2.0        # close when loss >= 2x credit received
    close_at_dte: int = 14                 # gamma-risk exit
    delta_alert_multiple: float = 2.0      # warn when short |delta| >= 2x its entry value
    strike_proximity_pct: float = 0.01     # warn when underlying is within 1% of (or through) the short strike
    # Optional partial exit. OFF by default (None): it needs >= 2 contracts and, per the simulation in the strategy
    # review, it lowers variance slightly but does NOT raise expectancy and is NOT "risk-free" (the remaining
    # contracts keep their full max loss). Typical values: 0.30 or 0.40.
    scale_out_profit_pct: float | None = None
    scale_out_fraction: float = 0.5        # share of contracts to close at the scale-out trigger


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]


class ExitAction(StrEnum):
    HOLD = "HOLD"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    CLOSE_DTE = "CLOSE_DTE"
    SCALE_OUT = "SCALE_OUT"  # advisory: close part of the position, let the rest run to the 50% target / DTE exit


class AlertType(StrEnum):
    """Advisory flags raised alongside (never instead of) an :class:`ExitAction`. Read-only: nothing here trades."""

    DELTA_ALERT = "DELTA_ALERT"
    STRIKE_PROXIMITY = "STRIKE_PROXIMITY"
    EARNINGS_RISK_ALERT = "EARNINGS_RISK_ALERT"
    EARNINGS_UNKNOWN = "EARNINGS_UNKNOWN"


class Severity(StrEnum):
    WARN = "WARN"   # needs a human look
    HIGH = "HIGH"   # needs action now: close, roll or take profit


@dataclass(frozen=True)
class RiskAlert:
    type: AlertType
    severity: Severity
    message: str


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def check_signal(self, signal: SpreadSignal, positions: list[SpreadPosition],
                     equity: float, contracts: int = 1, *, check_earnings: bool = False,
                     earnings: EarningsInfo | None = None, today: date | None = None) -> RiskDecision:
        """Pre-trade gate. With ``check_earnings=True`` a candidate is rejected unless its earnings
        status is CLEAR (unknown/unconfirmed dates are rejected too: fail safe)."""
        lim, reasons = self.limits, []
        risk = signal.max_loss * contracts
        open_risk = sum(p.max_loss for p in positions)
        if equity <= 0:
            return RiskDecision(False, ["account equity unavailable"])
        if risk > equity * lim.max_risk_per_trade_pct:
            reasons.append(f"trade risk ${risk:.0f} > {lim.max_risk_per_trade_pct:.1%} of equity")
        if open_risk + risk > equity * lim.max_total_risk_pct:
            reasons.append(f"total risk ${open_risk + risk:.0f} > {lim.max_total_risk_pct:.1%} of equity")
        if len(positions) >= lim.max_open_positions:
            reasons.append(f"already {len(positions)} open positions")
        if sum(p.symbol == signal.symbol for p in positions) >= lim.max_positions_per_symbol:
            reasons.append(f"max positions for {signal.symbol} reached")
        if check_earnings:
            check = classify_earnings(earnings, signal.expiry, today or date.today())
            if check is not EarningsCheck.CLEAR:
                reasons.append(f"earnings check {check.value} for {signal.symbol} before {signal.expiry}")
        return RiskDecision(not reasons, reasons)

    def evaluate_exit(self, pos: SpreadPosition, today: date | None = None) -> ExitAction:
        lim = self.limits
        if pos.profit_pct >= lim.profit_target_pct:
            return ExitAction.TAKE_PROFIT
        if pos.entry_credit > 0 and (pos.current_debit - pos.entry_credit) >= lim.stop_loss_multiple * pos.entry_credit:
            return ExitAction.STOP_LOSS
        if pos.dte(today) <= lim.close_at_dte:
            return ExitAction.CLOSE_DTE
        if lim.scale_out_profit_pct is not None and pos.profit_pct >= lim.scale_out_profit_pct \
                and self.scale_out_contracts(pos) > 0:
            return ExitAction.SCALE_OUT
        return ExitAction.HOLD

    def scale_out_contracts(self, pos: SpreadPosition) -> int:
        """How many contracts to close at the scale-out trigger: 0 if the rule is off, the position is a single
        contract, or it has already been scaled out (current size below its initial size); otherwise
        ``scale_out_fraction`` of the contracts, at least 1 and always leaving at least 1 open."""
        lim = self.limits
        if lim.scale_out_profit_pct is None or pos.contracts < 2:
            return 0
        if pos.initial_contracts is not None and pos.contracts < pos.initial_contracts:
            return 0
        return min(pos.contracts - 1, max(1, int(pos.contracts * lim.scale_out_fraction)))

    def evaluate_alerts(self, pos: SpreadPosition, today: date | None = None, *, check_earnings: bool = False,
                        earnings: EarningsInfo | None = None) -> list[RiskAlert]:
        """Advisory alerts for an open spread. Inputs that are unknown (``None``) skip their own check;
        the earnings check is the exception: with ``check_earnings=True`` missing data is flagged for review."""
        lim, alerts = self.limits, []

        # 1) Delta alert: short |delta| has grown to >= 2x what it was at entry.
        if pos.short_delta is not None and pos.entry_short_delta:
            now, entry = abs(pos.short_delta), abs(pos.entry_short_delta)
            if now >= lim.delta_alert_multiple * entry:
                alerts.append(RiskAlert(
                    AlertType.DELTA_ALERT, Severity.WARN,
                    f"short delta {now:.2f} is {now / entry:.1f}x entry {entry:.2f} "
                    f"(alert at {lim.delta_alert_multiple:g}x): review, roll for a credit or close"))

        # 2) Strike proximity: underlying within 1% of the short strike, or already through it.
        dist = pos.distance_to_short_pct
        if dist is not None:
            if dist <= 0:
                alerts.append(RiskAlert(
                    AlertType.STRIKE_PROXIMITY, Severity.HIGH,
                    f"underlying {pos.underlying_price:g} has BREACHED short strike {pos.short_strike:g}: "
                    "close or roll for a credit now"))
            elif dist <= lim.strike_proximity_pct + 1e-9:  # tolerance: 1.0% must alert despite float error
                alerts.append(RiskAlert(
                    AlertType.STRIKE_PROXIMITY, Severity.WARN,
                    f"underlying {pos.underlying_price:g} is {dist:.2%} from short strike {pos.short_strike:g} "
                    f"(alert within {lim.strike_proximity_pct:.0%})"))

        # 3) Earnings inside the trade's life (or date unknown/unconfirmed).
        if check_earnings:
            check = classify_earnings(earnings, pos.expiry, today or date.today())
            when = earnings.date if earnings else None
            if check is EarningsCheck.RISK:
                alerts.append(RiskAlert(
                    AlertType.EARNINGS_RISK_ALERT, Severity.HIGH,
                    f"earnings {when} fall before expiry {pos.expiry}: exit or roll past the announcement "
                    "before it prints (gap risk is unhedged beyond the long strike)"))
            elif check is EarningsCheck.UNCONFIRMED:
                alerts.append(RiskAlert(
                    AlertType.EARNINGS_UNKNOWN, Severity.WARN,
                    f"earnings {when} are only projected and close to expiry {pos.expiry}: confirm manually"))
            elif check is EarningsCheck.UNKNOWN:
                alerts.append(RiskAlert(
                    AlertType.EARNINGS_UNKNOWN, Severity.WARN,
                    f"no usable earnings date for {pos.symbol}: confirm manually before expiry {pos.expiry}"))
        return alerts


def pair_legs(legs: list[PositionLeg]) -> list[SpreadPosition]:
    """Pair short/long legs (same symbol, expiry, right) into vertical credit spreads.
    Unpaired legs are ignored (naked positions are reported elsewhere, never auto-managed)."""
    groups: dict[tuple, list[PositionLeg]] = {}
    for leg in legs:
        groups.setdefault((leg.symbol, leg.expiry, leg.right), []).append(leg)

    out: list[SpreadPosition] = []
    for (symbol, expiry, right), grp in groups.items():
        shorts = sorted((leg for leg in grp if leg.quantity < 0), key=lambda x: x.strike, reverse=(right == "P"))
        longs = [leg for leg in grp if leg.quantity > 0]
        for short in shorts:
            qty = -short.quantity
            # protective long: below the short for puts, above for calls; nearest strike first
            candidates = sorted(
                (leg for leg in longs if (leg.strike < short.strike if right == "P" else leg.strike > short.strike)),
                key=lambda x: abs(x.strike - short.strike))
            for long in candidates:
                if qty <= 0:
                    break
                n = min(qty, long.quantity)
                if n <= 0:
                    continue
                out.append(SpreadPosition(
                    spread_type=SpreadType.BULL_PUT if right == "P" else SpreadType.BEAR_CALL,
                    symbol=symbol, expiry=expiry, short_strike=short.strike, long_strike=long.strike,
                    contracts=n, entry_credit=round(abs(short.avg_price) - abs(long.avg_price), 4),
                    current_debit=round(abs(short.market_price) - abs(long.market_price), 4)))
                qty -= n
                longs[longs.index(long)] = replace(long, quantity=long.quantity - n)
    return out
