"""Risk rules: pre-trade exposure limits and post-trade exit rules (50% profit target)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 0.02   # of equity
    max_total_risk_pct: float = 0.20       # all open spreads' max loss, of equity
    max_open_positions: int = 10
    max_positions_per_symbol: int = 2
    profit_target_pct: float = 0.50        # close at 50% of credit captured
    stop_loss_multiple: float = 2.0        # close when loss >= 2x credit received
    close_at_dte: int = 14                 # gamma-risk exit


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]


class ExitAction(StrEnum):
    HOLD = "HOLD"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    CLOSE_DTE = "CLOSE_DTE"


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def check_signal(self, signal: SpreadSignal, positions: list[SpreadPosition],
                     equity: float, contracts: int = 1) -> RiskDecision:
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
        return RiskDecision(not reasons, reasons)

    def evaluate_exit(self, pos: SpreadPosition, today: date | None = None) -> ExitAction:
        lim = self.limits
        if pos.profit_pct >= lim.profit_target_pct:
            return ExitAction.TAKE_PROFIT
        if pos.entry_credit > 0 and (pos.current_debit - pos.entry_credit) >= lim.stop_loss_multiple * pos.entry_credit:
            return ExitAction.STOP_LOSS
        if pos.dte(today) <= lim.close_at_dte:
            return ExitAction.CLOSE_DTE
        return ExitAction.HOLD


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
