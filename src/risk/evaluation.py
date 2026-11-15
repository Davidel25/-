"""Shared, I/O-light position evaluation used by both the ``ScanService`` and the ``src.risk.report`` CLI.

``evaluate_position`` is pure. ``enrich_positions`` only *reads* from the broker (spot prices and option
chains); nothing in this module can place an order."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum

from src.broker.base import BrokerClient, BrokerError
from src.data.greeks import bs_delta
from src.models import OptionQuote, SpreadPosition
from src.risk.earnings import EarningsInfo, EarningsProvider
from src.risk.manager import ExitAction, RiskAlert, RiskManager, Severity

RISK_FREE = 0.045


class Status(StrEnum):
    ACTION = "ACTION"  # close / roll / take profit now
    REVIEW = "REVIEW"  # needs a human look or data is missing
    OK = "OK"


@dataclass(frozen=True)
class PositionReport:
    position: SpreadPosition
    action: ExitAction
    alerts: list[RiskAlert]
    notes: list[str] = field(default_factory=list)  # data gaps (fail-safe: they force REVIEW)
    scale_out_contracts: int = 0                    # contracts to close when action is SCALE_OUT

    @property
    def status(self) -> Status:
        if self.action is not ExitAction.HOLD or any(a.severity is Severity.HIGH for a in self.alerts):
            return Status.ACTION
        return Status.REVIEW if (self.alerts or self.notes) else Status.OK

    @property
    def recommendation(self) -> str:
        if self.action is ExitAction.SCALE_OUT:
            n, total = self.scale_out_contracts, self.position.contracts
            return (f"Scale out: close {n} of {total} contracts at the working limit and let the other "
                    f"{total - n} run to the profit target or the DTE exit. Remaining contracts keep full max loss.")
        if self.action is ExitAction.TAKE_PROFIT:
            return "Take profit: buy the spread back at the working limit."
        if self.action is ExitAction.STOP_LOSS:
            return "Stop hit: close the spread (loss >= stop multiple of credit)."
        if self.action is ExitAction.CLOSE_DTE:
            return "DTE cutoff: close or roll (for a credit) to avoid final-days gamma."
        if self.status is Status.ACTION:
            return "Act now: close, or roll out/away for a net credit only."
        if self.status is Status.REVIEW:
            return "Review manually before the next session."
        return "Hold."


def evaluate_position(risk: RiskManager, pos: SpreadPosition, today: date, earnings: EarningsProvider | None,
                      notes: list[str] | None = None) -> PositionReport:
    """Pure: no I/O. ``earnings=None`` disables the earnings check."""
    info: EarningsInfo | None = earnings.get(pos.symbol) if earnings else None
    alerts = risk.evaluate_alerts(pos, today, check_earnings=earnings is not None, earnings=info)
    return PositionReport(pos, risk.evaluate_exit(pos, today), alerts, list(notes or []),
                          risk.scale_out_contracts(pos))


def find_short_quote(chain: list[OptionQuote], pos: SpreadPosition) -> OptionQuote | None:
    right = pos.spread_type.right
    return next((q for q in chain if q.expiry == pos.expiry and q.right == right and q.strike == pos.short_strike),
                None)


def find_long_quote(chain: list[OptionQuote], pos: SpreadPosition) -> OptionQuote | None:
    right = pos.spread_type.right
    return next((q for q in chain if q.expiry == pos.expiry and q.right == right and q.strike == pos.long_strike), None)


def quote_delta(quote: OptionQuote, today: date) -> float | None:
    """|delta| from the broker if present, else Black-Scholes from the quote's own IV, else None."""
    if quote.delta is not None:
        return abs(quote.delta)
    if quote.iv and quote.dte(today) > 0:
        return abs(bs_delta(quote.underlying_price, quote.strike, quote.dte(today) / 365.0, RISK_FREE,
                            quote.iv, quote.right))
    return None


async def enrich_positions(broker: BrokerClient, positions: list[SpreadPosition], today: date,
                           entry_deltas: dict[str, float]) -> list[tuple[SpreadPosition, list[str]]]:
    """Attach underlying price, current short delta and entry delta. Any gap becomes a note, never a guess."""
    out: list[tuple[SpreadPosition, list[str]]] = []
    spots: dict[str, float] = {}
    for pos in positions:
        notes: list[str] = []
        try:
            if pos.symbol not in spots:
                spots[pos.symbol] = await broker.get_underlying_price(pos.symbol)
            pos = replace(pos, underlying_price=spots[pos.symbol])
            dte = pos.dte(today)
            chain = await broker.get_option_chain(pos.symbol, min_dte=max(dte - 1, 0), max_dte=dte + 1)
            quote = find_short_quote(chain, pos)
            delta = quote_delta(quote, today) if quote else None
            if delta is None:
                notes.append("short-leg delta unavailable: delta alert skipped")
            long_q = find_long_quote(chain, pos)
            pos = replace(pos, short_delta=delta, long_delta=quote_delta(long_q, today) if long_q else None,
                          short_iv=quote.iv if quote else None, long_iv=long_q.iv if long_q else None)
        except BrokerError as exc:
            notes.append(f"market data unavailable ({exc}): delta/proximity checks skipped")
        entry = entry_deltas.get(pos.key)
        if entry is None:
            notes.append(f"no entry delta journaled for {pos.key}: delta alert skipped")
        out.append((replace(pos, entry_short_delta=entry), notes))
    return out
