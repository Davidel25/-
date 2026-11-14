"""Read-only risk-status report for open credit spreads.

Run:  python -m src.risk.report [--earnings-file earnings.json] [--entry-deltas entry_deltas.json]
      python -m src.risk.report --demo          # offline, mock broker with sample positions

What it does
    1. Connects through the project's ``BrokerClient`` (IB Gateway / TWS via ``IBClient``, which connects with
       ``readonly=True``; or the mock broker) and pulls the option positions, spot prices and option chains.
    2. Pairs the legs into vertical credit spreads and enriches each with the underlying price and the short
       leg's current delta (broker model delta, Black-Scholes fallback).
    3. Runs every rule in :class:`RiskManager`: 50% take-profit, 2x-credit stop, DTE cutoff, delta alert,
       strike-proximity alert and the earnings-window check.
    4. Prints one status block per position and exits non-zero if anything needs attention.

Safety: this module never imports or calls any order API. It only reads and recommends. The IBKR *MCP* tools
are available inside a Claude session only; a standalone script cannot call them, so it talks to the same
IB Gateway the MCP would, through ``ib_async`` in read-only mode.

Entry deltas: brokers do not keep the delta you opened a trade at. Record it in a small JSON journal keyed by
``SpreadPosition.key``, e.g. ``{"SPY:BULL_PUT:2026-12-18:480/475": 0.15}``. Without it the delta alert is
skipped and the position is marked REVIEW.

Exit codes: 0 all clear | 1 at least one position needs action or review | 2 broker/data error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path

from src.broker.base import BrokerClient, BrokerError
from src.broker.factory import create_broker
from src.config import Settings
from src.data.greeks import bs_delta
from src.models import OptionQuote, PositionLeg, SpreadPosition
from src.risk.earnings import EarningsInfo, EarningsProvider, JsonFile, StaticEarningsProvider
from src.risk.manager import ExitAction, RiskAlert, RiskManager, Severity, pair_legs

log = logging.getLogger("risk_report")
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

    @property
    def status(self) -> Status:
        if self.action is not ExitAction.HOLD or any(a.severity is Severity.HIGH for a in self.alerts):
            return Status.ACTION
        return Status.REVIEW if (self.alerts or self.notes) else Status.OK

    @property
    def recommendation(self) -> str:
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
    return PositionReport(pos, risk.evaluate_exit(pos, today), alerts, list(notes or []))


def find_short_quote(chain: list[OptionQuote], pos: SpreadPosition) -> OptionQuote | None:
    right = pos.spread_type.right
    return next((q for q in chain if q.expiry == pos.expiry and q.right == right and q.strike == pos.short_strike),
                None)


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
            pos = replace(pos, short_delta=delta)
        except BrokerError as exc:
            notes.append(f"market data unavailable ({exc}): delta/proximity checks skipped")
        entry = entry_deltas.get(pos.key)
        if entry is None:
            notes.append(f"no entry delta journaled for {pos.key}: delta alert skipped")
        out.append((replace(pos, entry_short_delta=entry), notes))
    return out


def format_report(reports: list[PositionReport], today: date) -> str:
    lines = [f"CREDIT SPREAD RISK REPORT  {today}  (READ-ONLY: no orders are placed)", "=" * 78]
    for r in sorted(reports, key=lambda x: (x.status is Status.OK, x.status is Status.REVIEW, x.position.symbol)):
        p = r.position
        und = f"{p.underlying_price:g}" if p.underlying_price is not None else "n/a"
        dist = f" ({p.distance_to_short_pct:+.1%} to short)" if p.distance_to_short_pct is not None else ""
        delta = f"{p.short_delta:.2f}" if p.short_delta is not None else "n/a"
        entry = f"{p.entry_short_delta:.2f}" if p.entry_short_delta is not None else "n/a"
        lines += [
            f"[{r.status.value:6}] {p.symbol} {p.spread_type.value} {p.short_strike:g}/{p.long_strike:g} "
            f"x{p.contracts}  exp {p.expiry} (DTE {p.dte(today)})",
            f"         credit {p.entry_credit:.2f} -> debit {p.current_debit:.2f} | P&L ${p.unrealized_pnl:.0f} "
            f"({p.profit_pct:.0%} of credit) | max loss ${p.max_loss:.0f}",
            f"         underlying {und}{dist} | short delta {delta} (entry {entry}) | exit rule: {r.action.value}",
        ]
        lines += [f"         ! [{a.severity.value}] {a.type.value}: {a.message}" for a in r.alerts]
        lines += [f"         ? {n}" for n in r.notes]
        lines.append(f"         -> {r.recommendation}")
        lines.append("")
    counts = {s: sum(r.status is s for r in reports) for s in Status}
    lines.append(f"SUMMARY: {len(reports)} positions | ACTION {counts[Status.ACTION]} | "
                 f"REVIEW {counts[Status.REVIEW]} | OK {counts[Status.OK]}")
    return "\n".join(lines)


def load_entry_deltas(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text())
        return {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"cannot read entry-delta journal {path}: {exc}") from exc


async def build_reports(broker: BrokerClient, risk: RiskManager, today: date, earnings: EarningsProvider | None,
                        entry_deltas: dict[str, float]) -> list[PositionReport]:
    positions = pair_legs(await broker.get_option_positions())
    enriched = await enrich_positions(broker, positions, today, entry_deltas)
    return [evaluate_position(risk, p, today, earnings, notes) for p, notes in enriched]


async def run(broker: BrokerClient, risk: RiskManager | None = None, *, today: date | None = None,
              earnings: EarningsProvider | None = None, entry_deltas: dict[str, float] | None = None) -> int:
    """Connect, report, disconnect. Returns the process exit code (0 clear, 1 attention, 2 broker error)."""
    today = today or date.today()
    try:
        await broker.connect()
        reports = await build_reports(broker, risk or RiskManager(), today, earnings, entry_deltas or {})
    except BrokerError as exc:
        print(f"BROKER ERROR: {exc}\nIs IB Gateway/TWS running and logged in (2FA approved)? No report produced.",
              file=sys.stderr)
        return 2
    finally:
        try:
            await broker.disconnect()
        except Exception as exc:  # noqa: BLE001 - never mask the real error with a failed disconnect
            log.debug("disconnect failed: %s", exc)
    print(format_report(reports, today) if reports else f"No open credit spreads found ({today}).")
    return 0 if all(r.status is Status.OK for r in reports) else 1


def _demo_broker(today: date) -> BrokerClient:
    from src.broker.mock_client import MockBroker
    exp = today + timedelta(days=30)
    exp += timedelta(days=(4 - exp.weekday()) % 7)  # the mock broker lists Friday expiries only
    legs = [
        PositionLeg("SPY", exp, 480, "P", -1, 2.00, 0.95), PositionLeg("SPY", exp, 475, "P", 1, 1.00, 0.45),  # winner
        PositionLeg("QQQ", exp, 430, "C", -1, 3.00, 6.20), PositionLeg("QQQ", exp, 435, "C", 1, 2.00, 3.10),  # tested
        PositionLeg("AAPL", exp, 175, "P", -1, 2.00, 1.80), PositionLeg("AAPL", exp, 170, "P", 1, 1.00, 0.85),
    ]
    return MockBroker(positions=legs, today=today)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only credit-spread risk report (never places orders).")
    ap.add_argument("--earnings-file", help="JSON earnings calendar (see src/risk/earnings.py)")
    ap.add_argument("--entry-deltas", help="JSON journal {position key: entry |delta|}")
    ap.add_argument("--demo", action="store_true", help="offline demo with the mock broker")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        # Always run the earnings check: without a file every single stock is flagged for manual review.
        earnings: EarningsProvider = JsonFile(args.earnings_file) if args.earnings_file else StaticEarningsProvider()
        deltas = load_entry_deltas(args.entry_deltas)
        settings = Settings.from_env()
        broker = _demo_broker(date.today()) if args.demo else create_broker(settings)
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    return asyncio.run(run(broker, earnings=earnings, entry_deltas=deltas))


if __name__ == "__main__":
    raise SystemExit(main())
