"""Earnings-date data model, provider interface and the conservative window check.

Design notes
------------
* Pure logic only: nothing here touches the network. A provider is anything with ``get(symbol)``.
* The IBKR MCP / TWS API has no dependable earnings calendar, and scraping a public site is brittle, so the
  shipped provider reads a small JSON file you maintain from the company's investor-relations release
  (``JsonFile`` below). Plug in a vendor feed by implementing :class:`EarningsProvider`.
* **Fail safe:** a missing, stale or unconfirmed date is never treated as "clear". Only an ETF/index
  (no earnings), or a *confirmed* date safely after expiry, returns :attr:`EarningsCheck.CLEAR`.

File format (``--earnings-file``)::

    {
      "AAPL": {"date": "2026-10-29", "confirmed": true},
      "TJX":  {"date": "2026-11-18", "confirmed": false},
      "SPY":  {"etf": true}
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

# Symbols with no single-company earnings event. Extend via StaticEarningsProvider(no_earnings=...).
DEFAULT_NO_EARNINGS = frozenset({"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "TLT", "GLD", "SMH"})

# An *unconfirmed* (projected) date can move earlier; within this many days after expiry it still
# gets a manual-review flag instead of being treated as clear.
UNCONFIRMED_BUFFER_DAYS = 14


class EarningsStatus(StrEnum):
    CONFIRMED = "CONFIRMED"            # company-announced date
    PROJECTED = "PROJECTED"            # estimated from history; may shift
    UNKNOWN = "UNKNOWN"                # no data
    NOT_APPLICABLE = "NOT_APPLICABLE"  # ETF / index: no earnings event


class EarningsCheck(StrEnum):
    CLEAR = "CLEAR"              # safe: no earnings before expiry
    RISK = "RISK"                # earnings fall between today and expiry
    UNCONFIRMED = "UNCONFIRMED"  # date outside the window but not confirmed, close enough to shift inside
    UNKNOWN = "UNKNOWN"          # no usable date -> manual review


@dataclass(frozen=True)
class EarningsInfo:
    symbol: str
    date: date | None
    status: EarningsStatus


class EarningsProvider(Protocol):
    def get(self, symbol: str) -> EarningsInfo: ...


def classify_earnings(info: EarningsInfo | None, expiry: date, today: date,
                      unconfirmed_buffer_days: int = UNCONFIRMED_BUFFER_DAYS) -> EarningsCheck:
    """Compare the next earnings date with an option expiry. Window is inclusive of both ends
    (``today <= earnings <= expiry``) because the report time (pre-market vs after-close) is not known."""
    if info is None or info.status is EarningsStatus.UNKNOWN:
        return EarningsCheck.UNKNOWN
    if info.status is EarningsStatus.NOT_APPLICABLE:
        return EarningsCheck.CLEAR
    if info.date is None or info.date < today:
        return EarningsCheck.UNKNOWN  # no date, or a stale one: the real next date is not known
    if info.date <= expiry:
        return EarningsCheck.RISK  # confirmed or projected, in either case it lands inside the trade
    if info.status is EarningsStatus.CONFIRMED:
        return EarningsCheck.CLEAR
    if info.date <= expiry + timedelta(days=unconfirmed_buffer_days):
        return EarningsCheck.UNCONFIRMED
    return EarningsCheck.CLEAR


class StaticEarningsProvider:
    """In-memory provider: explicit entries plus a set of symbols known to have no earnings."""

    def __init__(self, entries: dict[str, EarningsInfo] | None = None,
                 no_earnings: frozenset[str] | set[str] = DEFAULT_NO_EARNINGS) -> None:
        self._entries = {k.upper(): v for k, v in (entries or {}).items()}
        self._no_earnings = {s.upper() for s in no_earnings}

    def get(self, symbol: str) -> EarningsInfo:
        sym = symbol.upper()
        if sym in self._entries:
            return self._entries[sym]
        if sym in self._no_earnings:
            return EarningsInfo(sym, None, EarningsStatus.NOT_APPLICABLE)
        return EarningsInfo(sym, None, EarningsStatus.UNKNOWN)


class JsonFile(StaticEarningsProvider):
    """Provider backed by a hand-maintained JSON file (format in the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read earnings file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"earnings file {path} must be a JSON object keyed by symbol")
        entries: dict[str, EarningsInfo] = {}
        no_earnings = set(DEFAULT_NO_EARNINGS)
        for sym, row in raw.items():
            sym = sym.upper()
            if not isinstance(row, dict):
                raise ValueError(f"earnings file {path}: entry for {sym} must be an object")
            if row.get("etf"):
                no_earnings.add(sym)
                continue
            try:
                when = date.fromisoformat(row["date"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"earnings file {path}: bad or missing 'date' for {sym}") from exc
            status = EarningsStatus.CONFIRMED if row.get("confirmed") else EarningsStatus.PROJECTED
            entries[sym] = EarningsInfo(sym, when, status)
        super().__init__(entries, no_earnings)
