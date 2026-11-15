"""Load historical option-chain snapshots for the backtester.

Expected CSV (one row per option per day, END-OF-DAY snapshots, ideally every trading day):

    date,symbol,expiry,strike,right,bid,ask,underlying_price[,delta][,iv][,open_interest][,volume]

* ``right`` is P or C; ``iv`` is annualised (0.20 = 20%); ``delta`` is signed. A missing ``delta`` is filled with
  Black-Scholes from ``iv`` (rows with neither are dropped). Crossed or empty quotes are dropped and counted.
* Without an ``open_interest`` column the liquidity filter cannot be applied; a note is recorded and shown in the
  report.
* The 50-day trend filter and the 30-day historical-vol filter are computed from this file's own daily underlying
  prices, so the file needs at least ~50 trading days of history before the first trade you care about.

Option history is not free (CBOE DataShop, ORATS, OptionMetrics, Polygon, ...): this module only reads it."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.market_data import normalize_quote
from src.models import OptionQuote

REQUIRED = ("date", "symbol", "expiry", "strike", "right", "bid", "ask", "underlying_price")
NO_OI = 10**9  # stands in for "unknown open interest" so the liquidity filter does not silently reject everything


@dataclass
class ChainStore:
    chains: dict[date, dict[str, list[OptionQuote]]] = field(default_factory=dict)
    index: dict[tuple[date, str, date, str, float], OptionQuote] = field(default_factory=dict)
    prices: dict[str, dict[date, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dropped: int = 0
    _sorted: dict[str, list[date]] = field(default_factory=dict, repr=False)

    @property
    def dates(self) -> list[date]:
        return sorted(self.chains)

    @property
    def symbols(self) -> list[str]:
        return sorted(self.prices)

    def chain(self, d: date, symbol: str) -> list[OptionQuote]:
        return self.chains.get(d, {}).get(symbol, [])

    def quote(self, d: date, symbol: str, expiry: date, right: str, strike: float) -> OptionQuote | None:
        return self.index.get((d, symbol, expiry, right, float(strike)))

    def closes(self, symbol: str, upto: date) -> list[float]:
        """Daily underlying closes up to and including ``upto`` (oldest first). No look-ahead."""
        days = self._sorted.get(symbol)
        if days is None:
            days = self._sorted[symbol] = sorted(self.prices.get(symbol, {}))
        k = bisect.bisect_right(days, upto)
        return [self.prices[symbol][d] for d in days[:k]]


def from_frame(df: pd.DataFrame) -> ChainStore:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"option data is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise ValueError("option data is empty")
    df = df.copy()
    for col in ("date", "expiry"):
        try:
            try:
                df[col] = pd.to_datetime(df[col], format="ISO8601").dt.date
            except ValueError:   # not ISO: accept other unambiguous formats too, element by element
                df[col] = pd.to_datetime(df[col], format="mixed").dt.date
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot parse the {col} column: {exc}") from exc
    df["right"] = df["right"].astype(str).str.upper().str[0]
    if not df["right"].isin(["P", "C"]).all():
        raise ValueError("right must be P or C")
    store = ChainStore()
    has_oi = "open_interest" in df.columns
    if not has_oi:
        store.notes.append("no open_interest column: the open-interest liquidity filter was NOT applied")
    if "iv" not in df.columns and "delta" not in df.columns:
        raise ValueError("option data needs an iv and/or delta column")
    for row in df.itertuples(index=False):
        raw = OptionQuote(
            symbol=str(row.symbol).upper(), expiry=row.expiry, strike=float(row.strike), right=row.right,
            bid=float(row.bid), ask=float(row.ask), underlying_price=float(row.underlying_price),
            delta=None if "delta" not in df.columns or pd.isna(row.delta) else float(row.delta),
            iv=None if "iv" not in df.columns or pd.isna(row.iv) else float(row.iv),
            volume=int(row.volume) if "volume" in df.columns and not pd.isna(row.volume) else 0,
            open_interest=int(row.open_interest) if has_oi and not pd.isna(row.open_interest) else NO_OI)
        if raw.delta is None and not raw.iv:
            store.dropped += 1
            continue
        q = normalize_quote(raw, row.date)
        if q is None or q.expiry <= row.date:
            store.dropped += 1
            continue
        store.chains.setdefault(row.date, {}).setdefault(q.symbol, []).append(q)
        store.index[(row.date, q.symbol, q.expiry, q.right, q.strike)] = q
        store.prices.setdefault(q.symbol, {})[row.date] = q.underlying_price
    if not store.chains:
        raise ValueError("no usable option rows after validation")
    if store.dropped:
        store.notes.append(f"{store.dropped} unusable rows dropped (crossed/empty quotes, no delta/iv, or expired)")
    return store


def load_chains(path: str | Path) -> ChainStore:
    try:
        df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read option data {path}: {exc}") from exc
    return from_frame(df)


def load_earnings(path: str | Path) -> dict[str, list[date]]:
    """CSV with columns symbol,date: every historical earnings date you know. Used to enforce the earnings exclusion
    in the replay. Coverage must extend past the last trade expiry, or later dates are treated as unknown (blocked)."""
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = df["symbol"].astype(str).str.upper()
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read earnings file {path}: {exc}") from exc
    out: dict[str, list[date]] = {}
    for sym, d in zip(df["symbol"], df["date"], strict=True):
        out.setdefault(sym, []).append(d)
    return {k: sorted(set(v)) for k, v in out.items()}
