"""Market-data service: pulls chains from a broker and normalises them (fills missing
Greeks, drops unusable quotes) into models / DataFrames."""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date

import pandas as pd

from src.broker.base import BrokerClient
from src.data.greeks import bs_delta
from src.models import OptionQuote

log = logging.getLogger(__name__)
RISK_FREE_RATE = 0.045
FALLBACK_IV = 0.25


def normalize_quote(q: OptionQuote, today: date | None = None) -> OptionQuote | None:
    """Drop crossed/empty quotes; compute delta with Black-Scholes when the broker gave none."""
    if q.bid < 0 or q.ask <= 0 or q.ask < q.bid:
        return None
    if q.delta is not None:
        return q
    t = max(q.dte(today), 0) / 365.0
    delta = bs_delta(q.underlying_price, q.strike, t, RISK_FREE_RATE, q.iv or FALLBACK_IV, q.right)
    return replace(q, delta=delta)


def quotes_to_frame(quotes: list[OptionQuote]) -> pd.DataFrame:
    cols = ["symbol", "expiry", "strike", "right", "bid", "ask", "mid", "delta", "iv",
            "volume", "open_interest", "underlying_price"]
    if not quotes:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(
        [{**{c: getattr(q, c) for c in cols if c != "mid"}, "mid": q.mid} for q in quotes]
    )[cols]


class MarketDataService:
    def __init__(self, broker: BrokerClient, min_dte: int = 20, max_dte: int = 60,
                 strike_band_pct: float = 0.15) -> None:
        self.broker = broker
        self.min_dte, self.max_dte, self.strike_band_pct = min_dte, max_dte, strike_band_pct

    async def get_chain(self, symbol: str, today: date | None = None) -> list[OptionQuote]:
        raw = await self.broker.get_option_chain(
            symbol, self.min_dte, self.max_dte, self.strike_band_pct)
        chain = [n for q in raw if (n := normalize_quote(q, today))]
        log.debug("%s: %d/%d quotes usable", symbol, len(chain), len(raw))
        return chain

    async def get_chain_frame(self, symbol: str) -> pd.DataFrame:
        return quotes_to_frame(await self.get_chain(symbol))
