"""Deterministic in-memory broker for local development and CI. No network, no IB needed.

Option prices come from Black-Scholes with a mild put skew so chains look realistic."""
from __future__ import annotations

import math
from datetime import date, timedelta

from src.broker.base import BrokerClient, BrokerError, expiries_in_window
from src.data.greeks import bs_delta, bs_price
from src.models import OptionQuote, PositionLeg

DEFAULT_PRICES = {"SPY": 500.0, "QQQ": 430.0, "IWM": 210.0, "AAPL": 190.0}


class MockBroker(BrokerClient):
    def __init__(
        self,
        prices: dict[str, float] | None = None,
        positions: list[PositionLeg] | None = None,
        equity: float = 50_000.0,
        today: date | None = None,
        base_iv: float = 0.22,
        fail_connect: bool = False,
    ) -> None:
        self.prices = dict(prices or DEFAULT_PRICES)
        self.positions = list(positions or [])
        self.equity = equity
        self.today = today or date.today()
        self.base_iv = base_iv
        self.fail_connect = fail_connect
        self._connected = False
        self.calls: list[str] = []  # lets tests assert on interactions

    async def connect(self) -> None:
        self.calls.append("connect")
        if self.fail_connect:
            raise BrokerError("mock connection failure")
        self._connected = True

    async def disconnect(self) -> None:
        self.calls.append("disconnect")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _require(self) -> None:
        if not self._connected:
            raise BrokerError("not connected")

    async def get_underlying_price(self, symbol: str) -> float:
        self._require()
        if symbol not in self.prices:
            raise BrokerError(f"unknown symbol {symbol}")
        return self.prices[symbol]

    def _weekly_expiries(self, count: int = 12) -> list[date]:
        days_to_friday = (4 - self.today.weekday()) % 7
        first = self.today + timedelta(days=days_to_friday or 7)
        return [first + timedelta(weeks=i) for i in range(count)]

    async def get_option_chain(
        self, symbol: str, min_dte: int = 20, max_dte: int = 60, strike_band_pct: float = 0.15
    ) -> list[OptionQuote]:
        self._require()
        spot = await self.get_underlying_price(symbol)
        step = 1.0 if spot < 250 else 5.0
        lo = math.floor(spot * (1 - strike_band_pct) / step) * step
        hi = math.ceil(spot * (1 + strike_band_pct) / step) * step
        strikes = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
        quotes: list[OptionQuote] = []
        for expiry in expiries_in_window(self._weekly_expiries(), min_dte, max_dte, self.today):
            t = (expiry - self.today).days / 365.0
            for k in strikes:
                for right in ("P", "C"):
                    skew = 0.15 * max(0.0, (spot - k) / spot) if right == "P" else 0.0
                    iv = self.base_iv + skew
                    mid = max(bs_price(spot, k, t, 0.045, iv, right), 0.01)
                    half = max(0.01, mid * 0.01)
                    quotes.append(
                        OptionQuote(
                            symbol=symbol, expiry=expiry, strike=k, right=right,
                            bid=round(max(mid - half, 0.0), 2), ask=round(mid + half, 2),
                            underlying_price=spot,
                            delta=round(bs_delta(spot, k, t, 0.045, iv, right), 4),
                            iv=round(iv, 4), volume=500, open_interest=2000,
                        )
                    )
        return quotes

    async def get_option_positions(self) -> list[PositionLeg]:
        self._require()
        return list(self.positions)

    async def get_account_equity(self) -> float | None:
        self._require()
        return self.equity
