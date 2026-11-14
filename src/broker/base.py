"""Broker abstraction. Strategy/risk/dashboard code depends only on this interface,
so the mock, paper and live brokers are interchangeable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from src.models import OptionQuote, PositionLeg


class BrokerError(RuntimeError):
    """Raised for connection or data-request failures."""


class BrokerClient(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    async def get_underlying_price(self, symbol: str) -> float: ...

    @abstractmethod
    async def get_option_chain(
        self, symbol: str, min_dte: int = 20, max_dte: int = 60, strike_band_pct: float = 0.15
    ) -> list[OptionQuote]:
        """Return quotes (puts and calls) for expiries within [min_dte, max_dte] and
        strikes within +/- ``strike_band_pct`` of spot."""

    @abstractmethod
    async def get_option_positions(self) -> list[PositionLeg]: ...

    async def get_account_equity(self) -> float | None:
        """Net liquidation value, if the broker can provide it."""
        return None

    async def __aenter__(self) -> BrokerClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()


def expiries_in_window(expiries: list[date], min_dte: int, max_dte: int, today: date | None = None) -> list[date]:
    today = today or date.today()
    return sorted(e for e in expiries if min_dte <= (e - today).days <= max_dte)
