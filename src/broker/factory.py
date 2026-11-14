from __future__ import annotations

from src.broker.base import BrokerClient
from src.broker.mock_client import MockBroker
from src.config import Settings


def create_broker(settings: Settings) -> BrokerClient:
    if settings.broker == "mock":
        return MockBroker(equity=settings.account_equity)
    if settings.broker == "ib":
        from src.broker.ib_client import IBClient  # lazy: keeps ib libs optional for CI
        return IBClient.from_settings(settings)
    raise ValueError(f"unknown BROKER '{settings.broker}' (expected 'mock' or 'ib')")
