import asyncio

import pytest

from src.broker.base import BrokerError
from src.broker.factory import create_broker
from src.broker.mock_client import MockBroker
from src.config import IB_PORT_LIVE, IB_PORT_PAPER, Settings


def run(coro):
    return asyncio.run(coro)


def test_requires_connection(broker):
    with pytest.raises(BrokerError):
        run(broker.get_underlying_price("SPY"))


def test_connect_and_chain(broker):
    async def go():
        async with broker:
            assert broker.is_connected()
            return await broker.get_option_chain("SPY", 25, 50)
    chain = run(go())
    assert chain and {q.right for q in chain} == {"P", "C"}
    assert all(25 <= q.dte(broker.today) <= 50 for q in chain)
    assert not broker.is_connected()


def test_failed_connect():
    with pytest.raises(BrokerError):
        run(MockBroker(fail_connect=True).connect())


def test_unknown_symbol(broker):
    async def go():
        async with broker:
            await broker.get_underlying_price("ZZZZ")
    with pytest.raises(BrokerError):
        run(go())


def test_factory_and_ports(monkeypatch):
    assert isinstance(create_broker(Settings(broker="mock")), MockBroker)
    with pytest.raises(ValueError):
        create_broker(Settings(broker="nope"))
    monkeypatch.delenv("IB_PORT", raising=False)
    monkeypatch.setenv("TRADING_MODE", "live")
    assert Settings.from_env().ib_port == IB_PORT_LIVE
    monkeypatch.setenv("TRADING_MODE", "paper")
    assert Settings.from_env().ib_port == IB_PORT_PAPER
    monkeypatch.setenv("TRADING_MODE", "yolo")
    with pytest.raises(ValueError):
        Settings.from_env()
