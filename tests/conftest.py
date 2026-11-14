from datetime import date

import pytest

from src.broker.mock_client import MockBroker

TODAY = date(2026, 1, 5)  # a Monday; fixed so expiries/DTE are deterministic


@pytest.fixture
def today():
    return TODAY


@pytest.fixture
def broker():
    return MockBroker(today=TODAY)
