import asyncio
from datetime import date

import pytest

from src.data.greeks import bs_delta, bs_price
from src.data.market_data import MarketDataService, normalize_quote, quotes_to_frame
from src.models import OptionQuote


def test_put_call_parity():
    c = bs_price(100, 100, 0.5, 0.03, 0.2, "C")
    p = bs_price(100, 100, 0.5, 0.03, 0.2, "P")
    assert c - p == pytest.approx(100 - 100 * 2.718281828 ** (-0.03 * 0.5), abs=1e-3)


def test_delta_signs_and_bounds():
    assert 0.5 < bs_delta(100, 100, 0.5, 0.03, 0.2, "C") < 0.65
    assert -0.5 < bs_delta(100, 100, 0.5, 0.03, 0.2, "P") < -0.35
    assert bs_delta(100, 50, 0.5, 0.03, 0.2, "P") > -0.001      # deep OTM put
    assert bs_delta(100, 100, 0, 0.03, 0.2, "C") == 0.0          # expiry, ATM -> not ITM


def _q(**kw):
    base = dict(symbol="SPY", expiry=date(2026, 2, 6), strike=480, right="P", bid=1.0, ask=1.1,
                underlying_price=500, delta=None, iv=0.2)
    return OptionQuote(**{**base, **kw})


def test_normalize_fills_delta_and_drops_bad_quotes():
    q = normalize_quote(_q(), today=date(2026, 1, 5))
    assert q is not None and -0.3 < q.delta < 0
    assert normalize_quote(_q(bid=2.0, ask=1.0)) is None   # crossed
    assert normalize_quote(_q(ask=0.0, bid=0.0)) is None   # empty
    assert normalize_quote(_q(delta=-0.2)).delta == -0.2   # broker value kept


def test_chain_service_and_frame(broker):
    async def go():
        async with broker:
            return await MarketDataService(broker, 25, 50).get_chain("QQQ", today=broker.today)
    chain = asyncio.run(go())
    df = quotes_to_frame(chain)
    assert len(df) == len(chain) > 0
    assert {"mid", "delta", "open_interest"} <= set(df.columns)
    assert quotes_to_frame([]).empty
