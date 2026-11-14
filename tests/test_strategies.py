import asyncio
from dataclasses import replace

from src.data.market_data import MarketDataService
from src.models import SpreadType
from src.strategies.credit_spreads import ScannerConfig, scan_all, scan_bear_call, scan_bull_put


def chain(broker):
    async def go():
        async with broker:
            return await MarketDataService(broker, 25, 50).get_chain("SPY", today=broker.today)
    return asyncio.run(go())


def test_bull_put_signals_obey_filters(broker, today):
    cfg = ScannerConfig()
    sigs = scan_bull_put(chain(broker), cfg, today)
    assert sigs
    for s in sigs:
        assert s.spread_type is SpreadType.BULL_PUT
        assert s.short_strike < s.underlying_price and s.long_strike == s.short_strike - s.width
        assert cfg.min_short_delta <= s.short_delta <= cfg.max_short_delta
        assert cfg.min_dte <= s.dte <= cfg.max_dte
        assert s.roc >= cfg.min_roc and s.max_loss <= cfg.max_risk_per_spread
        assert s.breakeven == round(s.short_strike - s.credit, 2)
    assert [s.roc for s in sigs] == sorted((s.roc for s in sigs), reverse=True)


def test_bear_call_signals(broker, today):
    sigs = scan_bear_call(chain(broker), ScannerConfig(), today)
    assert sigs
    for s in sigs:
        assert s.short_strike > s.underlying_price and s.long_strike == s.short_strike + s.width
        assert s.breakeven == round(s.short_strike + s.credit, 2)


def test_spread_math(broker, today):
    s = scan_bull_put(chain(broker), ScannerConfig(), today)[0]
    assert s.max_profit == round(s.credit * 100, 2)
    assert s.max_loss == round((s.width - s.credit) * 100, 2)
    assert abs(s.roc - s.credit / (s.width - s.credit)) < 1e-3
    assert abs(s.pop - (1 - s.short_delta)) < 1e-9


def test_tight_filters_yield_nothing(broker, today):
    q = chain(broker)
    assert scan_all(q, ScannerConfig(min_roc=5.0), today) == []
    assert scan_all(q, ScannerConfig(max_risk_per_spread=1.0), today) == []
    assert scan_all(q, ScannerConfig(min_open_interest=10**9), today) == []
    assert scan_all(q, ScannerConfig(min_dte=1, max_dte=2), today) == []


def test_natural_pricing_is_more_conservative(broker, today):
    q = chain(broker)
    cfg_mid = ScannerConfig(min_roc=0.0, min_credit=0.0, top_n_per_symbol=10_000)
    cfg_nat = replace(cfg_mid, pricing="natural")
    mid = {s.key: s.credit for s in scan_bull_put(q, cfg_mid, today)}
    nat = {s.key: s.credit for s in scan_bull_put(q, cfg_nat, today)}
    common = mid.keys() & nat.keys()
    assert common and all(nat[k] <= mid[k] for k in common)


def test_top_n_and_scan_all(broker, today):
    q = chain(broker)
    cfg = ScannerConfig(top_n_per_symbol=3)
    assert len(scan_bull_put(q, cfg, today)) <= 3
    both = scan_all(q, cfg, today)
    assert {s.spread_type for s in both} <= set(SpreadType)
