"""Edge / friction filters: IV-HV, trend, term structure, skew, credit floor, spread width, weekday, DTE bucket."""
import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest

from src.data.market_data import MarketDataService
from src.data.stats import UnderlyingStats
from src.models import SpreadType
from src.strategies.credit_spreads import (
    ScannerConfig,
    in_preferred_dte,
    scan_all,
    scan_bear_call,
    scan_bull_put,
    schedule_gate,
    underlying_gate,
)

GOOD = UnderlyingStats("SPY", 500.0, hv=0.15, sma50=480.0, iv=0.20, term_slope=0.03, skew=0.04)  # Monday TODAY


def chain(broker, lo=25, hi=50):
    async def go():
        async with broker:
            return await MarketDataService(broker, lo, hi).get_chain("SPY", today=broker.today)
    return asyncio.run(go())


def test_filters_are_off_by_default_and_fail_closed_when_enabled_without_stats(broker, today):
    q = chain(broker)
    assert scan_all(q, ScannerConfig(), today)                                   # permissive base config
    for cfg in (ScannerConfig(min_iv_hv_ratio=1.1), ScannerConfig(require_trend_alignment=True),
                ScannerConfig(min_term_slope=0.01), ScannerConfig(max_put_skew=0.1)):
        assert scan_bull_put(q, cfg, today, None) == []                          # missing stats -> nothing
    assert scan_bull_put(q, ScannerConfig(), today, None)


def test_iv_hv_gate(broker, today):
    q, cfg = chain(broker), ScannerConfig(min_iv_hv_ratio=1.1)
    assert scan_bull_put(q, cfg, today, GOOD)                                    # 0.20 / 0.15 = 1.33
    cheap = replace(GOOD, iv=0.15)                                               # ratio 1.0 -> options not overpriced
    assert scan_bull_put(q, cfg, today, cheap) == []
    assert "IV/HV 1.00" in underlying_gate(cheap, SpreadType.BULL_PUT, cfg)[0]
    assert underlying_gate(replace(GOOD, hv=None), SpreadType.BULL_PUT, cfg) == ["IV/HV unavailable"]


def test_trend_alignment_bull_put_above_sma_bear_call_below(broker, today):
    q, cfg = chain(broker), ScannerConfig(require_trend_alignment=True)
    up, down = replace(GOOD, price=500, sma50=480), replace(GOOD, price=500, sma50=520)
    assert scan_bull_put(q, cfg, today, up) and scan_bear_call(q, cfg, today, up) == []
    assert scan_bear_call(q, cfg, today, down) and scan_bull_put(q, cfg, today, down) == []
    at = replace(GOOD, price=500, sma50=500)
    assert scan_bull_put(q, cfg, today, at) == [] and scan_bear_call(q, cfg, today, at) == []   # on the line: neither
    assert underlying_gate(replace(GOOD, sma50=None), SpreadType.BULL_PUT, cfg) == ["50-day SMA unavailable"]


def test_term_structure_blocks_flat_and_inverted_curves(broker, today):
    q, cfg = chain(broker), ScannerConfig(min_term_slope=0.01)
    assert scan_all(q, cfg, today, GOOD)
    flat, inverted = replace(GOOD, term_slope=0.0), replace(GOOD, term_slope=-0.05)
    assert scan_all(q, cfg, today, flat) == [] and scan_all(q, cfg, today, inverted) == []
    assert "flat" in underlying_gate(flat, SpreadType.BEAR_CALL, cfg)[0]
    assert "inverted" in underlying_gate(inverted, SpreadType.BEAR_CALL, cfg)[0]
    assert underlying_gate(replace(GOOD, term_slope=None), SpreadType.BEAR_CALL, cfg) == ["term structure unavailable"]
    assert scan_all(q, cfg, today, replace(GOOD, term_slope=0.01))               # boundary is inclusive


def test_put_skew_only_blocks_bull_puts(broker, today):
    q, cfg = chain(broker), ScannerConfig(max_put_skew=0.10)
    steep = replace(GOOD, skew=0.15)
    assert scan_bull_put(q, cfg, today, steep) == []                 # market is bidding for crash protection
    assert scan_bear_call(q, cfg, today, steep)                      # calls are not exposed to that
    assert scan_bull_put(q, cfg, today, replace(GOOD, skew=0.10))    # boundary inclusive
    assert underlying_gate(replace(GOOD, skew=None), SpreadType.BULL_PUT, cfg) == ["put skew unavailable"]
    assert underlying_gate(replace(GOOD, skew=None), SpreadType.BEAR_CALL, cfg) == []


def test_credit_floor_and_absolute_spread_cap(broker, today):
    q = chain(broker)
    base = scan_all(q, ScannerConfig(top_n_per_symbol=500), today)
    credits = sorted(s.credit for s in base)
    cut = credits[len(credits) // 2]                                             # a floor that removes the cheaper half
    floor = scan_all(q, ScannerConfig(min_credit=cut, top_n_per_symbol=500), today)
    assert floor and len(floor) < len(base) and all(s.credit >= cut for s in floor)
    assert scan_all(q, ScannerConfig(min_credit=0.60), today)                    # the preset's 0.60 floor is reachable
    tight = scan_all(q, ScannerConfig(max_bid_ask_abs=0.02, top_n_per_symbol=500), today)
    assert len(tight) < len(base)                                                # expensive strikes have wider quotes
    assert scan_all(q, ScannerConfig(max_bid_ask_abs=0.0001), today) == []


def test_weekday_filter(broker, today):
    assert today.weekday() == 0                                                  # the fixture date is a Monday
    q = chain(broker)
    assert scan_all(q, ScannerConfig(allowed_weekdays=(0,)), today)
    blocked = ScannerConfig(allowed_weekdays=(1, 2, 3))
    assert scan_all(q, blocked, today) == []
    assert scan_all(q, blocked, today + timedelta(days=1))                       # Tuesday is fine (DTE shifts by one)
    assert schedule_gate(blocked, today) == ["entries allowed on TWT only (today is M)"]
    assert schedule_gate(ScannerConfig(), today) == []


def test_preferred_dte_bucket_is_ranked_first_without_dropping_others(broker, today):
    q = chain(broker)
    plain = scan_all(q, ScannerConfig(top_n_per_symbol=500), today)
    pref = scan_all(q, ScannerConfig(top_n_per_symbol=500, preferred_dte=(35, 45)), today)
    assert {s.key for s in pref} == {s.key for s in plain}                       # a preference, not a filter
    flags = [in_preferred_dte(s.dte, ScannerConfig(preferred_dte=(35, 45))) for s in pref]
    assert any(flags) and not all(flags) and flags == sorted(flags, reverse=True)  # all preferred rows come first
    assert not in_preferred_dte(40, ScannerConfig())


@pytest.mark.parametrize("hi,expect_dtes", [(50, {39, 46}), (45, {39})])
def test_data_window_limits_signals_even_with_wider_chain(broker, today, hi, expect_dtes):
    wide = chain(broker, 7, 70)                                                  # chain wider than the trade window
    sigs = scan_all(wide, ScannerConfig(min_dte=35, max_dte=hi, top_n_per_symbol=500), today)
    assert {s.dte for s in sigs} == expect_dtes


def test_disciplined_preset_bundles_all_rules_and_accepts_overrides():
    d = ScannerConfig.disciplined()
    assert (d.min_credit, d.min_roc, d.pricing, d.max_bid_ask_pct) == (0.60, 0.20, "natural", 0.05)
    assert d.min_iv_hv_ratio == 1.1 and d.require_trend_alignment and d.min_term_slope == 0.01
    assert d.max_put_skew == 0.10 and d.preferred_dte == (35, 45) and (d.min_dte, d.max_dte) == (30, 50)
    assert d.allowed_weekdays is None                                           # unvalidated: opt in explicitly
    assert ScannerConfig.disciplined(max_bid_ask_abs=0.02).max_bid_ask_abs == 0.02
