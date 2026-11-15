"""Stats, market-hours guard, stale-quote guard and the SQLite trade log."""
import sqlite3
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.data.freshness import assess_freshness
from src.data.stats import UnderlyingStats, atm_iv, historical_vol, skew_25d, sma, term_slope
from src.market_hours import is_market_open, is_trading_day, nyse_holidays, session_close
from src.models import OptionQuote, SpreadPosition, SpreadSignal, SpreadType, dte_bucket
from src.storage.trade_log import TradeLog

NY = ZoneInfo("America/New_York")
TODAY = date(2026, 1, 5)


def oq(dte, strike, right="C", iv=0.2, delta=None, spot=100.0, qt=None):
    return OptionQuote("SPY", TODAY + timedelta(days=dte), strike, right, 1.0, 1.1, spot, delta, iv, 0, 500, qt)


# --- stats -----------------------------------------------------------------------------------------------------
def test_sma_and_historical_vol():
    assert sma([1, 2, 3, 4], 2) == 3.5 and sma([1, 2], 5) is None
    assert historical_vol([100.0] * 40, 30) == 0.0
    zig = [100 * (1.01 if i % 2 else 1.0) for i in range(41)]
    hv = historical_vol(zig, 30)
    assert hv is not None and 0.14 < hv < 0.20                      # ~ 1% daily moves -> ~16% annualised
    assert historical_vol([100.0] * 10, 30) is None                 # not enough history
    assert historical_vol([100.0] * 30 + [0.0], 30) is None         # bad price -> unusable


def test_atm_iv_picks_expiry_near_35_dte_and_averages_near_the_money():
    chain = [oq(30, 100, iv=0.20), oq(30, 101, "P", iv=0.22), oq(45, 100, iv=0.50), oq(30, 130, iv=0.9)]
    assert atm_iv(chain, TODAY) == pytest.approx(0.21)             # 30 DTE is closer to 35; far strike ignored
    assert atm_iv([oq(30, 100, iv=None)], TODAY) is None
    assert atm_iv([], TODAY) is None


def test_term_slope_contango_flat_inverted_and_missing():
    front, back = 20, 48
    contango = [oq(front, 100, iv=0.20), oq(back, 100, iv=0.22)]
    inverted = [oq(front, 100, iv=0.26), oq(back, 100, iv=0.22)]
    flat = [oq(front, 100, iv=0.20), oq(back, 100, iv=0.20)]
    assert term_slope(contango, TODAY) == pytest.approx(0.10)
    assert term_slope(inverted, TODAY) < 0
    assert term_slope(flat, TODAY) == pytest.approx(0.0)
    assert term_slope([oq(front, 100, iv=0.2)], TODAY) is None                       # one expiry
    assert term_slope([oq(20, 100, iv=0.2), oq(25, 100, iv=0.3)], TODAY) is None     # expiries too close together


def test_skew_25d_put_minus_call():
    chain = [oq(35, 95, "P", iv=0.30, delta=-0.25), oq(35, 105, "C", iv=0.20, delta=0.25),
             oq(35, 90, "P", iv=0.40, delta=-0.10)]
    assert skew_25d(chain, TODAY) == pytest.approx(0.10)
    assert skew_25d(chain[:1], TODAY) is None                                          # no call to compare with
    assert skew_25d([], TODAY) is None


def test_underlying_stats_ratio():
    assert UnderlyingStats("X", 100, hv=0.2, iv=0.22).iv_hv_ratio == pytest.approx(1.1)
    assert UnderlyingStats("X", 100, hv=None, iv=0.22).iv_hv_ratio is None


# --- dte bucket ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dte,bucket", [(10, "<25"), (25, "25-29"), (30, "30-34"), (35, "35-45"), (45, "35-45"),
                                        (46, "46+"), (50, "46+"), (51, ">50")])
def test_dte_bucket(dte, bucket):
    assert dte_bucket(dte) == bucket


# --- market hours ----------------------------------------------------------------------------------------------
def ny(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


def test_regular_session_boundaries():
    assert is_market_open(ny(2026, 11, 18, 9, 30)) and is_market_open(ny(2026, 11, 18, 15, 59))
    assert not is_market_open(ny(2026, 11, 18, 9, 29)) and not is_market_open(ny(2026, 11, 18, 16, 0))
    assert not is_market_open(ny(2026, 11, 14, 11, 0))                       # Saturday
    assert is_market_open(datetime(2026, 11, 18, 15, 0, tzinfo=UTC))        # 10:00 EST expressed in UTC
    assert is_market_open(datetime(2026, 11, 18, 15, 0))                    # naive values are treated as UTC


def test_holidays_and_early_closes_2026():
    h = nyse_holidays(2026)
    assert {date(2026, 11, 26), date(2026, 12, 25), date(2026, 4, 3), date(2026, 1, 19), date(2026, 2, 16),
            date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7)} <= h   # Jul 4 is a Saturday
    assert not is_market_open(ny(2026, 11, 26, 11, 0))                      # Thanksgiving
    assert is_market_open(ny(2026, 11, 27, 12, 59)) and not is_market_open(ny(2026, 11, 27, 13, 0))   # half day
    assert session_close(date(2026, 12, 24)).hour == 13
    assert not is_trading_day(date(2026, 7, 3)) and is_trading_day(date(2026, 7, 2))


def test_saturday_new_year_is_not_observed_on_december_31():
    assert date(2028, 1, 1).weekday() == 5
    assert is_trading_day(date(2027, 12, 31))
    assert date(2023, 1, 2) in nyse_holidays(2023)                           # Sunday Jan 1 -> Monday


# --- stale-quote guard --------------------------------------------------------------------------------------------
NOW = datetime(2026, 11, 18, 15, 0, tzinfo=UTC)


def chain_aged(*ages):
    return [oq(30, 100 + i, qt=None if a is None else NOW - timedelta(seconds=a)) for i, a in enumerate(ages)]


def test_fresh_chain_passes():
    fr = assess_freshness(chain_aged(*[5] * 20), NOW, 120)
    assert fr.ok and fr.newest_age_sec == 5


def test_fully_stale_or_untimed_chain_fails_closed():
    assert not assess_freshness(chain_aged(*[3600] * 10), NOW, 120).ok            # frozen feed
    assert not assess_freshness(chain_aged(*[None] * 10), NOW, 120).ok            # no timestamps at all
    assert not assess_freshness([], NOW, 120).ok
    assert "stale" in assess_freshness(chain_aged(*[3600] * 10), NOW, 120).describe()


def test_small_illiquid_tail_is_tolerated_but_a_large_stale_share_is_not():
    assert assess_freshness(chain_aged(*([5] * 19 + [9999])), NOW, 120).ok        # 5% stale
    assert not assess_freshness(chain_aged(*([5] * 10 + [9999] * 10)), NOW, 120).ok


# --- trade log -------------------------------------------------------------------------------------------------
EXP = date(2026, 2, 16)                                                           # 40 DTE from Wed 2026-01-07
WED = datetime(2026, 1, 7, 15, 0, tzinfo=UTC)


def spos(contracts=1, debit=0.5, delta=0.18, entry_credit=1.0, expiry=EXP):
    return SpreadPosition(SpreadType.BULL_PUT, "SPY", expiry, 480, 475, contracts, entry_credit, debit, 500.0, delta)


def sig(delta=0.15):
    return SpreadSignal(SpreadType.BULL_PUT, "SPY", EXP, 40, 500, 480, 475, 5, 1.0, 100, 400,
                        0.25, delta, 0.85, 479, 0.04)


def test_entry_delta_is_captured_automatically_on_first_sight_only():
    log = TradeLog()
    assert log.observe_position(spos(delta=0.18), "HOLD", WED) == (0.18, "auto")
    assert log.observe_position(spos(delta=0.31), "HOLD", WED) == (0.18, "auto")  # later scans do not move it
    assert log.entry_deltas() == {spos().key: 0.18}
    assert log.initial_contracts(spos().key) == 1


def test_signal_delta_beats_first_scan_delta():
    log = TradeLog()
    log.record_signal(sig(0.15), WED)
    assert log.observe_position(spos(delta=0.30), "HOLD", WED) == (0.15, "signal")


def test_missing_delta_is_backfilled_later_and_manual_override_wins():
    log = TradeLog()
    assert log.observe_position(spos(delta=None), "HOLD", WED) == (None, None)
    assert log.observe_position(spos(delta=0.2), "HOLD", WED) == (0.2, "auto_late")
    assert log.override_entry_delta(spos().key, -0.12) and log.entry_deltas()[spos().key] == 0.12
    assert not log.override_entry_delta("nope", 0.1)


def test_position_close_is_debounced_and_pnl_is_net_of_modelled_commissions():
    log = TradeLog(commission_per_contract=0.65)
    p = spos(contracts=2, debit=0.5)
    log.observe_position(p, "TAKE_PROFIT", WED)
    assert log.close_missing(set(), WED + timedelta(days=5)) == []                # first miss: only counted
    log.observe_position(p, "TAKE_PROFIT", WED + timedelta(days=6))               # reappears -> counter resets
    assert log.close_missing(set(), WED + timedelta(days=7)) == []
    assert log.close_missing(set(), WED + timedelta(days=8)) == [p.key]           # second consecutive miss closes it
    (t,) = log.closed_trades()
    assert t["gross_pnl"] == 100.0 and t["commission"] == pytest.approx(5.2) and t["net_pnl"] == pytest.approx(94.8)
    assert t["exit_reason"] == "TAKE_PROFIT" and t["pnl_is_estimate"] == 1
    perf = log.performance()
    assert (perf.closed, perf.wins, perf.net_pnl, perf.win_rate) == (1, 1, 94.8, 1.0)


def test_performance_empty_and_losing_trade():
    log = TradeLog()
    assert log.performance().win_rate is None
    loser = spos(debit=3.0)
    log.observe_position(loser, "STOP_LOSS", WED)
    log.close_missing(set(), WED, min_missing=1)
    assert log.performance().net_pnl < 0 and log.performance().wins == 0


def test_breakdown_by_dte_bucket_and_weekday_and_ranged_history():
    log = TradeLog()
    a = spos(expiry=EXP)                                                            # opened Wed, 40 DTE -> "35-45"
    b = spos(expiry=date(2026, 2, 2), debit=3.0)                                    # opened Mon 01-05 -> 28 DTE "25-29"
    log.observe_position(a, "TAKE_PROFIT", WED)
    log.observe_position(b, "STOP_LOSS", datetime(2026, 1, 5, 15, 0, tzinfo=UTC))
    log.close_missing(set(), datetime(2026, 1, 20, 15, 0, tzinfo=UTC), min_missing=1)
    by_bucket = {r["group"]: r for r in log.breakdown("dte_bucket")}
    assert by_bucket["35-45"]["wins"] == 1 and by_bucket["25-29"]["wins"] == 0
    assert {r["group"] for r in log.breakdown("weekday")} == {0, 2}
    with pytest.raises(ValueError):
        log.breakdown("nonsense")
    assert len(log.closed_trades(start="2026-01-20", end="2026-01-20")) == 2
    assert log.closed_trades(start="2026-01-21") == [] and log.closed_trades(end="2026-01-19") == []
    assert len(log.closed_trades(limit=1)) == 1


def test_signal_dedupe_and_bucket_columns():
    log = TradeLog()
    assert log.record_signal(sig(), WED) and not log.record_signal(sig(), WED)       # once per key per day
    assert log.record_signal(sig(), WED + timedelta(days=1))
    row = log._db.execute("SELECT dte, dte_bucket, weekday FROM signals ORDER BY id").fetchone()
    assert (row["dte"], row["dte_bucket"], row["weekday"]) == (40, "35-45", 2) and log.signal_count() == 2


def test_snapshots_upsert_per_day_and_range():
    log = TradeLog()
    log.record_snapshot(date(2026, 1, 5), 10, -5, 2)
    log.record_snapshot(date(2026, 1, 5), 30, 5, 3)                                  # same day: latest scan wins
    log.record_snapshot(date(2026, 1, 6), 1, 1, 1)
    snaps = log.snapshots()
    assert len(snaps) == 2 and snaps[0]["unrealized_bull_put"] == 30 and snaps[0]["open_positions"] == 3
    assert [s["day"] for s in log.snapshots(start="2026-01-06")] == ["2026-01-06"]
    assert log.snapshots(end="2026-01-04") == []


def test_log_is_durable_across_connections(tmp_path):
    path = tmp_path / "sub" / "trades.db"
    log = TradeLog(path)
    log.observe_position(spos(), "HOLD", WED)
    log.close()
    again = TradeLog(path)
    assert again.entry_deltas() == {spos().key: 0.18}
    again.close()
    assert sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0] == "wal"
