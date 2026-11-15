"""The adopted strategy parameters: lower short delta + relaxed floors, 65% profit target, SPY/QQQ-only universe."""
import asyncio
from datetime import date, timedelta

import pytest

from src.alerts.notifiers import AlertDispatcher
from src.backtest.__main__ import main as backtest_main
from src.backtest.data import from_frame
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.synthetic import SymbolProfile, generate_universe
from src.broker.mock_client import MockBroker
from src.config import CORE_UNIVERSE, Settings
from src.models import PositionLeg, SpreadPosition, SpreadSignal, SpreadType
from src.risk.earnings import EarningsInfo, EarningsStatus
from src.risk.manager import DISCIPLINED_PROFIT_TARGET, ExitAction, RiskLimits, RiskManager
from src.service import ScanService
from src.strategies.credit_spreads import ScannerConfig, scan_all

TODAY = date.today()
FRIDAY = TODAY + timedelta(days=(4 - TODAY.weekday()) % 7 or 7) + timedelta(weeks=4)   # an expiry the mock lists


# --- 1. lower short delta + relaxed floors ---
NO_STATS_GATES = dict(min_iv_hv_ratio=None, require_trend_alignment=False, min_term_slope=None, max_put_skew=None)


def synthetic_day(vrp=0.15, days=70, seed=3, symbol="SPY"):
    """A realistic chain (contango, put skew, $5 strikes) on the last of ``days`` simulated days."""
    df, _ = generate_universe({symbol: SymbolProfile(real_vol=0.16, vrp=vrp, half_spread_pct=0.003)}, days, seed)
    store = from_frame(df)
    d = store.dates[-1]
    return store.chain(d, symbol), d


def test_preset_targets_the_lower_delta_band_with_relaxed_floors():
    d = ScannerConfig.disciplined()
    assert (d.min_short_delta, d.max_short_delta) == (0.12, 0.15)
    assert (d.min_credit, d.min_roc) == (0.45, 0.12)
    assert (d.min_dte, d.max_dte, d.preferred_dte) == (30, 50, (35, 45))           # unchanged
    assert d.pricing == "natural" and d.min_iv_hv_ratio == 1.1          # friction/edge rules unchanged
    assert d.require_trend_alignment
    assert d.allowed_weekdays is None
    legacy = ScannerConfig.disciplined(min_short_delta=0.10, max_short_delta=0.25, min_credit=0.60, min_roc=0.20)
    got = (legacy.min_short_delta, legacy.max_short_delta, legacy.min_credit, legacy.min_roc)
    assert got == (0.10, 0.25, 0.60, 0.20)


def test_the_band_and_floors_return_trades_but_the_old_floors_return_none():
    chain, d = synthetic_day()
    new = scan_all(chain, ScannerConfig.disciplined(**NO_STATS_GATES, top_n_per_symbol=50), d)
    assert new, "the relaxed floors must find trades in the 0.12-0.15 band"
    for s in new:
        assert 0.12 <= s.short_delta <= 0.15 and s.credit >= 0.45 and s.roc >= 0.12 and 30 <= s.dte <= 50
    old_floors = ScannerConfig.disciplined(**NO_STATS_GATES, min_credit=0.60, min_roc=0.20)
    assert scan_all(chain, old_floors, d) == []     # the simulation finding: the old 0.60 / 20% floors choke this band


def test_lower_delta_signals_collect_less_premium_than_the_old_band():
    chain, d = synthetic_day()
    low = scan_all(chain, ScannerConfig.disciplined(**NO_STATS_GATES, top_n_per_symbol=50), d)
    high = scan_all(chain, ScannerConfig.disciplined(**NO_STATS_GATES, min_short_delta=0.20, max_short_delta=0.25,
                                                     min_credit=0.45, min_roc=0.12, top_n_per_symbol=50), d)
    assert low and high
    avg = lambda xs: sum(x.credit / x.width for x in xs) / len(xs)                  # noqa: E731
    assert avg(low) < avg(high)                                    # smaller credit per unit of width
    mean_pop = lambda xs: sum(x.pop for x in xs) / len(xs)         # noqa: E731
    assert mean_pop(low) > mean_pop(high)                          # higher modelled probability of profit


def test_full_disciplined_filters_still_find_trades_end_to_end_in_a_healthy_premium_world():
    df, _ = generate_universe({"SPY": SymbolProfile(real_vol=0.16, vrp=0.20, half_spread_pct=0.003)}, 400, 5)
    r = run_backtest(from_frame(df), BacktestConfig(equity=500_000.0, max_new_per_day=2))   # defaults = the new rules
    assert len(r.trades) >= 5 and r.config.limits.profit_target_pct == 0.65
    assert all(0.12 <= t.entry_short_delta <= 0.15 for t in r.trades)


def five_wide_signal(max_loss=443.0, credit=0.57):
    return SpreadSignal(SpreadType.BULL_PUT, "SPY", FRIDAY, 35, 500, 485, 480, 5.0, credit, credit * 100, max_loss,
                        0.129, 0.14, 0.86, 484.4, 0.03)


def test_the_configured_risk_cap_admits_a_five_wide_spread_on_a_20k_account():
    """The relaxed floors pick 5-wide spreads (~$443 = 2.2% of $20k). The old 2% cap ($400) rejected them all."""
    sig = five_wide_signal()
    configured = RiskManager(Settings().risk_limits())
    assert configured.limits.max_risk_per_trade_pct == 0.10
    assert configured.check_signal(sig, [], 20_000).approved                     # $443 <= $2,000
    old = RiskManager(Settings(max_risk_per_trade_pct=0.02).risk_limits())
    refused = old.check_signal(sig, [], 20_000)
    assert not refused.approved and "trade risk $443" in refused.reasons[0]    # the old behaviour, still reachable
    # 2.5% is the tightest round setting that still lets these spreads through on $20k
    assert RiskManager(Settings(max_risk_per_trade_pct=0.025).risk_limits()).check_signal(sig, [], 20_000).approved
    assert not RiskManager(Settings(max_risk_per_trade_pct=0.021).risk_limits()).check_signal(sig, [], 20_000).approved


def test_the_cap_defaults_to_ten_percent_in_settings_env_and_the_backtest_while_the_library_default_stays_two():
    assert RiskLimits().max_risk_per_trade_pct == 0.02                            # neutral library default
    assert RiskLimits.disciplined().max_risk_per_trade_pct == 0.10
    assert RiskLimits.disciplined(max_risk_per_trade_pct=0.03).max_risk_per_trade_pct == 0.03
    assert Settings().max_risk_per_trade_pct == 0.10 == Settings.from_env().max_risk_per_trade_pct
    assert BacktestConfig().limits.max_risk_per_trade_pct == 0.10


def test_the_ten_percent_ceiling_is_accepted_and_anything_above_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("MAX_RISK_PER_TRADE_PCT", "0.10")
    assert Settings.from_env().risk_limits().max_risk_per_trade_pct == 0.10    # exactly the ceiling: no startup error
    for bad in ("0.1001", "0.11", "0.5", "0", "-0.02"):
        monkeypatch.setenv("MAX_RISK_PER_TRADE_PCT", bad)
        with pytest.raises(ValueError, match=r"MAX_RISK_PER_TRADE_PCT must be in \(0, 0.1\]"):
            Settings.from_env()
    monkeypatch.delenv("MAX_RISK_PER_TRADE_PCT")
    assert Settings.from_env().max_risk_per_trade_pct == 0.10


def test_the_service_approves_lower_delta_spreads_on_a_20k_account():
    """Real service path: mock chain, $20k equity, the 0.12-0.15 delta band and the relaxed floors."""
    cfg = ScannerConfig.disciplined(**NO_STATS_GATES, top_n_per_symbol=20)
    settings = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, account_equity=20_000)
    svc = ScanService(settings, MockBroker(today=TODAY, equity=20_000), AlertDispatcher([]), scanner_cfg=cfg)
    asyncio.run(svc.run_once())
    assert svc.state.equity == 20_000 and svc.state.signals and svc.state.rejected == 0
    assert all(0.12 <= s["short_delta"] <= 0.15 and s["credit"] >= 0.45 for s in svc.state.signals)
    assert any(s["max_loss"] > 400 for s in svc.state.signals)             # spreads the old 2% cap would have rejected
    old = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, account_equity=20_000,
                   max_risk_per_trade_pct=0.02)
    svc_old = ScanService(old, MockBroker(today=TODAY, equity=20_000), AlertDispatcher([]), scanner_cfg=cfg)
    asyncio.run(svc_old.run_once())
    assert svc_old.state.signals == [] and svc_old.state.rejected > 0             # the 2% cap blocks them all


def test_the_loose_per_trade_cap_does_not_remove_the_other_risk_limits():
    rm = RiskManager(Settings().risk_limits())
    # the scanner still refuses anything above $1,000 of risk per lot, whatever the per-trade cap says
    chain, d = synthetic_day(days=70)
    wide = scan_all(chain, ScannerConfig.disciplined(**NO_STATS_GATES, widths=(5.0, 20.0), top_n_per_symbol=50), d)
    assert wide and all(s.max_loss <= 1_000 for s in wide)
    # total open risk is still capped at 20% of equity: nine $443 spreads fit in $4,000, the tenth does not
    book = [SpreadPosition(SpreadType.BULL_PUT, "SPY", FRIDAY, 480 - i, 475 - i, 1, 0.57, 0.5, 500.0)
            for i in range(9)]
    assert all(p.max_loss == pytest.approx(443) for p in book)
    rm.limits.max_open_positions = 20
    rm.limits.max_positions_per_symbol = 20
    assert rm.check_signal(five_wide_signal(), book[:8], 20_000).approved
    over = rm.check_signal(five_wide_signal(), book, 20_000)
    assert not over.approved and "total risk" in over.reasons[0]


# --- 2. 65% profit target ---
def pos(credit, debit):
    return SpreadPosition(SpreadType.BULL_PUT, "SPY", date(2026, 3, 20), 480, 475, 1, credit, debit)


def test_configured_target_is_65_percent_while_the_library_default_stays_50():
    assert DISCIPLINED_PROFIT_TARGET == 0.65 == RiskLimits.disciplined().profit_target_pct
    assert RiskLimits().profit_target_pct == 0.50
    assert RiskLimits.disciplined(profit_target_pct=0.6).profit_target_pct == 0.6
    assert Settings().profit_target_pct == 0.65 and Settings().risk_limits().profit_target_pct == 0.65
    assert BacktestConfig().limits.profit_target_pct == 0.65


def test_take_profit_fires_at_65_percent_not_before():
    rm = RiskManager(RiskLimits.disciplined())
    today = date(2026, 1, 5)
    assert rm.evaluate_exit(pos(2.0, 0.7), today) is ExitAction.TAKE_PROFIT        # 65% exactly (credit 2.0, debit 0.7)
    assert rm.evaluate_exit(pos(2.0, 0.75), today) is ExitAction.HOLD               # 62.5%: no longer an exit
    assert rm.evaluate_exit(pos(2.0, 1.0), today) is ExitAction.HOLD                # the old 50% target does not fire
    assert RiskManager(RiskLimits()).evaluate_exit(pos(2.0, 1.0), today) is ExitAction.TAKE_PROFIT   # library default
    assert rm.evaluate_exit(pos(2.0, 6.0), today) is ExitAction.STOP_LOSS            # the 2x stop is unchanged


def test_target_is_configurable_and_validated(monkeypatch):
    monkeypatch.setenv("PROFIT_TARGET_PCT", "0.6")
    monkeypatch.setenv("MAX_RISK_PER_TRADE_PCT", "0.025")
    s = Settings.from_env()
    assert s.profit_target_pct == 0.6 and s.risk_limits().max_risk_per_trade_pct == 0.025
    for bad in ("0", "1.5", "-0.2"):
        monkeypatch.setenv("PROFIT_TARGET_PCT", bad)
        with pytest.raises(ValueError, match="PROFIT_TARGET_PCT"):
            Settings.from_env()
    monkeypatch.setenv("PROFIT_TARGET_PCT", "0.65")
    monkeypatch.setenv("MAX_RISK_PER_TRADE_PCT", "0.5")
    with pytest.raises(ValueError, match="MAX_RISK_PER_TRADE_PCT"):
        Settings.from_env()
    monkeypatch.setenv("MAX_RISK_PER_TRADE_PCT", "0.02")
    monkeypatch.setenv("SCALE_OUT_PROFIT_PCT", "0.7")                  # at/above the target: could never fire
    with pytest.raises(ValueError, match="below PROFIT_TARGET_PCT"):
        Settings.from_env()
    monkeypatch.setenv("SCALE_OUT_PROFIT_PCT", "0.4")
    assert Settings.from_env().risk_limits().scale_out_profit_pct == 0.4


def legs(debit_short, debit_long):
    return [PositionLeg("SPY", FRIDAY, 480, "P", -1, 2.00, debit_short),
            PositionLeg("SPY", FRIDAY, 475, "P", 1, 1.00, debit_long)]


def test_service_uses_the_configured_target():
    # credit 1.00, debit 0.40 -> 60% captured: a hold at the 65% target, an exit at 60%
    for target, expected in ((0.65, "HOLD"), (0.60, "TAKE_PROFIT")):
        s = Settings(broker="mock", watchlist=["SPY"], scan_enabled=False, profit_target_pct=target)
        svc = ScanService(s, MockBroker(today=TODAY, positions=legs(0.40, 0.0)), AlertDispatcher([]),
                          scanner_cfg=ScannerConfig(top_n_per_symbol=1))
        asyncio.run(svc.run_once())
        assert svc.state.positions[0]["action"] == expected, target


def test_backtest_cli_defaults_to_65_and_accepts_an_override(capsys):
    assert backtest_main(["--demo", "--demo-days", "40"]) == 0
    assert "exits at 65% profit" in capsys.readouterr().out
    assert backtest_main(["--demo", "--demo-days", "40", "--profit-target", "0.5"]) == 0
    assert "exits at 50% profit" in capsys.readouterr().out


# --- 3. universe restricted to SPY / QQQ ---
def test_default_universe_is_spy_and_qqq_only():
    s = Settings()
    assert CORE_UNIVERSE == ("SPY", "QQQ") and s.watchlist == ["SPY", "QQQ"] and s.universe_allowlist == ("SPY", "QQQ")


def test_environment_can_widen_or_narrow_the_universe(monkeypatch):
    assert Settings.from_env().watchlist == ["SPY", "QQQ"]
    monkeypatch.setenv("WATCHLIST", "spy, aapl, iwm")
    monkeypatch.setenv("UNIVERSE_ALLOWLIST", "SPY,IWM")
    s = Settings.from_env()
    assert s.watchlist == ["SPY", "AAPL", "IWM"] and s.universe_allowlist == ("SPY", "IWM")


def build(watchlist, allowlist=None, positions=None, earnings=None):
    kw = {} if allowlist is None else {"universe_allowlist": allowlist}
    settings = Settings(broker="mock", watchlist=watchlist, scan_enabled=False, **kw)
    return ScanService(settings, MockBroker(today=TODAY, positions=positions or []), AlertDispatcher([]),
                       earnings=earnings, scanner_cfg=ScannerConfig(top_n_per_symbol=3))


def test_scan_universe_split_is_case_insensitive_and_deduplicated():
    svc = build(["spy", "SPY", "Aapl", "qqq", "IWM"])
    assert svc.scan_universe() == (["SPY", "QQQ"], ["AAPL", "IWM"])


def test_symbols_outside_the_universe_are_reported_never_scanned():
    svc = build(["SPY", "AAPL", "IWM"])
    asyncio.run(svc.run_once())
    reasons = {k["symbol"]: k["reason"] for k in svc.state.skipped}
    assert "outside the core universe" in reasons["AAPL"] and "outside the core universe" in reasons["IWM"]
    assert {s["symbol"] for s in svc.state.signals} == {"SPY"}
    assert "get_option_chain" not in " ".join(svc.broker.calls) and svc.state.last_error is None


def test_open_positions_in_other_symbols_are_still_monitored():
    aapl = [PositionLeg("AAPL", FRIDAY, 190, "P", -1, 2.0, 1.8), PositionLeg("AAPL", FRIDAY, 185, "P", 1, 1.0, 0.9)]
    svc = build(["SPY"], positions=aapl)
    asyncio.run(svc.run_once())
    assert [p["symbol"] for p in svc.state.positions] == ["AAPL"]                  # monitored (exits, alerts, P&L)
    assert all(s["symbol"] == "SPY" for s in svc.state.signals)                    # but nothing new is opened in AAPL


def test_a_stock_can_be_scanned_only_by_explicitly_allowing_it():
    svc = build(["AAPL"], allowlist=("SPY", "QQQ", "AAPL"),
                earnings=_far_earnings("AAPL"))
    asyncio.run(svc.run_once())
    assert svc.state.signals and all(s["symbol"] == "AAPL" for s in svc.state.signals)
    assert not any("outside the core universe" in k["reason"] for k in svc.state.skipped)


def _far_earnings(symbol):
    from src.risk.earnings import StaticEarningsProvider
    return StaticEarningsProvider({symbol: EarningsInfo(symbol, TODAY + timedelta(days=300), EarningsStatus.CONFIRMED)})


def test_earnings_feed_is_not_asked_about_symbols_outside_the_universe():
    class Recorder:
        def __init__(self):
            self.refreshed = []

        def get(self, symbol):
            return EarningsInfo(symbol, None, EarningsStatus.NOT_APPLICABLE)

        def refresh(self, symbols):
            self.refreshed.append(list(symbols))

    rec = Recorder()
    svc = build(["SPY", "AAPL", "MSFT"], earnings=rec)
    asyncio.run(svc.run_once())
    assert rec.refreshed[0] == ["SPY"] and all("AAPL" not in r and "MSFT" not in r for r in rec.refreshed[:1])
