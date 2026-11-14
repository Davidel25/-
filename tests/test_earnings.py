import json
from datetime import date, timedelta

import pytest

from src.models import SpreadPosition, SpreadSignal, SpreadType
from src.risk.earnings import (
    EarningsCheck,
    EarningsInfo,
    EarningsStatus,
    JsonFile,
    StaticEarningsProvider,
    classify_earnings,
)
from src.risk.manager import AlertType, RiskManager, Severity

TODAY = date(2026, 11, 14)
EXP = date(2026, 12, 18)


def info(when, status=EarningsStatus.CONFIRMED, sym="TJX"):
    return EarningsInfo(sym, when, status)


def spread(symbol="TJX"):
    return SpreadPosition(SpreadType.BULL_PUT, symbol, EXP, 120, 115, 1, 0.9, 0.5)


def test_earnings_inside_window_is_risk_even_on_boundaries():
    assert classify_earnings(info(date(2026, 11, 18)), EXP, TODAY) is EarningsCheck.RISK
    assert classify_earnings(info(TODAY), EXP, TODAY) is EarningsCheck.RISK                 # today counts
    assert classify_earnings(info(EXP), EXP, TODAY) is EarningsCheck.RISK                   # expiry day counts
    assert classify_earnings(info(date(2026, 11, 18), EarningsStatus.PROJECTED), EXP, TODAY) is EarningsCheck.RISK


def test_confirmed_earnings_after_expiry_are_clear():
    assert classify_earnings(info(EXP + timedelta(days=1)), EXP, TODAY) is EarningsCheck.CLEAR
    assert classify_earnings(info(date(2027, 2, 1)), EXP, TODAY) is EarningsCheck.CLEAR


def test_unconfirmed_date_just_after_expiry_needs_review_but_far_dates_are_clear():
    near = info(EXP + timedelta(days=5), EarningsStatus.PROJECTED)
    far = info(EXP + timedelta(days=60), EarningsStatus.PROJECTED)
    assert classify_earnings(near, EXP, TODAY) is EarningsCheck.UNCONFIRMED
    assert classify_earnings(far, EXP, TODAY) is EarningsCheck.CLEAR


def test_missing_stale_or_unknown_data_fails_safe():
    assert classify_earnings(None, EXP, TODAY) is EarningsCheck.UNKNOWN
    assert classify_earnings(info(None, EarningsStatus.UNKNOWN), EXP, TODAY) is EarningsCheck.UNKNOWN
    assert classify_earnings(info(None, EarningsStatus.PROJECTED), EXP, TODAY) is EarningsCheck.UNKNOWN
    assert classify_earnings(info(TODAY - timedelta(days=1)), EXP, TODAY) is EarningsCheck.UNKNOWN  # stale date


def test_etfs_have_no_earnings_event():
    assert classify_earnings(StaticEarningsProvider().get("SPY"), EXP, TODAY) is EarningsCheck.CLEAR


def test_risk_manager_raises_high_priority_earnings_alert_inside_window():
    rm = RiskManager()
    alerts = rm.evaluate_alerts(spread(), TODAY, check_earnings=True, earnings=info(date(2026, 11, 18)))
    assert [a.type for a in alerts] == [AlertType.EARNINGS_RISK_ALERT]
    assert alerts[0].severity is Severity.HIGH and "exit or roll" in alerts[0].message


def test_risk_manager_is_silent_when_earnings_are_after_expiry_or_check_is_off():
    rm = RiskManager()
    assert rm.evaluate_alerts(spread(), TODAY, check_earnings=True, earnings=info(date(2027, 2, 1))) == []
    assert rm.evaluate_alerts(spread(), TODAY, check_earnings=False, earnings=info(date(2026, 11, 18))) == []


def test_risk_manager_flags_unknown_and_unconfirmed_for_manual_review():
    rm = RiskManager()
    unknown = rm.evaluate_alerts(spread(), TODAY, check_earnings=True, earnings=None)
    assert [a.type for a in unknown] == [AlertType.EARNINGS_UNKNOWN] and unknown[0].severity is Severity.WARN
    soft = rm.evaluate_alerts(spread(), TODAY, check_earnings=True,
                              earnings=info(EXP + timedelta(days=3), EarningsStatus.PROJECTED))
    assert [a.type for a in soft] == [AlertType.EARNINGS_UNKNOWN]


def sig(symbol="TJX"):
    return SpreadSignal(SpreadType.BULL_PUT, symbol, EXP, 34, 134, 120, 115, 5, 0.9, 90, 410,
                        0.22, 0.15, 0.85, 119, 0.1)


def test_check_signal_blocks_candidates_with_earnings_risk_when_enabled():
    rm = RiskManager()
    bad = rm.check_signal(sig(), [], 100_000, check_earnings=True, earnings=info(date(2026, 11, 18)), today=TODAY)
    assert not bad.approved and "earnings check RISK" in bad.reasons[-1]
    unknown = rm.check_signal(sig(), [], 100_000, check_earnings=True, earnings=None, today=TODAY)
    assert not unknown.approved                                   # fail safe
    ok = rm.check_signal(sig(), [], 100_000, check_earnings=True, earnings=info(date(2027, 2, 1)), today=TODAY)
    assert ok.approved
    assert rm.check_signal(sig(), [], 100_000).approved           # default: check disabled, behaviour unchanged


def test_json_file_provider(tmp_path):
    f = tmp_path / "earn.json"
    f.write_text(json.dumps({"tjx": {"date": "2026-11-18", "confirmed": True},
                             "dell": {"date": "2026-11-24"}, "xyz": {"etf": True}}))
    p = JsonFile(f)
    assert p.get("TJX") == EarningsInfo("TJX", date(2026, 11, 18), EarningsStatus.CONFIRMED)
    assert p.get("DELL").status is EarningsStatus.PROJECTED
    assert p.get("XYZ").status is EarningsStatus.NOT_APPLICABLE
    assert p.get("SPY").status is EarningsStatus.NOT_APPLICABLE   # built-in ETF list
    assert p.get("NVDA").status is EarningsStatus.UNKNOWN         # not listed -> unknown, never clear


@pytest.mark.parametrize("content", ["not json", "[]", '{"A": 5}', '{"A": {"date": "soon"}}', '{"A": {}}'])
def test_json_file_provider_rejects_bad_files(tmp_path, content):
    f = tmp_path / "bad.json"
    f.write_text(content)
    with pytest.raises(ValueError):
        JsonFile(f)
    with pytest.raises(ValueError):
        JsonFile(tmp_path / "missing.json")
