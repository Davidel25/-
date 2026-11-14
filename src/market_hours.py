"""US equity-options market-hours guard (regular session only; extended hours are treated as closed).

Regular session 09:30-16:00 America/New_York, Mon-Fri, excluding NYSE holidays; early closes (13:00) on the day
after Thanksgiving, Christmas Eve and July 3rd when they are weekdays. The holiday rules are computed, not
fetched: verify against the NYSE calendar if it matters to you, and remember that special closures (e.g. a
national day of mourning) are not covered."""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
OPEN, CLOSE, EARLY_CLOSE = time(9, 30), time(16, 0), time(13, 0)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:  # anonymous Gregorian algorithm
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    """Saturday holidays are observed Friday, Sunday holidays Monday."""
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


def nyse_holidays(year: int) -> set[date]:
    days = {
        _nth_weekday(year, 1, 0, 3),                 # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),                 # Presidents Day
        _easter(year) - timedelta(days=2),           # Good Friday
        _last_weekday(year, 5, 0),                   # Memorial Day
        _observed(date(year, 6, 19)),                # Juneteenth
        _observed(date(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                 # Labor Day
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving
        _observed(date(year, 12, 25)),               # Christmas
    }
    if date(year, 1, 1).weekday() != 5:              # a Saturday New Year's Day is NOT observed on Dec 31
        days.add(_observed(date(year, 1, 1)))
    return days


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def session_close(d: date) -> time:
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    early = {thanksgiving + timedelta(days=1), date(d.year, 12, 24)}
    if date(d.year, 7, 4).weekday() != 0:            # July 3rd is a half day unless July 4th is a Monday
        early.add(date(d.year, 7, 3))
    return EARLY_CLOSE if d in early else CLOSE


def is_market_open(now: datetime | None = None) -> bool:
    """True during the regular session. ``now`` must be tz-aware (naive values are assumed to be UTC)."""
    now = now or datetime.now(UTC)
    local = (now if now.tzinfo else now.replace(tzinfo=UTC)).astimezone(NY)
    return is_trading_day(local.date()) and OPEN <= local.time() < session_close(local.date())
