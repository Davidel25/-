"""Stale-quote guard. A scan on frozen or hours-old quotes produces credits that cannot be filled, so a chain
whose quotes are too old (or carry no timestamp at all) is rejected as a whole. Fail-safe by design."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from src.models import OptionQuote


@dataclass(frozen=True)
class Freshness:
    total: int
    stale: int
    unknown: int
    newest_age_sec: float | None

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.stale == 0 and self.unknown == 0

    def describe(self) -> str:
        if self.total == 0:
            return "empty chain"
        age = f", newest quote {self.newest_age_sec:.0f}s old" if self.newest_age_sec is not None else ""
        return f"{self.stale}/{self.total} quotes stale, {self.unknown} without timestamp{age}"


def assess_freshness(chain: list[OptionQuote], now: datetime, max_age_sec: float,
                     max_stale_fraction: float = 0.10) -> Freshness:
    """Count quotes older than ``max_age_sec``. Up to ``max_stale_fraction`` of the chain may be stale or
    untimed (illiquid strikes tick rarely); the rest must be recent. The *newest* age is reported so a fully
    frozen feed (weekend / halted data) is obvious."""
    if not chain:
        return Freshness(0, 0, 0, None)
    now = now if now.tzinfo else now.replace(tzinfo=UTC)
    ages = [(now - (q.quote_time if q.quote_time.tzinfo else q.quote_time.replace(tzinfo=UTC))).total_seconds()
            for q in chain if q.quote_time]
    unknown = len(chain) - len(ages)
    stale = sum(a > max_age_sec for a in ages)
    allowed = int(len(chain) * max_stale_fraction)
    newest = min(ages) if ages else None
    # fully frozen feed -> always fail; otherwise tolerate a small illiquid tail
    if newest is None or newest > max_age_sec:
        return Freshness(len(chain), len(chain), unknown, newest)
    bad = stale + unknown
    return Freshness(len(chain), 0 if bad <= allowed else bad, 0 if bad <= allowed else unknown, newest)
