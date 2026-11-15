"""Automated earnings-date feed: vendor adapter -> cached provider -> layered with your manual calendar.

Design rules (this feeds a safety gate, so it fails CLOSED):
  * A vendor date is never "confirmed": the feed cannot tell a company-announced date from a projected one, so it is
    reported as PROJECTED. Only your manual file (``EARNINGS_FILE``, see ``src/risk/earnings.py``) can mark CONFIRMED,
    and manual entries always win over the feed.
  * No usable date -> UNKNOWN -> signals are blocked. That covers: symbol missing from the cache, a failed or
    timed-out fetch with no fresh-enough cached value, an empty vendor response (a vendor that lacks data looks the
    same as "no earnings", so it is NOT read as clear), and a next date implausibly far away (the vendor probably
    skipped the near-term report).
  * A failed refresh keeps the previous value only while it is younger than ``max_stale_hours``; after that it expires.
  * Errors are sanitised: the API key never appears in an exception message, a log line, ``repr`` or the dashboard.
  * ETFs/indices (``DEFAULT_NO_EARNINGS``) never hit the network.

The Finnhub adapter follows the publicly documented ``/calendar/earnings`` shape as best I could confirm (symbol, date,
hour in an ``earningsCalendar`` list) but was NOT verified against the live API from this environment. If the vendor
rejects the auth style you will see the error on the dashboard and signals stay blocked; switch ``EARNINGS_AUTH``
between ``header`` and ``query``. Any other vendor can be added by implementing :class:`EarningsSource`.

Network I/O is blocking and lives only in ``refresh()``; ``ScanService`` runs it in a worker thread, and ``get()``
only reads memory."""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from src.config import Settings
from src.risk.earnings import (
    DEFAULT_NO_EARNINGS,
    EarningsInfo,
    EarningsProvider,
    EarningsStatus,
    JsonFile,
    StaticEarningsProvider,
)

log = logging.getLogger(__name__)


class EarningsFeedError(RuntimeError):
    """A feed request failed. The message never contains credentials, URLs or response bodies."""


class Fetcher(Protocol):
    def get_json(self, url: str, params: dict[str, str], headers: dict[str, str], timeout: float) -> Any: ...


class RequestsFetcher:
    """HTTP GET via ``requests`` with sanitised errors (no URL, params or headers in messages)."""

    def get_json(self, url: str, params: dict[str, str], headers: dict[str, str], timeout: float) -> Any:
        import requests

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.Timeout:
            raise EarningsFeedError("earnings feed timed out") from None
        except requests.RequestException as exc:
            raise EarningsFeedError(f"earnings feed network error ({type(exc).__name__})") from None
        if resp.status_code in (401, 403):
            raise EarningsFeedError(f"earnings feed rejected the credentials (HTTP {resp.status_code}); "
                                    "check EARNINGS_API_KEY / EARNINGS_AUTH")
        if resp.status_code == 429:
            raise EarningsFeedError("earnings feed rate limit hit (HTTP 429)")
        if resp.status_code != 200:
            raise EarningsFeedError(f"earnings feed HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError:
            raise EarningsFeedError("earnings feed returned invalid JSON") from None


@dataclass(frozen=True)
class EarningsRecord:
    date: date
    hour: str | None = None   # "bmo" (before open), "amc" (after close), "dmh" (during) when the vendor says


class EarningsSource(Protocol):
    name: str

    def fetch(self, symbol: str, today: date, horizon_days: int) -> list[EarningsRecord]:
        """Upcoming report dates for ``symbol`` in [today, today + horizon_days]. Raise EarningsFeedError on failure."""


class FinnhubSource:
    name = "finnhub"

    def __init__(self, api_key: str, fetcher: Fetcher | None = None, auth: str = "header", timeout: float = 10.0,
                 base_url: str = "https://finnhub.io/api/v1") -> None:
        if not api_key:
            raise ValueError("an API key is required for the Finnhub earnings feed (EARNINGS_API_KEY)")
        if auth not in {"header", "query"}:
            raise ValueError("EARNINGS_AUTH must be 'header' or 'query'")
        self._key, self._auth, self._timeout, self._base = api_key, auth, timeout, base_url.rstrip("/")
        self._fetcher = fetcher or RequestsFetcher()

    def __repr__(self) -> str:  # never expose the key
        return f"FinnhubSource(auth={self._auth!r}, key=***)"

    def fetch(self, symbol: str, today: date, horizon_days: int) -> list[EarningsRecord]:
        params = {"symbol": symbol, "from": today.isoformat(), "to": (today + timedelta(days=horizon_days)).isoformat()}
        headers: dict[str, str] = {}
        if self._auth == "header":
            headers["X-Finnhub-Token"] = self._key
        else:
            params["token"] = self._key
        data = self._fetcher.get_json(f"{self._base}/calendar/earnings", params, headers, self._timeout)
        rows = data.get("earningsCalendar") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise EarningsFeedError("earnings feed returned an unexpected response shape")
        out: list[EarningsRecord] = []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            try:
                when = date.fromisoformat(str(row["date"])[:10])
            except (KeyError, ValueError):
                continue
            hour = row.get("hour")
            out.append(EarningsRecord(when, str(hour).lower() if hour else None))
        return out


@dataclass
class _Entry:
    date: date | None
    status: EarningsStatus
    fetched_at: datetime
    hour: str | None = None
    note: str | None = None


class FeedEarningsProvider:
    """Cache-backed provider over an :class:`EarningsSource`. ``get`` reads memory; ``refresh`` does the I/O."""

    def __init__(self, source: EarningsSource, cache_path: str | Path | None = None, ttl_hours: float = 12.0,
                 max_stale_hours: float = 72.0, horizon_days: int = 120, max_plausible_days: int = 110,
                 max_calls: int = 30, pace_sec: float = 0.0,
                 no_earnings: frozenset[str] | set[str] = DEFAULT_NO_EARNINGS,
                 clock: Callable[[], datetime] | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.source, self.ttl, self.max_stale = source, timedelta(hours=ttl_hours), timedelta(hours=max_stale_hours)
        self.horizon_days, self.max_plausible_days = horizon_days, max_plausible_days
        self.max_calls, self.pace_sec, self._sleep = max_calls, pace_sec, sleep
        self._no_earnings = {s.upper() for s in no_earnings}
        self._clock = clock or (lambda: datetime.now(UTC))
        self._path = Path(cache_path) if cache_path else None
        self._entries: dict[str, _Entry] = {}
        self.failed: dict[str, str] = {}
        self.last_error: str | None = None
        self.last_refresh: datetime | None = None
        self._load()

    # -- provider interface -----------------------------------------------------------------------------------
    def get(self, symbol: str) -> EarningsInfo:
        sym = symbol.upper()
        if sym in self._no_earnings:
            return EarningsInfo(sym, None, EarningsStatus.NOT_APPLICABLE)
        entry = self._entries.get(sym)
        if entry is None or self._clock() - entry.fetched_at > self.max_stale:
            return EarningsInfo(sym, None, EarningsStatus.UNKNOWN)   # never fetched, or the data is too old to trust
        return EarningsInfo(sym, entry.date, entry.status)

    def refresh(self, symbols: list[str], today: date | None = None) -> None:
        """Fetch symbols whose cache entry is older than the TTL. Never raises: failures are recorded and the symbol
        keeps its previous value only until ``max_stale_hours``."""
        now = self._clock()
        today = today or now.date()
        todo = sorted({s.upper() for s in symbols} - self._no_earnings)
        todo = [s for s in todo if (e := self._entries.get(s)) is None or now - e.fetched_at >= self.ttl]
        errors: list[str] = []
        calls = 0
        for sym in todo:
            if calls >= self.max_calls:
                errors.append(f"call limit ({self.max_calls}) reached; {len(todo) - calls} symbol(s) not refreshed")
                break
            if calls and self.pace_sec:
                self._sleep(self.pace_sec)
            calls += 1
            try:
                self._entries[sym] = self._interpret(sym, self.source.fetch(sym, today, self.horizon_days), today, now)
                self.failed.pop(sym, None)
            except EarningsFeedError as exc:
                self.failed[sym] = str(exc)
                errors.append(f"{sym}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a feed bug must not take the scan down; fail closed instead
                self.failed[sym] = f"unexpected error ({type(exc).__name__})"
                errors.append(f"{sym}: {self.failed[sym]}")
        if todo:
            self.last_refresh = now
            self.last_error = "; ".join(errors) or None
            self._save()
        for line in errors:
            log.warning("earnings feed: %s", line)

    def _interpret(self, sym: str, rows: list[EarningsRecord], today: date, now: datetime) -> _Entry:
        upcoming = sorted((r for r in rows if r.date >= today), key=lambda r: r.date)
        if not upcoming:
            # "vendor has nothing" is indistinguishable from "vendor lacks data": never read it as clear
            return _Entry(None, EarningsStatus.UNKNOWN, now, note="no upcoming date in the feed")
        nxt = upcoming[0]
        if (nxt.date - today).days > self.max_plausible_days:
            return _Entry(None, EarningsStatus.UNKNOWN, now, nxt.hour,
                          note=f"next date {nxt.date} is implausibly far away: near-term report probably missing")
        return _Entry(nxt.date, EarningsStatus.PROJECTED, now, nxt.hour)

    # -- diagnostics ------------------------------------------------------------------------------------------
    def age_hours(self, symbol: str) -> float | None:
        e = self._entries.get(symbol.upper())
        return None if e is None else (self._clock() - e.fetched_at).total_seconds() / 3600.0

    def health(self) -> dict:
        now = self._clock()
        stale = sorted(s for s, e in self._entries.items() if now - e.fetched_at > self.max_stale)
        ok = not self.failed and not stale and self.last_refresh is not None
        refreshed = self.last_refresh.isoformat() if self.last_refresh else None
        return {"provider": self.source.name, "ok": ok, "last_refresh": refreshed, "last_error": self.last_error,
                "failed": dict(self.failed), "stale": stale, "symbols": len(self._entries)}

    # -- persistence ------------------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for sym, row in raw.items():
                self._entries[sym.upper()] = _Entry(
                    date.fromisoformat(row["date"]) if row.get("date") else None, EarningsStatus(row["status"]),
                    datetime.fromisoformat(row["fetched_at"]), row.get("hour"), row.get("note"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("ignoring unreadable earnings cache %s (%s)", self._path, type(exc).__name__)
            self._entries.clear()   # an unreadable cache means UNKNOWN for everything, never a guess

    def _save(self) -> None:
        if not self._path:
            return
        data = {s: {"date": e.date.isoformat() if e.date else None, "status": e.status.value,
                    "fetched_at": e.fetched_at.isoformat(), "hour": e.hour, "note": e.note}
                for s, e in self._entries.items()}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self._path)   # atomic: a crash never leaves a half-written cache
        except OSError as exc:
            log.warning("could not write earnings cache %s (%s)", self._path, type(exc).__name__)


class LayeredEarningsProvider:
    """Manual calendar first (it can say CONFIRMED and always wins), the automated feed for everything else."""

    def __init__(self, manual: EarningsProvider, feed: FeedEarningsProvider) -> None:
        self.manual, self.feed = manual, feed

    def get(self, symbol: str) -> EarningsInfo:
        info = self.manual.get(symbol)
        return info if info.status is not EarningsStatus.UNKNOWN else self.feed.get(symbol)

    def refresh(self, symbols: list[str], today: date | None = None) -> None:
        manual_known = {s.upper() for s in symbols if self.manual.get(s).status is not EarningsStatus.UNKNOWN}
        self.feed.refresh([s for s in symbols if s.upper() not in manual_known], today)

    def health(self) -> dict:
        return self.feed.health()

    def age_hours(self, symbol: str) -> float | None:
        return self.feed.age_hours(symbol)


def build_earnings_provider(settings: Settings, fetcher: Fetcher | None = None,
                            clock: Callable[[], datetime] | None = None) -> EarningsProvider:
    """Manual file (optional) layered over the configured automated feed (optional). Falls back to the manual/static
    provider, which flags every single stock for review, when no feed is configured."""
    manual: EarningsProvider = JsonFile(settings.earnings_file) if settings.earnings_file else StaticEarningsProvider()
    if settings.earnings_provider == "file":
        return manual
    if settings.earnings_provider != "finnhub":
        raise ValueError(f"unknown EARNINGS_PROVIDER '{settings.earnings_provider}' (expected 'file' or 'finnhub')")
    source = FinnhubSource(settings.earnings_api_key, fetcher, settings.earnings_auth)
    feed = FeedEarningsProvider(source, settings.earnings_cache_file or None, settings.earnings_ttl_hours,
                                settings.earnings_max_stale_hours, clock=clock)
    return LayeredEarningsProvider(manual, feed)
