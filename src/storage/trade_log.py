"""Durable SQLite trade log: signals, positions (with automatically captured entry delta), lifecycle events and
estimated net P&L after costs.

Read-only system: nothing here records *orders we placed*; positions are discovered from the broker's portfolio,
so entry/exit data are inferred. Be explicit about that:
  * ``entry_short_delta`` is taken from the matching signal if we generated one, else from the first scan that
    sees the position (``entry_delta_source`` says which), else from a manual journal override.
  * A position that disappears from the broker is marked CLOSED using the *last seen* debit, so P&L is an
    ESTIMATE (``pnl_is_estimate = 1``). Reconcile with broker statements before trusting it.
  * Net P&L = gross - commissions, with commissions = per-contract rate x 2 legs x contracts x 2 (open + close).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.models import CONTRACT_MULTIPLIER, SpreadPosition, SpreadSignal, dte_bucket

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, scan_date TEXT NOT NULL, created_at TEXT NOT NULL,
    symbol TEXT, spread_type TEXT, expiry TEXT, short_strike REAL, long_strike REAL, credit REAL, max_loss REAL,
    roc REAL, short_delta REAL, underlying_price REAL, dte INTEGER, dte_bucket TEXT, weekday INTEGER,
    UNIQUE (key, scan_date));
CREATE TABLE IF NOT EXISTS positions (
    key TEXT PRIMARY KEY, symbol TEXT, spread_type TEXT, expiry TEXT, short_strike REAL, long_strike REAL,
    contracts INTEGER, first_seen TEXT, last_seen TEXT, entry_credit REAL, entry_underlying REAL,
    entry_short_delta REAL, entry_delta_source TEXT, last_debit REAL, last_action TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN', closed_at TEXT, exit_debit REAL, exit_reason TEXT,
    commission REAL, gross_pnl REAL, net_pnl REAL, pnl_is_estimate INTEGER NOT NULL DEFAULT 1,
    missing_scans INTEGER NOT NULL DEFAULT 0, initial_contracts INTEGER, opened_dte INTEGER,
    opened_dte_bucket TEXT, opened_weekday INTEGER, tags TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, key TEXT, type TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS pnl_snapshots (
    day TEXT PRIMARY KEY, unrealized_bull_put REAL NOT NULL DEFAULT 0, unrealized_bear_call REAL NOT NULL DEFAULT 0,
    open_positions INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);
"""


# A short, fixed vocabulary keeps tags countable (free-form tags cannot be aggregated). Put detail in the notes.
ALLOWED_TAGS = ("earnings_gap", "macro_selloff", "macro_rally", "stop_hit", "profit_target", "dte_exit", "scale_out",
                "rolled", "scratch", "manual_exit", "data_issue", "other")
MAX_NOTES = 500

# Columns added after the first schema version: applied to databases created by older versions.
_ADDED_COLUMNS = {
    "signals": {"dte": "INTEGER", "dte_bucket": "TEXT", "weekday": "INTEGER"},
    "positions": {"missing_scans": "INTEGER NOT NULL DEFAULT 0", "initial_contracts": "INTEGER",
                  "opened_dte": "INTEGER", "opened_dte_bucket": "TEXT", "opened_weekday": "INTEGER",
                  "tags": "TEXT", "notes": "TEXT"},
}


@dataclass(frozen=True)
class Performance:
    closed: int
    wins: int
    gross_pnl: float
    commissions: float
    net_pnl: float

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.closed if self.closed else None


def _ts(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


class TradeLog:
    """Thin, thread-safe wrapper over one SQLite connection. ``path=":memory:"`` (or "") keeps it in RAM."""

    def __init__(self, path: str | Path = ":memory:", commission_per_contract: float = 0.65) -> None:
        self.commission_per_contract = commission_per_contract
        target = str(path) or ":memory:"
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(target, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            if target != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        for table, cols in _ADDED_COLUMNS.items():
            have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")  # noqa: S608

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- signals ---------------------------------------------------------------------------------------------
    def record_signal(self, s: SpreadSignal, now: datetime | None = None) -> bool:
        """Insert once per (signal key, day). Returns True if it was new."""
        when = now or datetime.now(UTC)
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO signals (key, scan_date, created_at, symbol, spread_type, expiry,"
                " short_strike, long_strike, credit, max_loss, roc, short_delta, underlying_price, dte, dte_bucket,"
                " weekday) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.key, when.date().isoformat(), _ts(when), s.symbol, s.spread_type.value, s.expiry.isoformat(),
                 s.short_strike, s.long_strike, s.credit, s.max_loss, s.roc, s.short_delta, s.underlying_price, s.dte,
                 dte_bucket(s.dte), when.astimezone(UTC).weekday()))
            return cur.rowcount == 1

    def signal_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

    def _signal_delta(self, key: str) -> float | None:
        row = self._db.execute("SELECT short_delta FROM signals WHERE key = ? ORDER BY id LIMIT 1", (key,)).fetchone()
        return row["short_delta"] if row else None

    # -- positions -------------------------------------------------------------------------------------------
    def observe_position(self, pos: SpreadPosition, action: str,
                         now: datetime | None = None) -> tuple[float | None, str | None]:
        """Record a position seen on this scan. On first sight, capture the entry delta automatically
        (signal delta > current delta). A missing entry delta is back-filled by a later scan ('auto_late').
        Returns (entry_short_delta, source)."""
        key, ts = pos.key, _ts(now)
        opened = (now or datetime.now(UTC)).astimezone(UTC)
        opened_dte = (pos.expiry - opened.date()).days
        with self._lock, self._db:
            row = self._db.execute("SELECT * FROM positions WHERE key = ?", (key,)).fetchone()
            if row is None:
                sig_delta = self._signal_delta(key)
                delta, source = ((sig_delta, "signal") if sig_delta is not None
                                 else (abs(pos.short_delta), "auto") if pos.short_delta is not None else (None, None))
                self._db.execute(
                    "INSERT INTO positions (key, symbol, spread_type, expiry, short_strike, long_strike, contracts,"
                    " first_seen, last_seen, entry_credit, entry_underlying, entry_short_delta, entry_delta_source,"
                    " last_debit, last_action, initial_contracts, opened_dte, opened_dte_bucket, opened_weekday)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (key, pos.symbol, pos.spread_type.value, pos.expiry.isoformat(), pos.short_strike, pos.long_strike,
                     pos.contracts, ts, ts, pos.entry_credit, pos.underlying_price, delta, source, pos.current_debit,
                     action, pos.contracts, opened_dte, dte_bucket(opened_dte), opened.weekday()))
                self._event(ts, key, "POSITION_OPENED", {"entry_short_delta": delta, "source": source})
                return delta, source
            delta, source = row["entry_short_delta"], row["entry_delta_source"]
            if delta is None and pos.short_delta is not None:
                delta, source = abs(pos.short_delta), "auto_late"
                self._db.execute("UPDATE positions SET entry_short_delta=?, entry_delta_source=? WHERE key=?",
                                 (delta, source, key))
            self._db.execute(
                "UPDATE positions SET last_seen=?, last_debit=?, last_action=?, contracts=?, status='OPEN',"
                " closed_at=NULL, missing_scans=0 WHERE key=?", (ts, pos.current_debit, action, pos.contracts, key))
            return delta, source

    def override_entry_delta(self, key: str, delta: float) -> bool:
        """Manual journal wins over automatic capture."""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE positions SET entry_short_delta=?, entry_delta_source='manual' WHERE key=?", (abs(delta), key))
            return cur.rowcount == 1

    def initial_contracts(self, key: str) -> int | None:
        """Size when the position was first seen; lets the scale-out rule know a position was already reduced."""
        with self._lock:
            row = self._db.execute("SELECT initial_contracts FROM positions WHERE key = ?", (key,)).fetchone()
        return row["initial_contracts"] if row else None

    def entry_deltas(self) -> dict[str, float]:
        with self._lock:
            rows = self._db.execute("SELECT key, entry_short_delta FROM positions WHERE entry_short_delta IS NOT NULL")
            return {r["key"]: r["entry_short_delta"] for r in rows}

    def close_missing(self, seen_keys: set[str], now: datetime | None = None, min_missing: int = 2) -> list[str]:
        """Mark OPEN positions that the broker no longer reports as CLOSED, with estimated P&L.
        A position must be absent for ``min_missing`` consecutive scans first: IB's portfolio can be briefly
        empty right after a (re)connect and must not close everything."""
        ts, closed = _ts(now), []
        with self._lock, self._db:
            for row in self._db.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall():
                if row["key"] in seen_keys:
                    continue
                if row["missing_scans"] + 1 < min_missing:
                    self._db.execute("UPDATE positions SET missing_scans = missing_scans + 1 WHERE key=?",
                                     (row["key"],))
                    continue
                n, debit = row["contracts"], row["last_debit"] or 0.0
                gross = (row["entry_credit"] - debit) * CONTRACT_MULTIPLIER * n
                fees = self.commission_per_contract * 2 * n * 2
                self._db.execute(
                    "UPDATE positions SET status='CLOSED', closed_at=?, exit_debit=?, exit_reason=?, commission=?,"
                    " gross_pnl=?, net_pnl=?, pnl_is_estimate=1 WHERE key=?",
                    (ts, debit, row["last_action"] or "HOLD", fees, round(gross, 2), round(gross - fees, 2),
                     row["key"]))
                self._event(ts, row["key"], "POSITION_CLOSED",
                            {"exit_debit": debit, "gross": round(gross, 2), "net": round(gross - fees, 2),
                             "estimate": True})
                closed.append(row["key"])
        return closed

    def _event(self, ts: str, key: str | None, type_: str, detail: dict) -> None:
        self._db.execute("INSERT INTO events (ts, key, type, detail) VALUES (?,?,?,?)",
                         (ts, key, type_, json.dumps(detail)))

    def record_event(self, type_: str, key: str | None = None, detail: dict | None = None,
                     now: datetime | None = None) -> None:
        with self._lock, self._db:
            self._event(_ts(now), key, type_, detail or {})

    # -- reporting -------------------------------------------------------------------------------------------
    def closed_trades(self, limit: int = 50, start: str | None = None, end: str | None = None) -> list[dict]:
        """Newest first. ``start`` / ``end`` are inclusive YYYY-MM-DD dates compared with the close date (UTC)."""
        sql, args = "SELECT * FROM positions WHERE status='CLOSED'", []
        if start:
            sql, args = sql + " AND substr(closed_at,1,10) >= ?", args + [start]
        if end:
            sql, args = sql + " AND substr(closed_at,1,10) <= ?", args + [end]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY closed_at DESC LIMIT ?", (*args, limit))
            return [self._public(r) for r in rows]

    @staticmethod
    def _public(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
        return d

    def set_tags(self, key: str, tags: list[str], notes: str | None = None) -> dict:
        """Tag a trade with values from :data:`ALLOWED_TAGS` and attach a free-text note. Replaces any previous
        tags/notes. Raises ValueError for unknown tags or an over-long note, KeyError for an unknown trade."""
        clean = list(dict.fromkeys(t.strip().lower() for t in tags if t and t.strip()))
        bad = [t for t in clean if t not in ALLOWED_TAGS]
        if bad:
            raise ValueError(f"unknown tag(s) {bad}; allowed: {', '.join(ALLOWED_TAGS)}")
        note = (notes or "").strip() or None
        if note and len(note) > MAX_NOTES:
            raise ValueError(f"notes are limited to {MAX_NOTES} characters")
        with self._lock, self._db:
            cur = self._db.execute("UPDATE positions SET tags=?, notes=? WHERE key=?",
                                   (json.dumps(clean) if clean else None, note, key))
            if cur.rowcount != 1:
                raise KeyError(key)
            self._event(_ts(None), key, "TRADE_TAGGED", {"tags": clean, "has_notes": bool(note)})
            return self._public(self._db.execute("SELECT * FROM positions WHERE key=?", (key,)).fetchone())

    def record_snapshot(self, day: date, unrealized_bull_put: float, unrealized_bear_call: float,
                        open_positions: int, now: datetime | None = None) -> None:
        """One row per day (the latest scan of the day wins): open-position P&L split by strategy."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO pnl_snapshots (day, unrealized_bull_put, unrealized_bear_call, open_positions, updated_at)"
                " VALUES (?,?,?,?,?) ON CONFLICT(day) DO UPDATE SET unrealized_bull_put=excluded.unrealized_bull_put,"
                " unrealized_bear_call=excluded.unrealized_bear_call, open_positions=excluded.open_positions,"
                " updated_at=excluded.updated_at",
                (day.isoformat(), round(unrealized_bull_put, 2), round(unrealized_bear_call, 2), open_positions,
                 _ts(now)))

    def snapshots(self, start: str | None = None, end: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM pnl_snapshots WHERE 1=1", []
        if start:
            sql, args = sql + " AND day >= ?", args + [start]
        if end:
            sql, args = sql + " AND day <= ?", args + [end]
        with self._lock:
            return [dict(r) for r in self._db.execute(sql + " ORDER BY day", args)]

    def performance(self) -> Performance:
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(net_pnl > 0),0) w, COALESCE(SUM(gross_pnl),0) g,"
                " COALESCE(SUM(commission),0) c, COALESCE(SUM(net_pnl),0) p FROM positions WHERE status='CLOSED'"
            ).fetchone()
        return Performance(r["n"], int(r["w"]), round(r["g"], 2), round(r["c"], 2), round(r["p"], 2))

    def breakdown(self, by: str = "dte_bucket") -> list[dict]:
        """Closed-trade performance grouped by entry DTE bucket or entry weekday (0 = Monday): the data needed to
        decide whether a weekday / DTE-bucket filter actually helps. Estimates, see the module docstring."""
        if by == "tag":
            return self._tag_breakdown()
        col = {"dte_bucket": "opened_dte_bucket", "weekday": "opened_weekday"}.get(by)
        if col is None:
            raise ValueError("by must be 'dte_bucket', 'weekday' or 'tag'")
        with self._lock:
            rows = self._db.execute(
                f"SELECT {col} g, COUNT(*) n, SUM(net_pnl > 0) w, SUM(net_pnl) net, AVG(net_pnl) avg"  # noqa: S608
                f" FROM positions WHERE status='CLOSED' AND {col} IS NOT NULL GROUP BY {col} ORDER BY {col}")
            return [{"group": r["g"], "trades": r["n"], "wins": int(r["w"] or 0), "net_pnl": round(r["net"] or 0, 2),
                     "avg_net_pnl": round(r["avg"] or 0, 2)} for r in rows]

    def _tag_breakdown(self) -> list[dict]:
        """A trade with several tags counts once under each; untagged trades are grouped as 'untagged'."""
        with self._lock:
            rows = self._db.execute("SELECT tags, net_pnl FROM positions WHERE status='CLOSED'").fetchall()
        agg: dict[str, list[float]] = {}
        for r in rows:
            for tag in (json.loads(r["tags"]) if r["tags"] else ["untagged"]):
                agg.setdefault(tag, []).append(r["net_pnl"] or 0.0)
        return [{"group": g, "trades": len(v), "wins": sum(x > 0 for x in v), "net_pnl": round(sum(v), 2),
                 "avg_net_pnl": round(sum(v) / len(v), 2)} for g, v in sorted(agg.items())]

    def events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]
