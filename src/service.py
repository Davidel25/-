"""Orchestrator: scan -> risk-check -> alert, and monitor open positions for exits."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from src.alerts.notifiers import AlertDispatcher
from src.broker.base import BrokerClient, BrokerError
from src.config import Settings
from src.data.market_data import MarketDataService
from src.models import SpreadPosition, SpreadSignal
from src.risk.earnings import EarningsProvider, JsonFile, StaticEarningsProvider
from src.risk.evaluation import PositionReport, Status, enrich_positions, evaluate_position
from src.risk.manager import ExitAction, RiskManager, pair_legs
from src.risk.report import load_entry_deltas
from src.strategies.credit_spreads import ScannerConfig, scan_all

log = logging.getLogger(__name__)


@dataclass
class AppState:
    """Shared, in-memory view consumed by the dashboard."""

    signals: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)  # flattened per-position alerts for the dashboard
    rejected: int = 0
    last_scan: str | None = None
    last_error: str | None = None
    broker_connected: bool = False
    equity: float | None = None
    scans: int = 0


class ScanService:
    def __init__(self, settings: Settings, broker: BrokerClient, dispatcher: AlertDispatcher,
                 risk: RiskManager | None = None, scanner_cfg: ScannerConfig | None = None,
                 state: AppState | None = None, earnings: EarningsProvider | None = None,
                 entry_deltas: dict[str, float] | None = None) -> None:
        self.settings, self.broker, self.dispatcher = settings, broker, dispatcher
        # Fail safe: without a calendar every single stock is flagged for manual review (only known ETFs are clear).
        self.earnings: EarningsProvider = earnings or (
            JsonFile(settings.earnings_file) if settings.earnings_file else StaticEarningsProvider())
        self._entry_deltas = entry_deltas

        self.risk = risk or RiskManager()
        self.cfg = scanner_cfg or ScannerConfig()
        self.state = state or AppState()
        self.data = MarketDataService(broker, self.cfg.min_dte, self.cfg.max_dte)

    async def _ensure_connected(self) -> None:
        if not self.broker.is_connected():
            await self.broker.connect()
        self.state.broker_connected = self.broker.is_connected()

    async def run_once(self) -> None:
        try:
            await self._ensure_connected()
            equity = await self.broker.get_account_equity() or self.settings.account_equity
            self.state.equity = equity
            positions = await self.check_positions()

            approved: list[SpreadSignal] = []
            rejected = 0
            for symbol in self.settings.watchlist:
                try:
                    chain = await self.data.get_chain(symbol)
                except BrokerError as exc:
                    log.warning("skip %s: %s", symbol, exc)
                    continue
                for sig in scan_all(chain, self.cfg):
                    if self.risk.check_signal(sig, positions, equity).approved:
                        approved.append(sig)
                    else:
                        rejected += 1
            approved.sort(key=lambda s: s.roc, reverse=True)
            self.state.signals = [s.to_dict() for s in approved]
            self.state.rejected = rejected
            for sig in approved:
                await self.dispatcher.signal(sig)
            self.state.last_error = None
        except BrokerError as exc:
            self.state.last_error = str(exc)
            self.state.broker_connected = self.broker.is_connected()
            log.error("scan failed: %s", exc)
        finally:
            self.state.last_scan = datetime.now(UTC).isoformat()
            self.state.scans += 1

    def _entry_deltas_now(self) -> dict[str, float]:
        """Re-read the journal each cycle so newly opened trades are picked up without a restart."""
        if self._entry_deltas is not None:
            return self._entry_deltas
        try:
            return load_entry_deltas(self.settings.entry_deltas_file or None)
        except ValueError as exc:
            log.warning("%s", exc)
            return {}

    async def check_positions(self) -> list[SpreadPosition]:
        """Pair legs, enrich with spot/delta, run exit rules + alerts, notify. Read-only: never trades."""
        today = date.today()
        positions = pair_legs(await self.broker.get_option_positions())
        enriched = await enrich_positions(self.broker, positions, today, self._entry_deltas_now())
        reports = [evaluate_position(self.risk, p, today, self.earnings, notes) for p, notes in enriched]
        self.state.positions, self.state.alerts = [], []
        for r in reports:
            await self._publish(r)
        return [r.position for r in reports]

    async def _publish(self, r: PositionReport) -> None:
        pos = r.position
        self.state.positions.append({
            **pos.to_dict(), "action": r.action.value, "status": r.status.value, "recommendation": r.recommendation,
            "alerts": [{"type": a.type.value, "severity": a.severity.value, "message": a.message} for a in r.alerts],
            "notes": r.notes,
        })
        label = f"{pos.symbol} {pos.spread_type.value} {pos.short_strike:g}/{pos.long_strike:g} exp {pos.expiry}"
        if r.action is not ExitAction.HOLD:
            msg = (f"CLOSE {r.action.value}: {label} | "
                   f"P&L ${pos.unrealized_pnl:.0f} ({pos.profit_pct:.0%} of credit)")
            await self.dispatcher.dispatch(f"exit:{r.action.value}:{pos.symbol}:{pos.expiry}:{pos.short_strike}", msg)
        for a in r.alerts:
            self.state.alerts.append({"position": pos.key, "type": a.type.value, "severity": a.severity.value,
                                      "message": a.message})
            await self.dispatcher.dispatch(f"alert:{a.type.value}:{pos.key}",
                                           f"ALERT [{a.severity.value}] {a.type.value}: {label} | {a.message}")

    def position_counts(self) -> dict[str, int]:
        c = {s.value: 0 for s in Status}
        for p in self.state.positions:
            c[p["status"]] += 1
        return c

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.settings.scan_interval_sec)
