"""Orchestrator: scan -> risk-check -> alert, and monitor open positions for exits."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.alerts.notifiers import AlertDispatcher
from src.broker.base import BrokerClient, BrokerError
from src.config import Settings
from src.data.market_data import MarketDataService
from src.models import SpreadPosition, SpreadSignal
from src.risk.manager import ExitAction, RiskManager, pair_legs
from src.strategies.credit_spreads import ScannerConfig, scan_all

log = logging.getLogger(__name__)


@dataclass
class AppState:
    """Shared, in-memory view consumed by the dashboard."""

    signals: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    rejected: int = 0
    last_scan: str | None = None
    last_error: str | None = None
    broker_connected: bool = False
    equity: float | None = None
    scans: int = 0


class ScanService:
    def __init__(self, settings: Settings, broker: BrokerClient, dispatcher: AlertDispatcher,
                 risk: RiskManager | None = None, scanner_cfg: ScannerConfig | None = None,
                 state: AppState | None = None) -> None:
        self.settings, self.broker, self.dispatcher = settings, broker, dispatcher
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

    async def check_positions(self) -> list[SpreadPosition]:
        positions = pair_legs(await self.broker.get_option_positions())
        self.state.positions = []
        for pos in positions:
            action = self.risk.evaluate_exit(pos)
            self.state.positions.append({**pos.to_dict(), "action": action.value})
            if action is not ExitAction.HOLD:
                msg = (f"CLOSE {action.value}: {pos.symbol} {pos.spread_type.value} "
                       f"{pos.short_strike:g}/{pos.long_strike:g} exp {pos.expiry} | "
                       f"P&L ${pos.unrealized_pnl:.0f} ({pos.profit_pct:.0%} of credit)")
                await self.dispatcher.dispatch(f"exit:{action.value}:{pos.symbol}:{pos.expiry}:{pos.short_strike}", msg)
        return positions

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.settings.scan_interval_sec)
