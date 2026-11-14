"""Orchestrator: scan -> guards -> earnings/risk gate -> alert, and monitor open positions for exits.

Read-only: it scans, alerts and logs. It never places orders."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

from src.alerts.notifiers import AlertDispatcher
from src.broker.base import BrokerClient, BrokerError
from src.config import Settings
from src.data.freshness import assess_freshness
from src.data.market_data import MarketDataService
from src.data.stats import UnderlyingStats, build_stats
from src.market_hours import is_market_open
from src.models import OptionQuote, SpreadPosition, SpreadSignal, SpreadType
from src.risk.earnings import EarningsProvider, JsonFile, StaticEarningsProvider
from src.risk.evaluation import PositionReport, Status, enrich_positions, evaluate_position
from src.risk.manager import ExitAction, RiskLimits, RiskManager, pair_legs
from src.risk.portfolio import PortfolioSnapshot, build_snapshot, signal_dollar_delta
from src.risk.report import load_entry_deltas
from src.storage.trade_log import TradeLog
from src.strategies.credit_spreads import ScannerConfig, rank_key, scan_all, schedule_gate, underlying_gate

log = logging.getLogger(__name__)
MAX_RECORDED = 50  # cap on per-scan diagnostics kept for the dashboard


@dataclass
class AppState:
    """Shared, in-memory view consumed by the dashboard."""

    signals: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)   # flattened per-position alerts for the dashboard
    portfolio: dict = field(default_factory=dict)      # beta-weighted delta snapshot (see src.risk.portfolio)
    skipped: list[dict] = field(default_factory=list)  # symbols skipped this scan (stale data, edge filters, ...)
    blocked: list[dict] = field(default_factory=list)  # candidates rejected by the earnings / risk gate
    rejected: int = 0
    last_scan: str | None = None
    last_error: str | None = None
    skip_reason: str | None = None                      # e.g. "market closed" when the whole scan was skipped
    market_open: bool | None = None
    broker_connected: bool = False
    equity: float | None = None
    scans: int = 0


class ScanService:
    def __init__(self, settings: Settings, broker: BrokerClient, dispatcher: AlertDispatcher,
                 risk: RiskManager | None = None, scanner_cfg: ScannerConfig | None = None,
                 state: AppState | None = None, earnings: EarningsProvider | None = None,
                 entry_deltas: dict[str, float] | None = None, trade_log: TradeLog | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.settings, self.broker, self.dispatcher = settings, broker, dispatcher
        self.clock = clock or (lambda: datetime.now(UTC))
        # Fail safe: without a calendar every single stock is flagged for manual review (only known ETFs are clear).
        self.earnings: EarningsProvider = earnings or (
            JsonFile(settings.earnings_file) if settings.earnings_file else StaticEarningsProvider())
        self._journal = entry_deltas  # manual overrides; None -> read settings.entry_deltas_file each cycle
        self.trade_log = trade_log or TradeLog(settings.trade_db_path or ":memory:", settings.commission_per_contract)
        self.risk = risk or RiskManager(RiskLimits(
            scale_out_profit_pct=settings.scale_out_profit_pct or None,
            max_net_beta_delta_pct=settings.max_net_beta_delta_pct))
        self.betas = dict(settings.betas)
        self._snapshot: PortfolioSnapshot | None = None
        self.cfg = scanner_cfg or (ScannerConfig.disciplined() if settings.disciplined_scanner else ScannerConfig())
        if settings.allowed_weekdays is not None and scanner_cfg is None:
            self.cfg = replace(self.cfg, allowed_weekdays=settings.allowed_weekdays)
        self.state = state or AppState()
        # A term-structure check needs expiries beyond the trade window; widen the *data* window only (scan_spreads
        # still restricts signals to cfg.min_dte..max_dte). This costs more IB market-data lines per symbol.
        lo, hi = ((max(7, self.cfg.min_dte - 18), self.cfg.max_dte + 20) if self.cfg.min_term_slope is not None
                  else (self.cfg.min_dte, self.cfg.max_dte))
        self.data = MarketDataService(broker, lo, hi)

    async def _ensure_connected(self) -> None:
        if not self.broker.is_connected():
            await self.broker.connect()
        self.state.broker_connected = self.broker.is_connected()

    async def run_once(self) -> None:
        now = self.clock()
        self.state.market_open = is_market_open(now)
        self.state.skip_reason = None
        try:
            if self.settings.enforce_market_hours and not self.state.market_open:
                # Outside the session the feed is frozen: neither new signals nor position alerts are trustworthy.
                self.state.skip_reason = "market closed"
                self.state.signals, self.state.skipped, self.state.blocked = [], [], []
                return
            await self._ensure_connected()
            equity = await self.broker.get_account_equity() or self.settings.account_equity
            self.state.equity = equity
            positions = await self.check_positions(equity)

            approved: list[SpreadSignal] = []
            rejected, skipped, blocked = 0, [], []
            today = now.date()
            closed_today = schedule_gate(self.cfg, today)
            for symbol in self.settings.watchlist:
                if closed_today:
                    skipped.append({"symbol": symbol, "reason": "; ".join(closed_today)})
                    continue
                try:
                    chain = await self.data.get_chain(symbol, today)
                except BrokerError as exc:
                    log.warning("skip %s: %s", symbol, exc)
                    skipped.append({"symbol": symbol, "reason": str(exc)})
                    continue
                if self.settings.enforce_quote_freshness:
                    fresh = assess_freshness(chain, now, self.settings.quote_age_limit())
                    if not fresh.ok:
                        skipped.append({"symbol": symbol, "reason": f"stale data: {fresh.describe()}"})
                        continue
                stats = await self._stats(symbol, chain, today)
                signals = scan_all(chain, self.cfg, today, stats)
                if not signals:
                    for st in (SpreadType.BULL_PUT, SpreadType.BEAR_CALL):
                        why = underlying_gate(stats, st, self.cfg)
                        if why:
                            skipped.append({"symbol": symbol, "reason": f"{st.value}: " + "; ".join(why)})
                info = self.earnings.get(symbol)
                for sig in signals:
                    # Candidates are alternatives, not a basket: each is tested against the CURRENT portfolio.
                    decision = self.risk.check_signal(
                        sig, positions, equity, check_earnings=self.settings.require_earnings_clear,
                        earnings=info, today=today, portfolio=self._snapshot,
                        signal_dollar_delta=signal_dollar_delta(sig, self.betas))
                    if decision.approved:
                        approved.append(sig)
                    else:
                        rejected += 1
                        blocked.append({"key": sig.key, "reasons": decision.reasons})
            approved.sort(key=rank_key(self.cfg), reverse=True)
            self.state.signals = [s.to_dict() for s in approved]
            self.state.rejected = rejected
            self.state.skipped, self.state.blocked = skipped[:MAX_RECORDED], blocked[:MAX_RECORDED]
            for sig in approved:
                self.trade_log.record_signal(sig, now)
                await self.dispatcher.signal(sig)
            self.state.last_error = None
        except BrokerError as exc:
            self.state.last_error = str(exc)
            self.state.broker_connected = self.broker.is_connected()
            log.error("scan failed: %s", exc)
        finally:
            self.state.last_scan = now.isoformat()
            self.state.scans += 1

    async def _stats(self, symbol: str, chain: list[OptionQuote], today: date) -> UnderlyingStats | None:
        """Underlying stats for the IV/HV and trend filters; None (=> those filters fail closed) if unavailable."""
        if not chain or (self.cfg.min_iv_hv_ratio is None and not self.cfg.require_trend_alignment):
            return None
        try:
            closes = await self.broker.get_daily_closes(symbol)
        except BrokerError as exc:
            log.warning("no price history for %s: %s", symbol, exc)
            closes = []
        return build_stats(symbol, chain[0].underlying_price, closes, chain, today, self.cfg.min_dte, self.cfg.max_dte)

    def _journal_now(self) -> dict[str, float]:
        """Manual entry-delta overrides. Re-read each cycle so edits apply without a restart."""
        if self._journal is not None:
            return self._journal
        try:
            return load_entry_deltas(self.settings.entry_deltas_file or None)
        except ValueError as exc:
            log.warning("%s", exc)
            return {}

    async def _portfolio(self, positions: list[SpreadPosition], equity: float) -> PortfolioSnapshot:
        """Beta-weighted delta snapshot. A missing SPY quote makes it incomplete, which blocks new signals."""
        try:
            spy = await self.broker.get_underlying_price("SPY")
        except BrokerError as exc:
            log.warning("no SPY price for beta-weighted delta: %s", exc)
            spy = None
        return build_snapshot(positions, self.betas, spy, equity, self.risk.limits.max_net_beta_delta_pct)

    async def check_positions(self, equity: float | None = None) -> list[SpreadPosition]:
        """Pair legs, enrich with spot/delta, capture entry delta, run exit rules + alerts, notify and log.
        Read-only: never trades."""
        now, today = self.clock(), date.today()
        positions = pair_legs(await self.broker.get_option_positions())
        journal = self._journal_now()
        known = {**self.trade_log.entry_deltas(), **journal}  # manual journal wins over automatic capture
        enriched = await enrich_positions(self.broker, positions, today, known)

        reports: list[PositionReport] = []
        for pos, notes in enriched:
            pos = replace(pos, initial_contracts=self.trade_log.initial_contracts(pos.key))
            action = self.risk.evaluate_exit(pos, today)
            delta, _source = self.trade_log.observe_position(pos, action.value, now)  # captures entry delta once
            if pos.key in journal:
                self.trade_log.override_entry_delta(pos.key, journal[pos.key])
                delta = abs(journal[pos.key])
            if pos.entry_short_delta is None and delta is not None:
                pos = replace(pos, entry_short_delta=delta)
                notes = [n for n in notes if not n.startswith("no entry delta")]
            reports.append(evaluate_position(self.risk, pos, today, self.earnings, notes))

        self.trade_log.close_missing({r.position.key for r in reports}, now)
        pnl = {t: sum(r.position.unrealized_pnl for r in reports if r.position.spread_type is t) for t in SpreadType}
        self.trade_log.record_snapshot(today, pnl[SpreadType.BULL_PUT], pnl[SpreadType.BEAR_CALL], len(reports), now)
        final = [r.position for r in reports]
        eq = equity or self.state.equity or self.settings.account_equity
        snap = self._snapshot = await self._portfolio(final, eq)
        self.state.portfolio = snap.to_dict()
        dollars = {e.key: e.dollar_delta for e in snap.exposures}
        self.state.positions, self.state.alerts = [], []
        for r in reports:
            await self._publish(r, dollars.get(r.position.key))
        for a in self.risk.evaluate_portfolio(snap):
            self.state.alerts.append({"position": "PORTFOLIO", "type": a.type.value, "severity": a.severity.value,
                                      "message": a.message})
            await self.dispatcher.dispatch(f"portfolio:{a.type.value}:{a.severity.value}",
                                           f"ALERT [{a.severity.value}] {a.type.value}: {a.message}")
        return final

    async def _publish(self, r: PositionReport, beta_delta_dollars: float | None = None) -> None:
        pos = r.position
        self.state.positions.append({
            **pos.to_dict(), "action": r.action.value, "status": r.status.value, "recommendation": r.recommendation,
            "scale_out_contracts": r.scale_out_contracts, "beta_delta_dollars": beta_delta_dollars,
            "alerts": [{"type": a.type.value, "severity": a.severity.value, "message": a.message} for a in r.alerts],
            "notes": r.notes,
        })
        label = f"{pos.symbol} {pos.spread_type.value} {pos.short_strike:g}/{pos.long_strike:g} exp {pos.expiry}"
        if r.action is not ExitAction.HOLD:
            verb = f"CLOSE {r.action.value}"
            if r.action is ExitAction.SCALE_OUT:
                verb = f"SCALE OUT {r.scale_out_contracts} of {pos.contracts} contracts"
            msg = f"{verb}: {label} | P&L ${pos.unrealized_pnl:.0f} ({pos.profit_pct:.0%} of credit)"
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
