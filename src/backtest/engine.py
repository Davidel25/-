"""Replay historical option-chain snapshots through the REAL scanner and risk rules.

For each trading day, in order:
  1. Open positions are repriced from that day's quotes and run through ``RiskManager.evaluate_exit`` (configured
     profit target, 2x-credit stop, 14-DTE exit: exactly the live rules). Closed at the day's natural cost-to-close.
  2. New entries: the scanner (``scan_all`` with the live edge filters and underlying stats computed ONLY from data up
     to that day), the earnings gate, and ``RiskManager.check_signal`` decide what is opened, at that day's prices.

Honest limits (also printed in every report)
  * End-of-day snapshots: a signal and its fill use the same EOD quotes. A real fill happens later and usually worse.
  * Natural pricing assumes you cross the spread on every leg; no queue position, no partial fills, no market impact.
  * No early assignment, dividends, or pin risk; positions that reach expiry settle at intrinsic.
  * Earnings dates are the ones YOU supply, known with hindsight; without them single stocks are not traded (ETFs are).
  * Trades overlap in time and across symbols, so they are correlated: effective sample size < trade count.
  * The strategy parameters were designed after reading about this market, so in-sample results flatter it.
    Use the out-of-sample split, and treat anything short of many independent months as inconclusive."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from src.backtest.data import ChainStore
from src.data.stats import build_stats
from src.models import CONTRACT_MULTIPLIER, SpreadPosition, SpreadSignal, SpreadType, dte_bucket
from src.risk.earnings import DEFAULT_NO_EARNINGS, EarningsCheck, EarningsInfo, EarningsStatus, classify_earnings
from src.risk.manager import ExitAction, RiskLimits, RiskManager
from src.strategies.credit_spreads import ScannerConfig, rank_key, scan_all


@dataclass
class BacktestConfig:
    scanner: ScannerConfig = field(default_factory=ScannerConfig.disciplined)
    limits: RiskLimits = field(default_factory=RiskLimits.disciplined)   # 65% target, 2x stop, DTE <= 14
    equity: float = 20_000.0
    commission_per_contract: float = 0.65
    slippage_per_leg: float = 0.0           # extra $ per share on each of the 4 leg fills (open + close)
    exit_pricing: str = "natural"           # cost to close: "natural" (short ask - long bid) | "mid"
    max_new_per_day: int = 1
    contracts: int = 1
    earnings: dict[str, list[date]] | None = None   # None -> single stocks are never traded (ETFs are)


@dataclass(frozen=True)
class Trade:
    symbol: str
    spread_type: str
    short_strike: float
    long_strike: float
    open_date: date
    close_date: date
    expiry: date
    dte_at_entry: int
    credit: float
    exit_debit: float
    exit_reason: str
    contracts: int
    gross_pnl: float
    costs: float
    net_pnl: float
    max_loss: float
    entry_short_delta: float
    iv_hv: float | None

    @property
    def weekday(self) -> int:
        return self.open_date.weekday()

    @property
    def dte_bucket(self) -> str:
        return dte_bucket(self.dte_at_entry)

    @property
    def held_days(self) -> int:
        return (self.close_date - self.open_date).days


@dataclass
class _Open:
    sig: SpreadSignal
    open_date: date
    iv_hv: float | None
    last_debit: float
    last_und: float

    def as_position(self, contracts: int) -> SpreadPosition:
        s = self.sig
        return SpreadPosition(
            s.spread_type, s.symbol, s.expiry, s.short_strike, s.long_strike, contracts, s.credit, self.last_debit,
            underlying_price=self.last_und, short_delta=s.short_delta, entry_short_delta=s.short_delta,
            initial_contracts=contracts, long_delta=s.long_delta)


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[date, float]]   # realized + open mark-to-market (before commissions on open trades)
    still_open: int
    first_date: date
    last_date: date
    days: int
    skipped: dict[str, int]
    notes: list[str]
    config: BacktestConfig


def _close_debit(short, long, pricing: str, width: float) -> float:
    raw = (short.ask - long.bid) if pricing == "natural" else (short.mid - long.mid)
    return min(max(raw, 0.0), width)


def _intrinsic(sig: SpreadSignal, und: float) -> float:
    if sig.spread_type is SpreadType.BULL_PUT:
        v = max(sig.short_strike - und, 0.0) - max(sig.long_strike - und, 0.0)
    else:
        v = max(und - sig.short_strike, 0.0) - max(und - sig.long_strike, 0.0)
    return min(max(v, 0.0), sig.width)


def _earnings_clear(symbol: str, d: date, expiry: date, earnings: dict[str, list[date]] | None) -> bool:
    """Same rule as the live gate: only a known next date strictly after expiry (or an ETF) is clear."""
    if symbol in DEFAULT_NO_EARNINGS:
        return True
    dates = (earnings or {}).get(symbol)
    if not dates:
        return False
    nxt = next((e for e in dates if e >= d), None)
    if nxt is None:
        return False   # the supplied history ends: later dates are unknown, so block
    info = EarningsInfo(symbol, nxt, EarningsStatus.CONFIRMED)
    return classify_earnings(info, expiry, d) is EarningsCheck.CLEAR


def run_backtest(store: ChainStore, cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    rm = RiskManager(cfg.limits)
    dates = store.dates
    open_: list[_Open] = []
    trades: list[Trade] = []
    curve: list[tuple[date, float]] = []
    realized = 0.0
    skipped: dict[str, int] = {}
    notes = list(store.notes)
    if cfg.scanner.pricing != "natural":
        notes.append("scanner pricing is not 'natural': entry credits assume you fill at the mid (optimistic)")
    if cfg.earnings is None:
        notes.append("no earnings history supplied: single stocks were NOT traded (only ETFs)")

    def bump(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def close(pos: _Open, d: date, debit: float, reason: str) -> None:
        nonlocal realized
        s = pos.sig
        gross = (s.credit - debit) * CONTRACT_MULTIPLIER * cfg.contracts
        fills = 4 * cfg.contracts                      # open and close, two legs each
        costs = cfg.commission_per_contract * fills + cfg.slippage_per_leg * CONTRACT_MULTIPLIER * fills
        net = gross - costs
        realized += net
        trades.append(Trade(s.symbol, s.spread_type.value, s.short_strike, s.long_strike, pos.open_date, d, s.expiry,
                            s.dte, s.credit, round(debit, 4), reason, cfg.contracts, round(gross, 2), round(costs, 2),
                            round(net, 2), s.max_loss * cfg.contracts, s.short_delta, pos.iv_hv))

    for d in dates:
        # 1) manage open positions with today's quotes (a position is never closed on the day it was opened)
        still: list[_Open] = []
        for pos in open_:
            s = pos.sig
            short = store.quote(d, s.symbol, s.expiry, s.spread_type.right, s.short_strike)
            long = store.quote(d, s.symbol, s.expiry, s.spread_type.right, s.long_strike)
            und = store.prices.get(s.symbol, {}).get(d)
            if und is not None:
                pos.last_und = und
            if d >= s.expiry and und is not None:
                close(pos, d, _intrinsic(s, und), "EXPIRY")
                continue
            if short is None or long is None:
                bump("position_quote_gap")          # no quote today: keep the last mark, try again tomorrow
                still.append(pos)
                continue
            pos.last_debit = _close_debit(short, long, cfg.exit_pricing, s.width)
            action = rm.evaluate_exit(pos.as_position(cfg.contracts), d)
            if action in (ExitAction.TAKE_PROFIT, ExitAction.STOP_LOSS, ExitAction.CLOSE_DTE):
                close(pos, d, pos.last_debit, action.value)
            else:
                still.append(pos)
        open_ = still

        # 2) new entries from today's chains
        opened_today = 0
        held = {(p.sig.symbol, p.sig.spread_type) for p in open_}
        for symbol in store.symbols:
            chain = store.chain(d, symbol)
            if not chain or opened_today >= cfg.max_new_per_day:
                continue
            closes = store.closes(symbol, d)
            stats = build_stats(symbol, chain[0].underlying_price, closes, chain, d, cfg.scanner.min_dte,
                                cfg.scanner.max_dte)
            signals = scan_all(chain, replace(cfg.scanner, top_n_per_symbol=50), d, stats)
            for sig in sorted(signals, key=rank_key(cfg.scanner), reverse=True):
                if opened_today >= cfg.max_new_per_day:
                    break
                if (symbol, sig.spread_type) in held:
                    bump("already_holding")
                    continue
                if not _earnings_clear(symbol, d, sig.expiry, cfg.earnings):
                    bump("earnings")
                    continue
                book = [p.as_position(cfg.contracts) for p in open_]
                if not rm.check_signal(sig, book, cfg.equity, cfg.contracts).approved:
                    bump("risk_limits")
                    continue
                open_.append(_Open(sig, d, stats.iv_hv_ratio, sig.credit, sig.underlying_price))
                held.add((symbol, sig.spread_type))
                opened_today += 1
                break    # one entry per symbol per day

        unreal = sum((p.sig.credit - p.last_debit) * CONTRACT_MULTIPLIER * cfg.contracts for p in open_)
        curve.append((d, realized + unreal))

    return BacktestResult(trades, curve, len(open_), dates[0], dates[-1], len(dates), skipped, notes, cfg)
