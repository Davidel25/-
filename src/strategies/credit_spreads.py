"""Scanner engine for Bull Put and Bear Call credit spreads.

Both strategies share one algorithm, parameterised by option right:
  Bull Put : sell OTM put  (strike < spot), buy put  ``width`` below   -> profits if price stays above short strike
  Bear Call: sell OTM call (strike > spot), buy call ``width`` above   -> profits if price stays below short strike

Filters: short |delta| window, DTE, liquidity, minimum credit and ROC, max risk per spread.
ROC = max_profit / max_loss = credit / (width - credit)."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from src.data.stats import UnderlyingStats
from src.models import CONTRACT_MULTIPLIER, OptionQuote, SpreadSignal, SpreadType


@dataclass
class ScannerConfig:
    min_dte: int = 25
    max_dte: int = 50
    min_short_delta: float = 0.10   # absolute values
    max_short_delta: float = 0.25
    widths: tuple[float, ...] = (1.0, 2.0, 5.0)
    min_credit: float = 0.15        # per share
    min_roc: float = 0.15           # 15% of capital at risk
    max_risk_per_spread: float = 1_000.0  # $ per 1 lot
    min_open_interest: int = 100
    max_bid_ask_pct: float = 0.15   # of mid, per leg
    pricing: str = "mid"            # "mid" | "natural" (short bid - long ask, conservative)
    top_n_per_symbol: int = 10
    # --- edge / friction filters (all off by default so the base scanner stays permissive; see disciplined()) ---
    max_bid_ask_abs: float | None = None   # per-leg bid/ask width cap in $ (e.g. 0.05)
    min_iv_hv_ratio: float | None = None   # require ATM IV / 30d HV >= this (needs UnderlyingStats; fails closed)
    require_trend_alignment: bool = False  # Bull Put only above SMA50, Bear Call only below (fails closed)
    min_term_slope: float | None = None    # IV(back)/IV(front) - 1 must be >= this (0.01 = 1% contango); fails closed
    max_put_skew: float | None = None      # Bull Put only: 25d put IV - 25d call IV must be <= this (0.10 = 10 vol pts)
    # --- entry timing / expiry preferences (see the trade-log breakdown to validate them with your own data) ---
    allowed_weekdays: tuple[int, ...] | None = None  # entry days, 0 = Monday ... 4 = Friday; None = any day
    preferred_dte: tuple[int, int] | None = None     # inclusive DTE bucket ranked first (e.g. (35, 45))

    @classmethod
    def disciplined(cls, **overrides) -> ScannerConfig:
        """Preset implementing the friction/edge rules from the strategy review. Individual fields can be
        overridden, e.g. ``ScannerConfig.disciplined(max_bid_ask_abs=0.02)``.

        Short delta 0.12-0.15 (was 0.10-0.25 / 0.20-0.25) came out of the synthetic strategy-adjustment simulations
        (experiments/REPORT.md): consistently better Sharpe and smaller losses/drawdowns than 0.20-0.25. That band
        collects less premium (about 11% of width versus 20%), so the credit/ROC floors drop with it: the former
        0.60 / 20% floors return NO trades at 0.12-0.15 delta on 5-wide spreads. Synthetic evidence: re-test on real
        option history before relying on it."""
        base = dict(min_dte=30, max_dte=50, preferred_dte=(35, 45), min_short_delta=0.12, max_short_delta=0.15,
                    min_credit=0.45, min_roc=0.12, pricing="natural", max_bid_ask_pct=0.05, max_bid_ask_abs=0.05,
                    min_iv_hv_ratio=1.1, require_trend_alignment=True, min_term_slope=0.01, max_put_skew=0.10)
        return cls(**{**base, **overrides})


def _credit(short: OptionQuote, long: OptionQuote, pricing: str) -> float:
    if pricing == "natural":
        return short.bid - long.ask
    return short.mid - long.mid


def _liquid(q: OptionQuote, cfg: ScannerConfig) -> bool:
    if cfg.max_bid_ask_abs is not None and (q.ask - q.bid) > cfg.max_bid_ask_abs + 1e-9:
        return False
    return q.open_interest >= cfg.min_open_interest and q.spread_pct <= cfg.max_bid_ask_pct


def underlying_gate(stats: UnderlyingStats | None, spread_type: SpreadType, cfg: ScannerConfig) -> list[str]:
    """Reasons an underlying is NOT eligible for ``spread_type`` (empty list = eligible). Missing data fails closed."""
    reasons: list[str] = []
    if cfg.min_iv_hv_ratio is not None:
        ratio = stats.iv_hv_ratio if stats else None
        if ratio is None:
            reasons.append("IV/HV unavailable")
        elif ratio < cfg.min_iv_hv_ratio:
            reasons.append(f"IV/HV {ratio:.2f} < {cfg.min_iv_hv_ratio:g}")
    if cfg.require_trend_alignment:
        if stats is None or stats.sma50 is None:
            reasons.append("50-day SMA unavailable")
        elif spread_type is SpreadType.BULL_PUT and stats.price <= stats.sma50:
            reasons.append(f"price {stats.price:g} not above SMA50 {stats.sma50:.2f} (Bull Put)")
        elif spread_type is SpreadType.BEAR_CALL and stats.price >= stats.sma50:
            reasons.append(f"price {stats.price:g} not below SMA50 {stats.sma50:.2f} (Bear Call)")
    if cfg.min_term_slope is not None:
        slope = stats.term_slope if stats else None
        if slope is None:
            reasons.append("term structure unavailable")
        elif slope < cfg.min_term_slope:
            shape = "inverted" if slope < 0 else "flat"
            reasons.append(f"term structure {shape} (slope {slope:+.1%} < {cfg.min_term_slope:+.1%})")
    if cfg.max_put_skew is not None and spread_type is SpreadType.BULL_PUT:
        skew = stats.skew if stats else None
        if skew is None:
            reasons.append("put skew unavailable")
        elif skew > cfg.max_put_skew:
            reasons.append(f"put skew {skew:.1%} > {cfg.max_put_skew:.1%} (crash hedging bid)")
    return reasons


def schedule_gate(cfg: ScannerConfig, today: date) -> list[str]:
    """Reasons new entries are not allowed on ``today`` (weekday filter)."""
    if cfg.allowed_weekdays is not None and today.weekday() not in cfg.allowed_weekdays:
        names = "MTWTFSS"
        return [f"entries allowed on {''.join(names[d] for d in sorted(cfg.allowed_weekdays))} only "
                f"(today is {names[today.weekday()]})"]
    return []


def in_preferred_dte(dte: int, cfg: ScannerConfig) -> bool:
    return cfg.preferred_dte is not None and cfg.preferred_dte[0] <= dte <= cfg.preferred_dte[1]


def rank_key(cfg: ScannerConfig):
    """Sort key (descending): signals inside the preferred DTE bucket first, then ROC, then PoP."""
    return lambda s: (in_preferred_dte(s.dte, cfg), s.roc, s.pop)


def scan_spreads(
    quotes: Iterable[OptionQuote],
    spread_type: SpreadType,
    cfg: ScannerConfig | None = None,
    today: date | None = None,
    stats: UnderlyingStats | None = None,
) -> list[SpreadSignal]:
    """Scan one underlying's chain. Returns signals sorted by ROC (desc). If the IV/HV or trend filter is
    enabled in ``cfg`` and ``stats`` do not satisfy it (or are missing), nothing is returned."""
    cfg = cfg or ScannerConfig()
    today = today or date.today()
    if schedule_gate(cfg, today) or underlying_gate(stats, spread_type, cfg):
        return []
    right = spread_type.right
    bull = spread_type is SpreadType.BULL_PUT

    by_exp: dict[date, dict[float, OptionQuote]] = defaultdict(dict)
    for q in quotes:
        if q.right == right and cfg.min_dte <= q.dte(today) <= cfg.max_dte:
            by_exp[q.expiry][q.strike] = q

    signals: list[SpreadSignal] = []
    for expiry, strikes in by_exp.items():
        for k, short in strikes.items():
            spot = short.underlying_price
            otm = k < spot if bull else k > spot
            if not otm or short.delta is None or not _liquid(short, cfg):
                continue
            if not cfg.min_short_delta <= abs(short.delta) <= cfg.max_short_delta:
                continue
            for width in cfg.widths:
                long = strikes.get(k - width if bull else k + width)
                if long is None or not _liquid(long, cfg):
                    continue
                credit = round(_credit(short, long, cfg.pricing), 2)
                max_loss = (width - credit) * CONTRACT_MULTIPLIER
                if credit < cfg.min_credit or credit >= width or max_loss > cfg.max_risk_per_spread:
                    continue
                roc = credit / (width - credit)
                if roc < cfg.min_roc:
                    continue
                signals.append(SpreadSignal(
                    spread_type=spread_type, symbol=short.symbol, expiry=expiry,
                    dte=(expiry - today).days, underlying_price=spot,
                    short_strike=k, long_strike=long.strike, width=width, credit=credit,
                    max_profit=round(credit * CONTRACT_MULTIPLIER, 2),
                    max_loss=round(max_loss, 2), roc=round(roc, 4),
                    short_delta=round(abs(short.delta), 4),
                    pop=round(1 - abs(short.delta), 4),
                    breakeven=round(k - credit if bull else k + credit, 2),
                    otm_pct=round(abs(spot - k) / spot, 4),
                    long_delta=round(abs(long.delta), 4) if long.delta is not None else None,
                ))
    signals.sort(key=rank_key(cfg), reverse=True)
    return signals[: cfg.top_n_per_symbol]


def scan_bull_put(quotes: Iterable[OptionQuote], cfg: ScannerConfig | None = None,
                  today: date | None = None, stats: UnderlyingStats | None = None) -> list[SpreadSignal]:
    return scan_spreads(quotes, SpreadType.BULL_PUT, cfg, today, stats)


def scan_bear_call(quotes: Iterable[OptionQuote], cfg: ScannerConfig | None = None,
                   today: date | None = None, stats: UnderlyingStats | None = None) -> list[SpreadSignal]:
    return scan_spreads(quotes, SpreadType.BEAR_CALL, cfg, today, stats)


def scan_all(quotes: list[OptionQuote], cfg: ScannerConfig | None = None,
             today: date | None = None, stats: UnderlyingStats | None = None) -> list[SpreadSignal]:
    out = scan_bull_put(quotes, cfg, today, stats) + scan_bear_call(quotes, cfg, today, stats)
    return sorted(out, key=rank_key(cfg or ScannerConfig()), reverse=True)
