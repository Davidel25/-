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


def _credit(short: OptionQuote, long: OptionQuote, pricing: str) -> float:
    if pricing == "natural":
        return short.bid - long.ask
    return short.mid - long.mid


def _liquid(q: OptionQuote, cfg: ScannerConfig) -> bool:
    return q.open_interest >= cfg.min_open_interest and q.spread_pct <= cfg.max_bid_ask_pct


def scan_spreads(
    quotes: Iterable[OptionQuote],
    spread_type: SpreadType,
    cfg: ScannerConfig | None = None,
    today: date | None = None,
) -> list[SpreadSignal]:
    """Scan one underlying's chain. Returns signals sorted by ROC (desc)."""
    cfg = cfg or ScannerConfig()
    today = today or date.today()
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
                ))
    signals.sort(key=lambda s: (s.roc, s.pop), reverse=True)
    return signals[: cfg.top_n_per_symbol]


def scan_bull_put(quotes: Iterable[OptionQuote], cfg: ScannerConfig | None = None,
                  today: date | None = None) -> list[SpreadSignal]:
    return scan_spreads(quotes, SpreadType.BULL_PUT, cfg, today)


def scan_bear_call(quotes: Iterable[OptionQuote], cfg: ScannerConfig | None = None,
                   today: date | None = None) -> list[SpreadSignal]:
    return scan_spreads(quotes, SpreadType.BEAR_CALL, cfg, today)


def scan_all(quotes: list[OptionQuote], cfg: ScannerConfig | None = None,
             today: date | None = None) -> list[SpreadSignal]:
    out = scan_bull_put(quotes, cfg, today) + scan_bear_call(quotes, cfg, today)
    return sorted(out, key=lambda s: s.roc, reverse=True)
