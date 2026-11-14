"""Underlying statistics used by the scanner's edge filters: historical vol, 50-day SMA and ATM implied vol.

Pure functions; all inputs are plain lists / quotes so they are trivially testable."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from src.models import OptionQuote

TRADING_DAYS = 252


def sma(closes: list[float], n: int = 50) -> float | None:
    """Simple moving average of the last ``n`` closes; None if there is not enough history."""
    return sum(closes[-n:]) / n if len(closes) >= n else None


def historical_vol(closes: list[float], n: int = 30) -> float | None:
    """Annualised close-to-close volatility (sample std of ``n`` log returns). None if history is too short
    or contains a non-positive price."""
    if len(closes) < n + 1:
        return None
    window = closes[-(n + 1):]
    if any(c <= 0 for c in window):
        return None
    rets = [math.log(b / a) for a, b in zip(window, window[1:], strict=False)]
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def atm_iv(chain: list[OptionQuote], today: date, min_dte: int = 25, max_dte: int = 50,
           band_pct: float = 0.03) -> float | None:
    """Average IV of the near-the-money options (within ``band_pct`` of spot) in the DTE window, using the
    expiry nearest 35 DTE. None if the broker supplied no usable IVs (so the filter fails closed)."""
    by_exp: dict[date, list[OptionQuote]] = {}
    for q in chain:
        if q.iv and q.iv > 0 and min_dte <= q.dte(today) <= max_dte \
                and abs(q.strike - q.underlying_price) <= q.underlying_price * band_pct:
            by_exp.setdefault(q.expiry, []).append(q)
    if not by_exp:
        return None
    expiry = min(by_exp, key=lambda e: abs((e - today).days - 35))
    ivs = [q.iv for q in by_exp[expiry] if q.iv]
    return sum(ivs) / len(ivs)


def _atm_iv_by_expiry(chain: list[OptionQuote], today: date, band_pct: float) -> dict[date, float]:
    by: dict[date, list[float]] = {}
    for q in chain:
        near = abs(q.strike - q.underlying_price) <= q.underlying_price * band_pct
        if q.iv and q.iv > 0 and q.dte(today) > 0 and near:
            by.setdefault(q.expiry, []).append(q.iv)
    return {e: sum(v) / len(v) for e, v in by.items()}


def term_slope(chain: list[OptionQuote], today: date, min_gap_days: int = 14, band_pct: float = 0.03) -> float | None:
    """ATM IV term slope as ``IV(back expiry) / IV(front expiry) - 1`` over the expiries present in the chain.
    Positive = contango (normal, favourable for premium sellers); ~0 = flat; negative = inverted (stress / event
    risk). None if fewer than two expiries have an IV, or they are less than ``min_gap_days`` apart (fail closed)."""
    iv = _atm_iv_by_expiry(chain, today, band_pct)
    if len(iv) < 2:
        return None
    front, back = min(iv), max(iv)
    if (back - front).days < min_gap_days:
        return None
    return iv[back] / iv[front] - 1.0


def skew_25d(chain: list[OptionQuote], today: date, min_dte: int = 25, max_dte: int = 50, target: float = 0.25,
             tol: float = 0.07) -> float | None:
    """Put-minus-call skew in IV points (0.05 = 5 vol points): IV of the ~25-delta put minus the ~25-delta call in
    the expiry nearest 35 DTE. High values mean the market is paying up for crash protection."""
    window = [q for q in chain if q.iv and q.delta is not None and min_dte <= q.dte(today) <= max_dte]
    if not window:
        return None
    expiry = min({q.expiry for q in window}, key=lambda e: abs((e - today).days - 35))

    def pick(right: str) -> OptionQuote | None:
        c = [q for q in window if q.expiry == expiry and q.right == right and abs(abs(q.delta) - target) <= tol]
        return min(c, key=lambda q: abs(abs(q.delta) - target)) if c else None

    put, call = pick("P"), pick("C")
    return (put.iv - call.iv) if put and call else None


@dataclass(frozen=True)
class UnderlyingStats:
    symbol: str
    price: float
    hv: float | None = None      # 30-day annualised historical vol
    sma50: float | None = None
    iv: float | None = None      # ATM implied vol (annualised)
    term_slope: float | None = None  # IV(back)/IV(front) - 1; > 0 is contango
    skew: float | None = None        # 25-delta put IV minus 25-delta call IV (vol points as a fraction)

    @property
    def iv_hv_ratio(self) -> float | None:
        return self.iv / self.hv if self.iv and self.hv else None


def build_stats(symbol: str, price: float, closes: list[float], chain: list[OptionQuote], today: date,
                min_dte: int = 25, max_dte: int = 50) -> UnderlyingStats:
    return UnderlyingStats(symbol, price, historical_vol(closes), sma(closes, 50),
                           atm_iv(chain, today, min_dte, max_dte), term_slope(chain, today),
                           skew_25d(chain, today, min_dte, max_dte))
