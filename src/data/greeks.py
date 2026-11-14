"""Black-Scholes pricing and delta (no scipy dependency). Used as a fallback when the
broker does not supply model Greeks and as the pricing engine for the mock broker."""
from __future__ import annotations

import math


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(spot: float, strike: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    return d1, d1 - sigma * math.sqrt(t)


def bs_price(spot: float, strike: float, t: float, r: float, sigma: float, right: str) -> float:
    """European option price. ``t`` in years. Falls back to intrinsic value at expiry."""
    right = right.upper()
    if t <= 0 or sigma <= 0:
        return max(0.0, spot - strike) if right == "C" else max(0.0, strike - spot)
    d1, d2 = _d1_d2(spot, strike, t, r, sigma)
    if right == "C":
        return spot * _cdf(d1) - strike * math.exp(-r * t) * _cdf(d2)
    return strike * math.exp(-r * t) * _cdf(-d2) - spot * _cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, r: float, sigma: float, right: str) -> float:
    """Signed delta: calls in [0, 1], puts in [-1, 0]."""
    right = right.upper()
    if t <= 0 or sigma <= 0:
        itm = spot > strike if right == "C" else spot < strike
        return (1.0 if itm else 0.0) * (1 if right == "C" else -1)
    d1, _ = _d1_d2(spot, strike, t, r, sigma)
    return _cdf(d1) if right == "C" else _cdf(d1) - 1.0
