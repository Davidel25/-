"""SYNTHETIC option chains for exercising the backtest pipeline. NOT market data.

Prices come from Black-Scholes on a simulated underlying, with an implied-vol premium (``vrp``) you choose. A backtest
on this data therefore shows only what happens IF such a premium exists; it says nothing about whether it does. Its job
is to prove the machinery (entries, exits, costs, statistics) works end to end, and to let tests run offline."""
from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pandas as pd

from src.data.greeks import bs_delta, bs_price

RATE = 0.045


def trading_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate_frame(symbols: tuple[str, ...] = ("SPY",), days: int = 260, seed: int = 1, vrp: float = 0.10,
                   real_vol: float = 0.16, drift: float = 0.07, start: date = date(2024, 1, 2),
                   spot0: float = 500.0, min_dte: int = 7, max_dte: int = 75) -> pd.DataFrame:
    """Daily chains for each symbol: Friday expiries from ``min_dte`` to ``max_dte`` days out, +/-12% strikes."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for si, symbol in enumerate(symbols):
        spot, sigma, iv_base = spot0 * (1 + 0.1 * si), real_vol, real_vol * (1 + vrp)
        for d in trading_days(start, days):
            # daily vol follows a mean-reverting random walk; implied vol tracks it with a lag plus the premium
            sigma = max(0.06, sigma + 0.08 * (real_vol - sigma) + rng.gauss(0, 0.012))
            z = rng.gauss(0, 1) * math.sqrt(3 / 5) / math.sqrt(rng.gammavariate(2.5, 1 / 2.5))   # fat tails
            spot *= math.exp((drift - 0.5 * sigma**2) / 252 + sigma / math.sqrt(252) * z)
            iv_base += 0.15 * (sigma * (1 + vrp) - iv_base)
            step = 1.0 if spot < 250 else 5.0
            lo, hi = math.floor(spot * 0.88 / step) * step, math.ceil(spot * 1.12 / step) * step
            first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
            for w in range(0, 12):
                expiry = first_friday + timedelta(weeks=w)
                dte = (expiry - d).days
                if not min_dte <= dte <= max_dte:
                    continue
                t = dte / 365.0
                for k in [lo + i * step for i in range(int((hi - lo) / step) + 1)]:
                    for right in ("P", "C"):
                        skew = 0.15 * max(0.0, (spot - k) / spot) if right == "P" else 0.0
                        iv = iv_base * (1 + 0.03 * (dte / 30 - 1)) + skew     # mild contango plus put skew
                        mid = max(bs_price(spot, k, t, RATE, iv, right), 0.01)
                        half = max(0.01, mid * 0.005)
                        rows.append({
                            "date": d.isoformat(), "symbol": symbol, "expiry": expiry.isoformat(), "strike": k,
                            "right": right, "bid": round(max(mid - half, 0.0), 2), "ask": round(mid + half, 2),
                            "underlying_price": round(spot, 2),
                            "delta": round(bs_delta(spot, k, t, RATE, iv, right), 4),
                            "iv": round(iv, 4), "open_interest": 2000, "volume": 500})
    return pd.DataFrame(rows)
