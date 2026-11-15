"""SYNTHETIC option chains for exercising the backtest pipeline. NOT market data.

Prices come from Black-Scholes on a simulated underlying, with an implied-vol premium (``vrp``) you choose. A backtest
on this data therefore shows only what happens IF such a premium exists; it says nothing about whether it does. Its job
is to prove the machinery (entries, exits, costs, statistics) works end to end, to let tests run offline, and to let
you ask "what would this rule do in a world with property X" where X is stated explicitly in a :class:`SymbolProfile`.

A profile controls: realized vol, the implied-vol premium over it, the quoted bid/ask width, non-earnings news gaps and
a quarterly earnings jump (options that span an earnings date are priced for it, with the same premium)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

from src.data.greeks import bs_delta, bs_price

RATE = 0.045


@dataclass(frozen=True)
class SymbolProfile:
    real_vol: float = 0.16               # long-run annualised realized vol
    vrp: float = 0.10                    # implied vol = realized x (1 + vrp)
    drift: float = 0.07
    spot0: float = 500.0
    half_spread_pct: float = 0.005       # half the quoted bid/ask width, as a fraction of mid
    min_half_spread: float = 0.01        # dollars
    jump_per_year: float = 0.0           # non-earnings news gaps per year
    jump_sd: float = 0.0                 # sd of a news gap (log return)
    earnings_every_days: int | None = None   # quarterly report cadence in calendar days (None = no earnings)
    earnings_jump_sd: float = 0.0        # sd of the earnings-day gap (log return)
    price_jumps_in_iv: bool = False      # True: IV includes news-gap variance, so ``vrp`` is the EFFECTIVE premium
    open_interest: int = 2000


def trading_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def earnings_schedule(profile: SymbolProfile, start: date, days: int, offset: int = 0) -> list[date]:
    """Report dates every ``earnings_every_days`` (weekdays only), running one cycle past the end of the data so a
    'next known date' always exists for the backtest's earnings gate."""
    if not profile.earnings_every_days:
        return []
    end = trading_days(start, days)[-1] + timedelta(days=profile.earnings_every_days + 7)
    out, d = [], start + timedelta(days=20 + offset)
    while d <= end:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(d)
        d += timedelta(days=profile.earnings_every_days)
    return out


def generate_universe(profiles: dict[str, SymbolProfile], days: int = 260, seed: int = 1,
                      start: date = date(2024, 1, 2), min_dte: int = 7, max_dte: int = 55,
                      ) -> tuple[pd.DataFrame, dict[str, list[date]]]:
    """Daily chains (Friday expiries from ``min_dte`` to ``max_dte`` days out, +/-12% strikes) and each symbol's
    earnings dates. Deterministic for a given seed."""
    rng = random.Random(seed)
    rows: list[dict] = []
    earnings: dict[str, list[date]] = {}
    for si, (symbol, p) in enumerate(profiles.items()):
        sched = earnings_schedule(p, start, days, offset=7 * si)
        earnings[symbol] = sched
        react = {next(d for d in (e + timedelta(days=k) for k in range(0, 5)) if d.weekday() < 5) for e in sched}
        spot, sigma, iv_base = p.spot0, p.real_vol, p.real_vol * (1 + p.vrp)
        for d in trading_days(start, days):
            # daily vol follows a mean-reverting random walk; implied vol tracks it with a lag plus the premium
            sigma = max(0.06, sigma + 0.08 * (p.real_vol - sigma) + rng.gauss(0, 0.012))
            z = rng.gauss(0, 1) * math.sqrt(3 / 5) / math.sqrt(rng.gammavariate(2.5, 1 / 2.5))   # fat tails
            ret = (p.drift - 0.5 * sigma**2) / 252 + sigma / math.sqrt(252) * z
            if p.jump_per_year > 0 and rng.random() < p.jump_per_year / 252:
                ret += rng.gauss(0, p.jump_sd)                                   # a news gap that is not earnings
            if d in react and p.earnings_jump_sd > 0:
                ret += rng.gauss(0, p.earnings_jump_sd)                          # the earnings reaction
            spot *= math.exp(ret)
            sigma_iv = math.sqrt(sigma**2 + p.jump_per_year * p.jump_sd**2) if p.price_jumps_in_iv else sigma
            iv_base += 0.15 * (sigma_iv * (1 + p.vrp) - iv_base)
            step = 1.0 if spot < 250 else 5.0
            lo, hi = math.floor(spot * 0.88 / step) * step, math.ceil(spot * 1.12 / step) * step
            nxt_earn = next((e for e in sched if e >= d), None)
            first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
            for w in range(0, 12):
                expiry = first_friday + timedelta(weeks=w)
                dte = (expiry - d).days
                if not min_dte <= dte <= max_dte:
                    continue
                t = dte / 365.0
                spans_earnings = nxt_earn is not None and p.earnings_jump_sd > 0 and nxt_earn <= expiry
                for k in [lo + i * step for i in range(int((hi - lo) / step) + 1)]:
                    for right in ("P", "C"):
                        skew = 0.15 * max(0.0, (spot - k) / spot) if right == "P" else 0.0
                        iv = iv_base * (1 + 0.03 * (dte / 30 - 1)) + skew        # mild contango plus put skew
                        if spans_earnings:                         # priced for the report, premium included
                            iv = math.sqrt(iv**2 + (1 + p.vrp) * p.earnings_jump_sd**2 / t)
                        mid = max(bs_price(spot, k, t, RATE, iv, right), 0.01)
                        half = max(p.min_half_spread, mid * p.half_spread_pct)
                        rows.append({
                            "date": d.isoformat(), "symbol": symbol, "expiry": expiry.isoformat(), "strike": k,
                            "right": right, "bid": round(max(mid - half, 0.0), 2), "ask": round(mid + half, 2),
                            "underlying_price": round(spot, 2),
                            "delta": round(bs_delta(spot, k, t, RATE, iv, right), 4),
                            "iv": round(iv, 4), "open_interest": p.open_interest, "volume": 500})
    return pd.DataFrame(rows), earnings


def generate_frame(symbols: tuple[str, ...] = ("SPY",), days: int = 260, seed: int = 1, vrp: float = 0.10,
                   real_vol: float = 0.16, drift: float = 0.07, start: date = date(2024, 1, 2),
                   spot0: float = 500.0, min_dte: int = 7, max_dte: int = 75) -> pd.DataFrame:
    """Daily chains for each symbol with identical parameters (kept for simple uses and existing tests)."""
    base = SymbolProfile(real_vol=real_vol, vrp=vrp, drift=drift, spot0=spot0)
    profiles = {s: replace(base, spot0=spot0 * (1 + 0.1 * i)) for i, s in enumerate(symbols)}
    return generate_universe(profiles, days, seed, start, min_dte, max_dte)[0]
