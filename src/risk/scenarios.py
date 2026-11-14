"""What-if scenario presets: reprice every open spread under an instantaneous market shock.

Method (pure, no I/O)
  * SPY moves by ``spot_move``; each underlying moves ``beta x spot_move`` (betas from ``src.risk.portfolio``).
  * Every leg's implied vol is shifted in parallel by ``iv_shift`` (absolute vol points: 0.05 = +5 points).
  * ``days`` of time may also pass. The spread is repriced with Black-Scholes before and after, and the move is
    mapped onto the broker's current mark: if the model says the spread has gone from v0 to v1 inside its 0..width
    range, the market debit moves the same fraction of ITS remaining range (mark..width when the spread gains value,
    mark..0 when it loses value). This anchors the result on the real mark (zero shock = zero change) and makes the
    extremes exact: a spread the model sees fully in the money is at max loss, one fully out of the money is at max
    profit, so results can never leave the spread's hard bounds.

This is a rough stress test, not a forecast. It assumes a parallel vol shift (no skew or term-structure change), an
instantaneous move (no gap-through-the-stop or liquidity effects), European exercise (no early assignment), and rough
static betas. Real crashes also move correlations and widen markets. A position whose IV is unknown is reported as
unavailable and EXCLUDED from the totals (the result is then flagged incomplete) rather than priced with a guess."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from src.data.greeks import bs_price
from src.models import CONTRACT_MULTIPLIER, SpreadPosition

RISK_FREE = 0.045
MIN_VOL = 0.01
EPS = 1e-6
SNAP = 1e-4   # normalised values this close to 0 or 1 are treated as exactly 0 or 1


@dataclass(frozen=True)
class ScenarioPreset:
    key: str
    label: str
    spot_move: float   # SPY move as a fraction (-0.03 = -3%)
    iv_shift: float    # parallel IV change in absolute vol points (0.05 = +5 points)
    days: int = 0      # calendar days that pass
    note: str = ""


PRESETS: tuple[ScenarioPreset, ...] = (
    ScenarioPreset("spy_down_3", "SPY -3%", -0.03, 0.0, 0, "Spot only, vol unchanged: understates a real sell-off"),
    ScenarioPreset("spy_down_3_iv5", "SPY -3% + IV +5 pts", -0.03, 0.05, 0,
                   "A typical sell-off: vol rises with the drop"),
    ScenarioPreset("spy_down_5", "SPY -5%", -0.05, 0.0, 0, "Spot only, vol unchanged"),
    ScenarioPreset("spy_up_3", "SPY +3%", 0.03, 0.0, 0, "Spot only: Bull Puts gain, Bear Calls lose"),
    ScenarioPreset("iv_up_10", "IV +10 pts", 0.0, 0.10, 0, "Vol shock with an unchanged index"),
    ScenarioPreset("crash_10", "SPY -10% + IV +15 pts", -0.10, 0.15, 0,
                   "Tail scenario: Bull Puts lose most of their max loss"),
    ScenarioPreset("week_passes", "5 days pass, no move", 0.0, 0.0, 5, "Pure time decay"),
)
PRESET_KEYS = tuple(p.key for p in PRESETS)


@dataclass(frozen=True)
class ScenarioPosition:
    key: str
    symbol: str
    spread_type: str
    pnl_change: float       # $ change versus now, clamped to the spread's bounds
    projected_pnl: float    # $ unrealized P&L after the shock
    at_max_loss: bool
    beta_known: bool


@dataclass(frozen=True)
class ScenarioResult:
    preset: ScenarioPreset
    positions: list[ScenarioPosition] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)   # position keys that could not be priced (missing IV/price)
    equity: float = 0.0

    @property
    def complete(self) -> bool:
        return not self.unavailable

    @property
    def total_change(self) -> float:
        return sum(p.pnl_change for p in self.positions)

    @property
    def projected_total_pnl(self) -> float:
        return sum(p.projected_pnl for p in self.positions)

    @property
    def pct_of_equity(self) -> float | None:
        return self.total_change / self.equity if self.equity > 0 else None

    @property
    def worst(self) -> ScenarioPosition | None:
        return min(self.positions, key=lambda p: p.pnl_change, default=None)

    @property
    def at_max_loss_count(self) -> int:
        return sum(p.at_max_loss for p in self.positions)

    def to_dict(self) -> dict:
        w = self.worst
        return {
            "key": self.preset.key, "label": self.preset.label, "note": self.preset.note,
            "spot_move": self.preset.spot_move, "iv_shift": self.preset.iv_shift, "days": self.preset.days,
            "total_change": round(self.total_change, 2), "projected_total_pnl": round(self.projected_total_pnl, 2),
            "pct_of_equity": self.pct_of_equity, "at_max_loss": self.at_max_loss_count, "complete": self.complete,
            "unavailable": self.unavailable,
            "worst": None if w is None else {"key": w.key, "change": round(w.pnl_change, 2)},
            "positions": [{"key": p.key, "symbol": p.symbol, "spread_type": p.spread_type,
                           "pnl_change": round(p.pnl_change, 2), "projected_pnl": round(p.projected_pnl, 2),
                           "at_max_loss": p.at_max_loss, "beta_known": p.beta_known} for p in self.positions],
        }


def _spread_fraction(pos: SpreadPosition, spot: float, years: float, iv_short: float, iv_long: float) -> float:
    """Model value of the spread as a fraction of its maximum possible value (0 = worthless, 1 = fully in the
    money). The maximum is the discounted width, so a deep-ITM spread is exactly 1 despite the interest-rate term."""
    right = pos.spread_type.right
    v = (bs_price(spot, pos.short_strike, years, RISK_FREE, iv_short, right)
         - bs_price(spot, pos.long_strike, years, RISK_FREE, iv_long, right))
    u = min(max(v / (pos.width * math.exp(-RISK_FREE * years)), 0.0), 1.0)
    return 0.0 if u < SNAP else 1.0 if u > 1.0 - SNAP else u


def reprice(pos: SpreadPosition, preset: ScenarioPreset, betas: dict[str, float],
            today: date | None = None) -> ScenarioPosition | None:
    """Scenario outcome for one spread, or None if its price or either leg's IV is unknown."""
    if pos.underlying_price is None or not pos.short_iv or not pos.long_iv:
        return None
    beta = betas.get(pos.symbol.upper())
    spot0 = pos.underlying_price
    spot1 = max(spot0 * (1.0 + (beta or 1.0) * preset.spot_move), 0.01)
    dte = pos.dte(today)
    t0, t1 = max(dte, 0) / 365.0, max(dte - preset.days, 0) / 365.0
    u0 = _spread_fraction(pos, spot0, t0, pos.short_iv, pos.long_iv)
    u1 = _spread_fraction(pos, spot1, t1, max(pos.short_iv + preset.iv_shift, MIN_VOL),
                          max(pos.long_iv + preset.iv_shift, MIN_VOL))
    width = pos.width
    mark = min(max(pos.current_debit, 0.0), width)               # cost to close now, per share
    if u1 >= u0:
        frac = (u1 - u0) / (1.0 - u0) if 1.0 - u0 > EPS else 0.0
        debit = mark + frac * (width - mark)
    else:
        debit = mark - (u0 - u1) / u0 * mark
    debit = min(max(debit, 0.0), width)
    projected = (pos.entry_credit - debit) * CONTRACT_MULTIPLIER * pos.contracts
    return ScenarioPosition(pos.key, pos.symbol, pos.spread_type.value, projected - pos.unrealized_pnl, projected,
                            debit >= width - 1e-3, beta is not None)


def run_scenario(positions: list[SpreadPosition], preset: ScenarioPreset, betas: dict[str, float], equity: float,
                 today: date | None = None) -> ScenarioResult:
    done, missing = [], []
    for pos in positions:
        r = reprice(pos, preset, betas, today)
        (done if r else missing).append(r or pos.key)
    return ScenarioResult(preset, done, missing, equity)


def run_presets(positions: list[SpreadPosition], betas: dict[str, float], equity: float,
                today: date | None = None) -> list[ScenarioResult]:
    return [run_scenario(positions, p, betas, equity, today) for p in PRESETS]
