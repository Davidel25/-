"""Beta-weighted portfolio delta: how many dollars of SPY-equivalent directional exposure the open spreads carry.

Why it matters: three Bull Puts on SPY, QQQ and IWM each pass a per-trade risk check, yet together they are one
large long-the-market bet. Beta-weighting puts every position on a common scale so a single cap can see that.

Definitions (all pure, no I/O)
  * Net delta of a vertical, per share: |delta short| - |delta long|, positive for a Bull Put (long the market),
    negative for a Bear Call. Times 100 x contracts gives share-equivalents.
  * Beta-weighted dollar delta = net delta shares x underlying price x beta to SPY.
  * SPY-equivalent shares = beta-weighted dollars / SPY price.
  * Utilisation = |net beta-weighted dollar delta| / equity versus ``RiskLimits.max_net_beta_delta_pct``.

Honest limits: betas are rough, static inputs (defaults exist only for a few ETFs; anything else is assumed 1.0 and
flagged); deltas are broker model deltas for a single moment; this ignores gamma, vega and the correlation breakdown
that happens in a crash. It is a guard rail against concentration, not a risk model."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models import CONTRACT_MULTIPLIER, SpreadPosition, SpreadSignal, SpreadType

# Rough long-run betas to SPY. Edit via the BETAS setting; unknown symbols are assumed 1.0 and flagged.
DEFAULT_BETAS: dict[str, float] = {"SPY": 1.0, "QQQ": 1.15, "IWM": 1.2, "DIA": 0.95}


def parse_betas(raw: str) -> dict[str, float]:
    """``"QQQ=1.15,IWM=1.2"`` -> {"QQQ": 1.15, "IWM": 1.2}. Malformed or non-positive entries are ignored."""
    out: dict[str, float] = {}
    for part in raw.split(","):
        sym, _, val = part.partition("=")
        try:
            beta = float(val)
        except ValueError:
            continue
        if sym.strip() and beta > 0:
            out[sym.strip().upper()] = beta
    return out


def net_delta_shares(spread_type: SpreadType, short_delta: float, long_delta: float | None,
                     contracts: int = 1) -> tuple[float, bool]:
    """Share-equivalent net delta of a vertical and whether it is exact. If the long leg's delta is unknown the
    short leg alone is used, which OVERSTATES exposure (conservative) and is flagged inexact."""
    exact = long_delta is not None
    net = abs(short_delta) - (abs(long_delta) if long_delta is not None else 0.0)
    sign = 1.0 if spread_type is SpreadType.BULL_PUT else -1.0
    return sign * net * CONTRACT_MULTIPLIER * contracts, exact


@dataclass(frozen=True)
class Exposure:
    key: str
    symbol: str
    net_delta_shares: float
    beta: float
    beta_known: bool
    exact: bool            # False when the long leg's delta was unavailable (conservative estimate)
    dollar_delta: float    # beta-weighted, in dollars


def position_exposure(pos: SpreadPosition, betas: dict[str, float]) -> Exposure | None:
    """None if the position lacks the price or short delta needed (the caller must treat that as incomplete)."""
    if pos.underlying_price is None or pos.short_delta is None:
        return None
    shares, exact = net_delta_shares(pos.spread_type, pos.short_delta, pos.long_delta, pos.contracts)
    beta = betas.get(pos.symbol.upper())
    return Exposure(pos.key, pos.symbol, shares, beta or 1.0, beta is not None, exact,
                    shares * pos.underlying_price * (beta or 1.0))


def signal_dollar_delta(sig: SpreadSignal, betas: dict[str, float], contracts: int = 1) -> float:
    """Beta-weighted dollar delta a candidate would add."""
    shares, _ = net_delta_shares(sig.spread_type, sig.short_delta, sig.long_delta, contracts)
    return shares * sig.underlying_price * betas.get(sig.symbol.upper(), 1.0)


@dataclass(frozen=True)
class PortfolioSnapshot:
    exposures: list[Exposure] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # position keys whose exposure could not be computed
    spy_price: float | None = None
    equity: float = 0.0
    limit_pct: float = 1.0

    @property
    def complete(self) -> bool:
        return not self.missing and self.spy_price is not None and self.equity > 0

    @property
    def net_dollar_delta(self) -> float:
        return sum(e.dollar_delta for e in self.exposures)

    @property
    def spy_equivalent_shares(self) -> float | None:
        return self.net_dollar_delta / self.spy_price if self.spy_price else None

    @property
    def pct_of_equity(self) -> float | None:
        return self.net_dollar_delta / self.equity if self.equity > 0 else None

    @property
    def utilization(self) -> float | None:
        """Share of the cap in use (1.0 = at the limit), by absolute exposure."""
        pct = self.pct_of_equity
        return abs(pct) / self.limit_pct if pct is not None and self.limit_pct > 0 else None

    def to_dict(self) -> dict:
        spy_eq = self.spy_equivalent_shares
        return {
            "net_beta_delta_dollars": round(self.net_dollar_delta, 2),
            "spy_equivalent_shares": None if spy_eq is None else round(spy_eq, 2),
            "pct_of_equity": self.pct_of_equity, "limit_pct": self.limit_pct, "utilization": self.utilization,
            "spy_price": self.spy_price, "equity": self.equity, "complete": self.complete, "missing": self.missing,
            "positions": [{"key": e.key, "symbol": e.symbol, "net_delta_shares": round(e.net_delta_shares, 2),
                           "beta": e.beta, "beta_known": e.beta_known, "exact": e.exact,
                           "beta_delta_dollars": round(e.dollar_delta, 2)} for e in self.exposures],
        }


def build_snapshot(positions: list[SpreadPosition], betas: dict[str, float], spy_price: float | None, equity: float,
                   limit_pct: float) -> PortfolioSnapshot:
    exposures, missing = [], []
    for pos in positions:
        e = position_exposure(pos, betas)
        (exposures if e else missing).append(e or pos.key)
    return PortfolioSnapshot(exposures, missing, spy_price, equity, limit_pct)
