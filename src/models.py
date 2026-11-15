"""Shared domain models. Plain dataclasses so every layer can import them cheaply."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum

CONTRACT_MULTIPLIER = 100


def dte_bucket(dte: int) -> str:
    """Coarse expiry bucket used for filtering preferences and for trade-log analysis."""
    if dte < 25:
        return "<25"
    if dte < 30:
        return "25-29"
    if dte < 35:
        return "30-34"
    if dte <= 45:
        return "35-45"
    return "46+" if dte <= 50 else ">50"


def position_key(symbol: str, spread_type: SpreadType | str, expiry: date, short_strike: float,
                 long_strike: float) -> str:
    """One canonical id for a spread, shared by signals, positions, the entry-delta journal and the trade log."""
    st = spread_type.value if isinstance(spread_type, SpreadType) else spread_type
    return f"{symbol}:{st}:{expiry}:{short_strike:g}/{long_strike:g}"


class SpreadType(StrEnum):
    BULL_PUT = "BULL_PUT"    # short put + long lower put  (bullish, credit)
    BEAR_CALL = "BEAR_CALL"  # short call + long higher call (bearish, credit)

    @property
    def right(self) -> str:
        return "P" if self is SpreadType.BULL_PUT else "C"


@dataclass(frozen=True)
class OptionQuote:
    """One option contract's market snapshot. Prices are per share."""

    symbol: str
    expiry: date
    strike: float
    right: str  # "P" or "C"
    bid: float
    ask: float
    underlying_price: float
    delta: float | None = None  # signed broker/model delta; None -> computed later
    iv: float | None = None
    volume: int = 0
    open_interest: int = 0
    quote_time: datetime | None = None  # when the broker last updated this quote (tz-aware); None = unknown

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        """Bid/ask width relative to mid (liquidity proxy)."""
        return (self.ask - self.bid) / self.mid if self.mid > 0 else float("inf")

    def dte(self, today: date | None = None) -> int:
        return (self.expiry - (today or date.today())).days


@dataclass(frozen=True)
class SpreadSignal:
    """A candidate credit spread produced by the scanner. Money fields are per 1 lot (x100)."""

    spread_type: SpreadType
    symbol: str
    expiry: date
    dte: int
    underlying_price: float
    short_strike: float
    long_strike: float
    width: float
    credit: float            # per share
    max_profit: float        # $ per spread
    max_loss: float          # $ per spread (the "max risk")
    roc: float               # return on capital = max_profit / max_loss
    short_delta: float       # absolute value
    pop: float               # approx. probability of profit = 1 - |short delta|
    breakeven: float
    otm_pct: float           # distance of short strike from spot, fraction
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    long_delta: float | None = None  # absolute; lets portfolio code net the vertical's delta

    @property
    def key(self) -> str:
        return position_key(self.symbol, self.spread_type, self.expiry, self.short_strike, self.long_strike)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spread_type"] = self.spread_type.value
        d["expiry"] = self.expiry.isoformat()
        d["created_at"] = self.created_at.isoformat()
        d["key"] = self.key
        return d


@dataclass(frozen=True)
class PositionLeg:
    """A single option position leg as reported by the broker. Prices per share."""

    symbol: str
    expiry: date
    strike: float
    right: str
    quantity: int        # signed: negative = short
    avg_price: float     # average fill price per share
    market_price: float  # current mark per share


@dataclass(frozen=True)
class SpreadPosition:
    """A paired short/long leg held in the account."""

    spread_type: SpreadType
    symbol: str
    expiry: date
    short_strike: float
    long_strike: float
    contracts: int
    entry_credit: float   # per share
    current_debit: float  # per share cost to close now
    # Optional live inputs for the delta / strike-proximity alerts. ``None`` means "unknown" (the broker's
    # position feed does not carry them); the corresponding alert is then skipped, never guessed.
    underlying_price: float | None = None
    short_delta: float | None = None        # current |delta| of the short leg
    entry_short_delta: float | None = None  # |delta| of the short leg when the trade was opened
    initial_contracts: int | None = None    # size when first seen (None = unknown/never scaled); drives scale-out
    long_delta: float | None = None         # current |delta| of the long leg (for the vertical's net delta)
    short_iv: float | None = None           # current implied vol of the short / long leg (annualised, 0.20 = 20%);
    long_iv: float | None = None            # used by the what-if scenarios, None -> scenario is not computable
    opened_on: date | None = None           # entry date; None = unknown (the broker feed omits it) -> time-stop skipped

    @property
    def key(self) -> str:
        """Stable identifier, also used as the entry-delta journal key."""
        return position_key(self.symbol, self.spread_type, self.expiry, self.short_strike, self.long_strike)

    @property
    def delta_ratio(self) -> float | None:
        """Current short |delta| as a multiple of its entry value (>= 2 trips the delta alert)."""
        if self.short_delta is None or not self.entry_short_delta:
            return None
        return abs(self.short_delta) / abs(self.entry_short_delta)

    @property
    def distance_to_short_pct(self) -> float | None:
        """Cushion between the underlying and the short strike as a fraction of the short strike.
        Positive = still out of the money, 0 = at the strike, negative = breached. None if price unknown."""
        if self.underlying_price is None or self.short_strike <= 0:
            return None
        if self.spread_type is SpreadType.BULL_PUT:
            return (self.underlying_price - self.short_strike) / self.short_strike
        return (self.short_strike - self.underlying_price) / self.short_strike

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    @property
    def max_loss(self) -> float:
        return (self.width - self.entry_credit) * CONTRACT_MULTIPLIER * self.contracts

    @property
    def unrealized_pnl(self) -> float:
        return (self.entry_credit - self.current_debit) * CONTRACT_MULTIPLIER * self.contracts

    @property
    def profit_pct(self) -> float:
        """Fraction of the initial credit already captured (0.5 == 50%)."""
        return (self.entry_credit - self.current_debit) / self.entry_credit if self.entry_credit > 0 else 0.0

    def dte(self, today: date | None = None) -> int:
        return (self.expiry - (today or date.today())).days

    def life_elapsed(self, today: date | None = None) -> float | None:
        """Share of the trade's life used (0 = just opened, 1 = expiry); None if the entry date is unknown."""
        if self.opened_on is None:
            return None
        life = (self.expiry - self.opened_on).days
        if life <= 0:
            return None
        return ((today or date.today()) - self.opened_on).days / life

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            spread_type=self.spread_type.value,
            expiry=self.expiry.isoformat(),
            width=self.width,
            max_loss=self.max_loss,
            unrealized_pnl=self.unrealized_pnl,
            profit_pct=self.profit_pct,
            dte=self.dte(),
            key=self.key,
            distance_to_short_pct=self.distance_to_short_pct,
            delta_ratio=self.delta_ratio,
        )
        return d
