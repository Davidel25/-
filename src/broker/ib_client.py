"""Interactive Brokers client over IB Gateway using ``ib_async`` (maintained fork of
``ib_insync``; falls back to ``ib_insync`` if that is what is installed).

Ports: 4001 = live, 4002 = paper (bare-metal Gateway). In docker-compose the app talks to
the gateway container on 4003 (live) / 4004 (paper) - see CLAUDE.md.

This module is READ-ONLY by design: it never places orders.
NOTE: needs a running Gateway, so it is exercised manually / in paper, not in CI."""
from __future__ import annotations

import logging
import math
from datetime import datetime

from src.broker.base import BrokerClient, BrokerError, expiries_in_window
from src.config import Settings
from src.models import CONTRACT_MULTIPLIER, OptionQuote, PositionLeg

log = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import ib_async as ib_lib
except ImportError:  # pragma: no cover
    try:
        import ib_insync as ib_lib
    except ImportError:
        ib_lib = None


def _clean(x: float | None) -> float | None:
    """IB uses nan / -1 for 'no data'."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or x < 0:
        return None
    return float(x)


class IBClient(BrokerClient):
    def __init__(self, host: str, port: int, client_id: int, market_data_type: int = 3,
                 connect_timeout: float = 15.0, max_batch: int = 100) -> None:
        if ib_lib is None:
            raise BrokerError("ib_async / ib_insync is not installed")
        self.host, self.port, self.client_id = host, port, client_id
        self.market_data_type = market_data_type
        self.connect_timeout = connect_timeout
        self.max_batch = max_batch  # IB limits concurrent market-data lines
        self.ib = ib_lib.IB()

    @classmethod
    def from_settings(cls, s: Settings) -> IBClient:
        return cls(s.ib_host, s.ib_port, s.ib_client_id, s.ib_market_data_type)

    async def connect(self) -> None:
        if self.is_connected():
            return
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id,
                                       timeout=self.connect_timeout, readonly=True)
        except (TimeoutError, OSError) as exc:
            raise BrokerError(f"cannot reach IB Gateway at {self.host}:{self.port}: {exc}") from exc
        self.ib.reqMarketDataType(self.market_data_type)
        log.info("connected to IB %s:%s clientId=%s", self.host, self.port, self.client_id)

    async def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    async def _underlying(self, symbol: str):
        stock = ib_lib.Stock(symbol, "SMART", "USD")
        qualified = await self.ib.qualifyContractsAsync(stock)
        if not qualified:
            raise BrokerError(f"could not qualify {symbol}")
        return qualified[0]

    async def get_underlying_price(self, symbol: str) -> float:
        contract = await self._underlying(symbol)
        (ticker,) = await self.ib.reqTickersAsync(contract)
        price = _clean(ticker.marketPrice()) or _clean(ticker.close)
        if price is None:
            raise BrokerError(f"no price for {symbol}")
        return price

    async def get_option_chain(
        self, symbol: str, min_dte: int = 20, max_dte: int = 60, strike_band_pct: float = 0.15
    ) -> list[OptionQuote]:
        stock = await self._underlying(symbol)
        spot = await self.get_underlying_price(symbol)
        params = await self.ib.reqSecDefOptParamsAsync(stock.symbol, "", stock.secType, stock.conId)
        params = [p for p in params if p.exchange == "SMART"] or params
        if not params:
            raise BrokerError(f"no option parameters for {symbol}")
        p = max(params, key=lambda x: len(x.strikes))

        expiries = expiries_in_window(
            [datetime.strptime(e, "%Y%m%d").date() for e in p.expirations], min_dte, max_dte)
        strikes = sorted(k for k in p.strikes if abs(k - spot) <= spot * strike_band_pct)

        contracts = [ib_lib.Option(symbol, e.strftime("%Y%m%d"), k, r, "SMART", tradingClass=p.tradingClass)
                     for e in expiries for k in strikes for r in ("P", "C")]
        # qualify/request in batches so we respect IB's market data line limits
        quotes: list[OptionQuote] = []
        for i in range(0, len(contracts), self.max_batch):
            batch = await self.ib.qualifyContractsAsync(*contracts[i:i + self.max_batch])
            batch = [c for c in batch if c.conId]
            tickers = await self.ib.reqTickersAsync(*batch)
            for t in tickers:
                q = self._to_quote(symbol, spot, t)
                if q:
                    quotes.append(q)
        return quotes

    @staticmethod
    def _to_quote(symbol: str, spot: float, t) -> OptionQuote | None:
        bid, ask = _clean(t.bid), _clean(t.ask)
        if bid is None or ask is None or ask <= 0:
            return None
        c = t.contract
        g = t.modelGreeks
        right = c.right[0].upper()
        oi = t.putOpenInterest if right == "P" else t.callOpenInterest
        return OptionQuote(
            symbol=symbol,
            expiry=datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d").date(),
            strike=float(c.strike), right=right, bid=bid, ask=ask, underlying_price=spot,
            delta=_clean_signed(g.delta) if g else None,
            iv=_clean(g.impliedVol) if g else None,
            volume=int(_clean(t.volume) or 0),
            open_interest=int(_clean(oi) or 0),
        )

    async def get_option_positions(self) -> list[PositionLeg]:
        legs: list[PositionLeg] = []
        for item in self.ib.portfolio():
            c = item.contract
            if c.secType != "OPT":
                continue
            legs.append(PositionLeg(
                symbol=c.symbol,
                expiry=datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d").date(),
                strike=float(c.strike), right=c.right[0].upper(),
                quantity=int(item.position),
                avg_price=item.averageCost / CONTRACT_MULTIPLIER,
                market_price=item.marketPrice,
            ))
        return legs

    async def get_account_equity(self) -> float | None:
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
        return None


def _clean_signed(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
