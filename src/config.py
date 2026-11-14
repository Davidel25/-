"""Environment-driven configuration. Everything has a safe default (mock broker, paper mode)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Gateway API ports on a bare-metal install. The gnzsnz/ib-gateway Docker image
# exposes 4003/4004 *inside* its network and forwards them to 4001/4002 on the host.
IB_PORT_LIVE = 4001
IB_PORT_PAPER = 4002


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _weekdays(name: str) -> tuple[int, ...] | None:
    """Comma-separated weekday numbers (0 = Monday ... 6); empty or invalid -> None (no filter)."""
    days = {int(x) for x in os.getenv(name, "").split(",") if x.strip().isdigit() and 0 <= int(x) <= 6}
    return tuple(sorted(days)) or None


def _list(name: str, default: str) -> list[str]:
    return [s.strip().upper() for s in os.getenv(name, default).split(",") if s.strip()]


@dataclass
class Settings:
    broker: str = "mock"            # "mock" | "ib"
    trading_mode: str = "paper"     # "paper" | "live"
    ib_host: str = "127.0.0.1"
    ib_port: int = IB_PORT_PAPER
    ib_client_id: int = 1
    ib_market_data_type: int = 3    # 1 live, 2 frozen, 3 delayed, 4 delayed-frozen
    watchlist: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    scan_interval_sec: int = 300
    scan_enabled: bool = True
    account_equity: float = 50_000.0
    telegram_token: str = ""
    telegram_chat_id: str = ""
    webhook_url: str = ""
    # --- guards / hardening (dataclass defaults keep mocks and tests simple; from_env turns the guards on) ---
    enforce_market_hours: bool = False   # skip scans outside the regular US session
    enforce_quote_freshness: bool = True # reject chains whose quotes are stale or untimed
    max_quote_age_sec: int = 0           # 0 = auto from IB_MARKET_DATA_TYPE (live 120s, delayed 1200s, frozen 60s)
    require_earnings_clear: bool = True  # block signals whose earnings date is unknown or before expiry
    disciplined_scanner: bool = False    # use ScannerConfig.disciplined() (credit floor, tight spreads, IV/HV, trend)
    trade_db_path: str = ""              # SQLite trade log; empty -> in-memory (lost on restart)
    commission_per_contract: float = 0.65
    allowed_weekdays: tuple[int, ...] | None = None  # entry-day filter, 0 = Monday (off by default: validate first)
    scale_out_profit_pct: float = 0.0    # 0 = off; 0.30 / 0.40 recommends closing half of a >= 2 contract position
    earnings_file: str = ""         # JSON earnings calendar (see src/risk/earnings.py); empty -> ETFs only
    entry_deltas_file: str = ""     # JSON journal {position key: entry |delta|} for the delta alert

    def quote_age_limit(self) -> float:
        """Maximum acceptable quote age in seconds. Delayed data (type 3/4) is ~15 min old by design."""
        if self.max_quote_age_sec > 0:
            return float(self.max_quote_age_sec)
        return {1: 120.0, 2: 60.0}.get(self.ib_market_data_type, 1200.0)

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("TRADING_MODE", "paper").lower()
        if mode not in {"paper", "live"}:
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        default_port = IB_PORT_LIVE if mode == "live" else IB_PORT_PAPER
        broker = os.getenv("BROKER", "mock").lower()
        return cls(
            broker=broker,
            trading_mode=mode,
            ib_host=os.getenv("IB_HOST", "127.0.0.1"),
            ib_port=int(os.getenv("IB_PORT", default_port)),
            ib_client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            ib_market_data_type=int(os.getenv("IB_MARKET_DATA_TYPE", "3")),
            watchlist=_list("WATCHLIST", "SPY,QQQ,IWM"),
            scan_interval_sec=int(os.getenv("SCAN_INTERVAL_SEC", "300")),
            scan_enabled=_bool("SCAN_ENABLED", True),
            account_equity=float(os.getenv("ACCOUNT_EQUITY", "50000")),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
            enforce_market_hours=_bool("ENFORCE_MARKET_HOURS", broker != "mock"),
            enforce_quote_freshness=_bool("ENFORCE_QUOTE_FRESHNESS", True),
            max_quote_age_sec=int(os.getenv("MAX_QUOTE_AGE_SEC", "0")),
            require_earnings_clear=_bool("REQUIRE_EARNINGS_CLEAR", True),
            disciplined_scanner=_bool("DISCIPLINED_SCANNER", True),
            trade_db_path=os.getenv("TRADE_DB_PATH", "data/trades.db"),
            commission_per_contract=float(os.getenv("COMMISSION_PER_CONTRACT", "0.65")),
            allowed_weekdays=_weekdays("ALLOWED_WEEKDAYS"),
            scale_out_profit_pct=float(os.getenv("SCALE_OUT_PROFIT_PCT", "0")),
            earnings_file=os.getenv("EARNINGS_FILE", ""),
            entry_deltas_file=os.getenv("ENTRY_DELTAS_FILE", ""),
        )
