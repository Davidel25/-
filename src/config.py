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
    earnings_file: str = ""         # JSON earnings calendar (see src/risk/earnings.py); empty -> ETFs only
    entry_deltas_file: str = ""     # JSON journal {position key: entry |delta|} for the delta alert

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("TRADING_MODE", "paper").lower()
        if mode not in {"paper", "live"}:
            raise ValueError("TRADING_MODE must be 'paper' or 'live'")
        default_port = IB_PORT_LIVE if mode == "live" else IB_PORT_PAPER
        return cls(
            broker=os.getenv("BROKER", "mock").lower(),
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
            earnings_file=os.getenv("EARNINGS_FILE", ""),
            entry_deltas_file=os.getenv("ENTRY_DELTAS_FILE", ""),
        )
