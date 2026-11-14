"""Notification handlers. Add a channel by subclassing ``Notifier``."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import requests

from src.config import Settings
from src.models import SpreadSignal

log = logging.getLogger(__name__)


class Notifier(ABC):
    @abstractmethod
    async def send(self, message: str) -> None: ...


class LogNotifier(Notifier):
    async def send(self, message: str) -> None:
        log.info("ALERT: %s", message)


class TelegramNotifier(Notifier):
    def __init__(self, token: str, chat_id: str) -> None:
        self.url, self.chat_id = f"https://api.telegram.org/bot{token}/sendMessage", chat_id

    async def send(self, message: str) -> None:
        r = await asyncio.to_thread(
            requests.post, self.url, json={"chat_id": self.chat_id, "text": message}, timeout=10)
        r.raise_for_status()


class WebhookNotifier(Notifier):
    """Generic JSON webhook (Slack/Discord/Make/n8n compatible: {"text": ...})."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def send(self, message: str) -> None:
        r = await asyncio.to_thread(requests.post, self.url, json={"text": message, "content": message}, timeout=10)
        r.raise_for_status()


def format_signal(s: SpreadSignal) -> str:
    name = "Bull Put" if s.spread_type.value == "BULL_PUT" else "Bear Call"
    return (f"{s.symbol} {name} {s.expiry:%d%b%y} {s.short_strike:g}/{s.long_strike:g} | "
            f"credit ${s.credit:.2f} | risk ${s.max_loss:.0f} | ROC {s.roc:.0%} | "
            f"Δ {s.short_delta:.2f} | POP {s.pop:.0%} | BE {s.breakeven:g} | DTE {s.dte}")


class AlertDispatcher:
    """Fan-out to all notifiers with per-key cooldown so a signal is not re-sent every scan."""

    def __init__(self, notifiers: list[Notifier], cooldown_sec: float = 6 * 3600) -> None:
        self.notifiers, self.cooldown_sec = notifiers, cooldown_sec
        self._last_sent: dict[str, float] = {}

    async def dispatch(self, key: str, message: str) -> bool:
        now = time.monotonic()
        if key in self._last_sent and now - self._last_sent[key] < self.cooldown_sec:
            return False
        self._last_sent[key] = now
        results = await asyncio.gather(*(n.send(message) for n in self.notifiers), return_exceptions=True)
        for n, res in zip(self.notifiers, results, strict=True):
            if isinstance(res, Exception):  # one broken channel must not block the others
                log.error("notifier %s failed: %s", type(n).__name__, res)
        return True

    async def signal(self, s: SpreadSignal) -> bool:
        return await self.dispatch(f"sig:{s.key}", format_signal(s))


def build_notifiers(s: Settings) -> list[Notifier]:
    notifiers: list[Notifier] = [LogNotifier()]
    if s.telegram_token and s.telegram_chat_id:
        notifiers.append(TelegramNotifier(s.telegram_token, s.telegram_chat_id))
    if s.webhook_url:
        notifiers.append(WebhookNotifier(s.webhook_url))
    return notifiers
