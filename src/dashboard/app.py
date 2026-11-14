"""FastAPI dashboard: JSON API + single-page HTML. Run: uvicorn src.dashboard.app:create_app --factory"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.alerts.notifiers import AlertDispatcher, build_notifiers
from src.broker.factory import create_broker
from src.config import Settings
from src.service import ScanService

INDEX_HTML = (Path(__file__).parent / "index.html").read_text()


def create_app(service: ScanService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (service.settings if service else Settings.from_env())
    if service is None:
        service = ScanService(settings, create_broker(settings), AlertDispatcher(build_notifiers(settings)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(service.run_forever()) if settings.scan_enabled else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await service.broker.disconnect()

    app = FastAPI(title="Options Credit-Spread Scanner", lifespan=lifespan)
    state = service.state

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    def status() -> dict:
        return {
            "broker": settings.broker, "trading_mode": settings.trading_mode,
            "broker_connected": state.broker_connected, "last_scan": state.last_scan,
            "last_error": state.last_error, "scans": state.scans, "equity": state.equity,
            "watchlist": settings.watchlist, "signals": len(state.signals),
            "positions": len(state.positions), "rejected_by_risk": state.rejected,
        }

    @app.get("/api/signals")
    def signals() -> list[dict]:
        return state.signals

    @app.get("/api/positions")
    def positions() -> list[dict]:
        return state.positions

    return app

