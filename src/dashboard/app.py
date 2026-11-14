"""FastAPI dashboard: JSON API + single-page HTML. Run: uvicorn src.dashboard.app:create_app --factory"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.alerts.notifiers import AlertDispatcher, build_notifiers
from src.broker.factory import create_broker
from src.config import Settings
from src.service import ScanService
from src.storage.trade_log import ALLOWED_TAGS, MAX_NOTES

INDEX_HTML = (Path(__file__).parent / "index.html").read_text()
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"


class TagUpdate(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    tags: list[str] = Field(default_factory=list, max_length=len(ALLOWED_TAGS))
    notes: str | None = Field(default=None, max_length=MAX_NOTES)


def _performance(service: ScanService) -> dict:
    p = service.trade_log.performance()
    return {"closed": p.closed, "wins": p.wins, "win_rate": p.win_rate, "gross_pnl": p.gross_pnl,
            "commissions": p.commissions, "net_pnl": p.net_pnl, "is_estimate": True}


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
            "position_status": service.position_counts(), "alerts": len(state.alerts),
            "market_open": state.market_open, "skip_reason": state.skip_reason,
            "skipped": len(state.skipped), "blocked": len(state.blocked),
            "performance": _performance(service),
        }

    @app.get("/api/signals")
    def signals() -> list[dict]:
        return state.signals

    @app.get("/api/positions")
    def positions() -> list[dict]:
        return state.positions

    @app.get("/api/trades")
    def trades(limit: int = Query(2000, ge=1, le=10000), start: str | None = Query(None, pattern=DATE_RE),
               end: str | None = Query(None, pattern=DATE_RE)) -> list[dict]:
        """Closed trades from the SQLite log, optionally limited to a close-date range (YYYY-MM-DD, inclusive).
        P&L is an ESTIMATE (last seen debit, modelled commissions) until reconciled with broker statements."""
        return service.trade_log.closed_trades(limit, start, end)

    @app.get("/api/pnl")
    def pnl(start: str | None = Query(None, pattern=DATE_RE), end: str | None = Query(None, pattern=DATE_RE)) -> dict:
        """Daily open-position (unrealized) P&L snapshots by strategy, plus the realized summary."""
        return {"snapshots": service.trade_log.snapshots(start, end), "performance": _performance(service)}

    @app.get("/api/tags")
    def tags() -> dict:
        return {"tags": list(ALLOWED_TAGS), "max_notes": MAX_NOTES}

    @app.post("/api/trades/tag")
    def tag_trade(body: TagUpdate) -> dict:
        """Tag / annotate a trade for later analysis. Writes only to the local journal; no broker access."""
        try:
            return service.trade_log.set_tags(body.key, body.tags, body.notes)
        except KeyError:
            raise HTTPException(404, f"unknown trade {body.key}") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/breakdown")
    def breakdown(by: str = Query("dte_bucket", pattern="^(dte_bucket|weekday|tag)$")) -> list[dict]:
        """Closed-trade results by entry DTE bucket, entry weekday (0 = Monday) or tag: evidence for the filters."""
        return service.trade_log.breakdown(by)

    @app.get("/api/portfolio")
    def portfolio() -> dict:
        """Beta-weighted (SPY-equivalent) delta of the open spreads versus the cap."""
        return state.portfolio

    @app.get("/api/scenarios")
    def scenarios() -> list[dict]:
        """What-if presets: every open spread repriced under each shock. A rough stress test, not a forecast."""
        return state.scenarios

    @app.get("/api/earnings")
    def earnings() -> dict:
        """What the earnings gate believes per symbol, plus feed health. UNKNOWN means signals are blocked."""
        return service.earnings_table()

    @app.get("/api/diagnostics")
    def diagnostics() -> dict:
        return {"skipped": state.skipped, "blocked": state.blocked}

    @app.get("/api/alerts")
    def alerts() -> list[dict]:
        return state.alerts

    return app

