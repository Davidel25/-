# CLAUDE.md

## Project overview
Options scanner, alerting and position-monitoring system for US **credit spreads**:
**Bull Put** (sell OTM put, buy lower put) and **Bear Call** (sell OTM call, buy higher call).
Data and positions come from **Interactive Brokers via IB Gateway**. A FastAPI dashboard shows
signals, open positions and system status. Designed to run on a single AWS EC2 host with Docker.

**Safety stance: this system is read-only.** It scans, alerts and *recommends* exits (50% profit
target, stop-loss, DTE exit). It never places orders. The IB Gateway is started with
`READ_ONLY_API=yes` and the client connects with `readonly=True`. Do not add order placement
without an explicit request, a paper-only gate, and tests.

## Layout
```
src/
  config.py            env-driven Settings (BROKER, TRADING_MODE, IB_*, WATCHLIST, ...)
  models.py            OptionQuote, SpreadSignal, PositionLeg, SpreadPosition (dataclasses)
  service.py           ScanService: scan -> risk check -> alert; position exit monitoring; AppState
  broker/              base.py (BrokerClient ABC), ib_client.py (IB Gateway), mock_client.py, factory.py
  data/                greeks.py (Black-Scholes price/delta), market_data.py (normalise chains -> models/DataFrame)
  strategies/          credit_spreads.py (ScannerConfig + scan_bull_put / scan_bear_call / scan_all)
  risk/                manager.py (RiskLimits, RiskManager, pair_legs, ExitAction, AlertType/RiskAlert),
                       earnings.py (EarningsProvider, classify_earnings, JsonFile calendar), report.py (read-only CLI)
  alerts/              notifiers.py (Log / Telegram / Webhook, AlertDispatcher with cooldown)
  market_hours.py      US session / NYSE holiday guard
  storage/             trade_log.py (SQLite: signals, positions, auto entry delta, P&L snapshots, breakdowns)
  dashboard/           app.py (FastAPI factory) + index.html (auto-refresh, filters, CSV export, P&L chart, toasts)
tests/                 pytest; everything runs against MockBroker, no network
```
Dependency direction: `dashboard -> service -> (strategies, risk, alerts, data) -> broker.base`.
Strategy/risk code never imports `ib_client`; always go through `BrokerClient`.

## Commands
```bash
pip install -r requirements-dev.txt
pytest -q                                   # all tests (mock broker, offline)
ruff check src tests                        # lint (ruff check --fix to autofix)
BROKER=mock uvicorn src.dashboard.app:create_app --factory --reload   # local dashboard on :8000
BROKER=ib TRADING_MODE=paper IB_PORT=4002 uvicorn src.dashboard.app:create_app --factory   # vs local paper Gateway
docker compose up -d --build                # app + IB Gateway
```
Read-only position risk report (take-profit / stop / DTE / delta alert / strike proximity / earnings):
`python -m src.risk.report --demo` or `python -m src.risk.report --earnings-file earnings.json --entry-deltas deltas.json`
(exit code 0 clear, 1 attention needed, 2 broker error). Unknown earnings dates are flagged, never treated as clear.
Dashboard API: `/api/health`, `/api/status`, `/api/signals`, `/api/positions`, `/` (HTML).

## Code style
- Python 3.11+, type hints everywhere, `from __future__ import annotations`.
- Ruff, line length 120 (config in `pyproject.toml`). Dataclasses for domain objects (frozen where possible).
- Broker I/O is `async`; keep pure logic (scanner, risk, greeks) synchronous and I/O-free so it is trivial to test.
- Prices are **per share**; dollar fields (`max_profit`, `max_loss`) are **per 1 lot** (x100). `delta` is signed; scanner thresholds use absolute values.
- Never swallow broker errors silently: raise `BrokerError`; `ScanService` records them in `AppState.last_error`.
- Pass `today` explicitly into date-dependent pure functions (tests are deterministic).
- Every new strategy filter or risk rule needs a unit test in `tests/`.

## Strategy definitions
- ROC = `credit / (width - credit)` = max profit / max loss. POP ~ `1 - |short delta|`.
- Breakeven: bull put `short - credit`; bear call `short + credit`.
- Defaults (`ScannerConfig`): DTE 25-50, |short delta| 0.10-0.25, widths 1/2/5, credit >= 0.15, ROC >= 15%,
  max risk $1000/lot, OI >= 100, bid/ask <= 15% of mid. `pricing="natural"` uses short bid - long ask (conservative).
- Risk (`RiskLimits`): <=2% equity per trade, <=20% total open risk, <=10 positions, <=2 per symbol;
  exits: **take profit at 50% of credit**, stop at loss >= 2x credit, close at DTE <= 14.

## IB Gateway notes
- API ports on a bare-metal Gateway: **4001 live, 4002 paper** (TWS uses 7496/7497).
- In docker-compose the app reaches the gateway container at `ib-gateway:4003` (live) / `:4004` (paper);
  the host publishes them as 4001/4002 on `127.0.0.1` only. Set `IB_PORT` accordingly (see `.env.example`).
- `IB_CLIENT_ID` must be unique per API connection to the same Gateway.
- Market data: `IB_MARKET_DATA_TYPE=3` (delayed) works without subscriptions; use `1` with a live OPRA subscription.
- IB limits concurrent market-data lines (~100 default) and message rate (~50/s); `IBClient` requests in batches of 100.
  Keep strike bands and DTE windows narrow.
- IBKR forces a daily Gateway restart and 2FA on live logins. Use the VNC port (SSH tunnel) to approve 2FA; the app
  reconnects on the next scan cycle.
- `ib_async` is the maintained fork of `ib_insync` (same API). `ib_client.py` imports either.
- `ib_client.py` cannot run in CI. Validate changes manually against a **paper** Gateway first.

## Deployment (AWS EC2 + Docker)
1. Launch Ubuntu 22.04/24.04, t3.small+ (2 GB RAM min; the Gateway JVM is hungry), 20 GB gp3. Install Docker + compose plugin.
2. Security group: allow **SSH only from your IP**. Do NOT open 8000, 4001-4004 or 5900 to the internet; the compose file
   binds them to `127.0.0.1`. Reach the dashboard with `ssh -L 8000:localhost:8000 ec2-user@host` (or SSM port-forward,
   or an authenticated reverse proxy - the dashboard has no auth).
3. `git clone`, `cp .env.example .env`, fill `TWS_USERID`/`TWS_PASSWORD`. Keep `.env` out of git (it is in `.gitignore`);
   prefer AWS SSM Parameter Store / Secrets Manager for real deployments.
4. Roll-out order: `BROKER=mock` -> `BROKER=ib TRADING_MODE=paper` (IB_PORT=4004) -> live (IB_PORT=4003) only after paper is stable.
5. `docker compose up -d --build`; `docker compose logs -f app`; `docker compose ps` shows health.
6. Pin the gateway image to a specific tag before relying on it, and verify env var names in the
   gnzsnz/ib-gateway-docker docs for that tag.

## Edge filters and guards (all read-only)
- `ScannerConfig.disciplined()` bundles: credit >= 0.60, ROC >= 20% natural, bid/ask <= 5% and <= $0.05 per leg, IV/HV >= 1.1,
  trend alignment (Bull Put above SMA50, Bear Call below), term-structure slope >= +1%, 25d put skew <= 10 vol pts (Bull Puts),
  DTE 30-50 with 35-45 ranked first. `allowed_weekdays` (entry days) is off by default: the filter exists, validate it with
  `GET /api/breakdown?by=weekday|dte_bucket` before enabling. Missing data always fails closed.
- `ScanService` blocks signals with unknown / in-window earnings, skips scans outside market hours and on stale quotes, captures
  `entry_short_delta` automatically (signal delta > first-scan delta; the manual journal overrides), and logs to SQLite.
  Closed-trade P&L is an ESTIMATE (last seen debit, modelled commissions); a position must be missing for 2 scans to count as closed.
- Earnings feed (`src/risk/earnings_feed.py`, `GET /api/earnings`, dashboard card): optional automated calendar
  (`EARNINGS_PROVIDER=finnhub` + `EARNINGS_API_KEY`) layered UNDER the manual `EARNINGS_FILE`. Fail closed: vendor dates are only
  ever PROJECTED, an empty answer / missing symbol / failed fetch older than `EARNINGS_MAX_STALE_HOURS` / implausibly distant date
  is UNKNOWN (signals blocked), ETFs never hit the network, the key never appears in logs/repr/errors. Network I/O runs in a worker
  thread inside `refresh()`; `get()` reads memory. The Finnhub adapter follows the documented shape but is NOT verified against
  the live API - if auth is rejected the dashboard shows the error; try `EARNINGS_AUTH=query`.
- Portfolio guard (`src/risk/portfolio.py`): net beta-weighted (SPY-equivalent) dollar delta of all open spreads vs
  `MAX_NET_BETA_DELTA_PCT` of equity (default 100%). Signals that would ADD exposure past the cap are blocked, exposure-reducing
  ones are allowed, incomplete data (missing delta/price/SPY quote) blocks (fail closed). Betas are rough static inputs (`BETAS`).
  Candidates are each tested against the CURRENT book, not as a basket. `GET /api/portfolio`, alert `BETA_DELTA_LIMIT`.
- What-if scenarios (`src/risk/scenarios.py`, `GET /api/scenarios`, dashboard card): every open spread is repriced with
  Black-Scholes under 7 presets (SPY -3% / -5% / +3%, IV +10 pts, SPY -3% + IV +5, SPY -10% + IV +15, 5 days pass). Stocks move
  beta x the SPY move; the model move is mapped onto the broker's mark so zero shock = zero change and a total wipe-out is
  exactly -max loss. Parallel vol shift only (no skew/term dynamics), instantaneous move, no early assignment: a rough stress
  test, not a forecast. Positions without leg IVs are excluded and the result is flagged incomplete.
- Trade tagging: fixed vocabulary (`ALLOWED_TAGS`) + a 500-char note per trade via `POST /api/trades/tag` (dashboard "Tag"
  button); `GET /api/breakdown?by=tag` aggregates P&L by tag. Old SQLite files are migrated in place (`_ADDED_COLUMNS`).
- Optional scale-out (`SCALE_OUT_PROFIT_PCT`, needs >= 2 contracts): advisory only, lowers variance, does not raise expectancy,
  and is NOT risk-free (the remaining contracts keep their full max loss).

## Backtesting (`src/backtest/`)
`python -m src.backtest --data chains.csv [--earnings earnings.csv] [--oos-from YYYY-MM-DD] [--out report.md] [--json out.json]`
replays DAILY end-of-day option-chain snapshots through the REAL scanner (`ScannerConfig.disciplined()` by default), earnings gate,
`RiskManager.check_signal` and the live exit rules (50% / 2x stop / 14 DTE), with commissions and natural-fill pricing. Input columns:
`date,symbol,expiry,strike,right,bid,ask,underlying_price[,delta][,iv][,open_interest][,volume]` (see `src/backtest/data.py`; option
history must be bought, e.g. CBOE DataShop / ORATS / OptionMetrics / Polygon). Statistics use a CLUSTER bootstrap (whole entry-months)
and the verdict is INCONCLUSIVE below 100 trades / 12 months. `--demo` runs on SYNTHETIC Black-Scholes chains: it proves the pipeline
and says nothing about real edge. Calibration on synthetic data (12 runs, ~150 trades over 41 months each): 0 false "positive edge"
verdicts at zero premium; with a 15% implied-vol premium most runs were still "not distinguishable from zero" because friction eats
the premium and ~150 trades cannot resolve a ~$5/trade effect - expect to need many symbols/years of data. Limits: EOD fills, no early
assignment, hindsight earnings dates, overlapping/correlated trades, filters tuned in-sample (use `--oos-from`).

## Known gaps / next steps
- Dashboard live state is in-memory; trade history / P&L snapshots persist in SQLite (`TRADE_DB_PATH`). Fills are not read from the broker, so P&L is estimated.
- No dashboard authentication, and `pair_legs` only recognises simple verticals (no iron condors / rolled legs).
- IV rank and ex-dividend filters are not implemented (earnings: see the feed above); add them in `strategies/` before trusting signals.
- Black-Scholes fallback delta ignores dividends and American exercise; prefer broker model Greeks.
