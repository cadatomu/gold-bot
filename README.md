# gold-bot — XAUUSD Trend-Following Bot

Algorithmic trading system for XAUUSD, H4 timeframe. Targets 15-30% annual return
with max drawdown ≤ 15% (Calmar > 1.0). Runs on a Linux VPS via MT5.

## Stack

| Concern | Library |
|---|---|
| Broker connectivity | MetaTrader5 (official Python package) |
| Backtesting | vectorbt |
| Hyperparameter tuning | Optuna |
| Reporting | quantstats |
| Config validation | pydantic |
| Logging | structlog |
| Alerts | python-telegram-bot |

## Setup

### 1. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your MT5 credentials
```

### 3. MetaTrader5 on Linux

The official `MetaTrader5` package requires the MT5 terminal binary to be running.
On a headless Linux VPS, two options:

**Option A — MT5 Linux build** (recommended):
IC Markets and Pepperstone provide a native Linux MT5 terminal.
Download from the broker's client portal and run it as a service.

**Option B — Wine**:
```bash
sudo apt install wine
# Install MT5 Windows terminal under Wine
# Set MT5_TERMINAL_PATH in .env to the Wine path
```

### 4. Run tests

```bash
pytest
```

All tests use mocks — no live MT5 connection required.

## Project Structure

```
gold_bot/
├── config/
│   ├── settings.yaml          # General settings (env, broker, risk)
│   └── strategy_params.yaml   # Strategy parameters (tunable)
├── src/
│   ├── data/
│   │   ├── mt5_connector.py   # MT5 wrapper (context manager, DI-friendly)
│   │   └── historical.py      # Download + cache + gap validation
│   ├── strategy/              # FASE 2 — signal generation (pure functions)
│   ├── risk/                  # FASE 3 — sizing + circuit breakers
│   ├── execution/             # FASE 4 — order management
│   ├── backtest/              # FASE 2 — vectorbt runner + walk-forward
│   ├── live/                  # FASE 4 — main loop + scheduler
│   └── monitoring/            # Logging + Telegram alerts
└── tests/                     # pytest, all mocked — no live terminal needed
```

## Strategy (v1 — TrendATR)

- **Timeframe**: H4
- **Entry**: Close > EMA(20) after pullback that touched EMA(20), with EMA(50) > EMA(200)
- **Volatility filter**: ATR(14) > 25th percentile of last 100 bars
- **SL**: 1.5 × ATR from entry bar
- **TP**: 3.0 × ATR (R:R = 1:2), optional trailing after 1.5R
- **Position size**: 0.5% equity risk per trade
- **News blackout**: ±30 min around FOMC, NFP, CPI

## Regulatory note

VPS is in the EU (Lithuania). Effective leverage capped at 1:20 for gold per ESMA.
Bot enforces 1:5–1:10 effective leverage via position sizing.

## Phases

| Phase | Status | Contents |
|---|---|---|
| FASE 1 | ✅ Done | Infrastructure, MT5 connector, historical data, tests |
| FASE 2 | ⏳ Pending | Strategy, vectorbt backtester, walk-forward |
| FASE 3 | ⏳ Pending | Position sizing, circuit breakers |
| FASE 4 | ⏳ Pending | Order manager, live loop, Telegram |
| FASE 5 | ⏳ Pending | Full validation, stress tests, Optuna tuning |
