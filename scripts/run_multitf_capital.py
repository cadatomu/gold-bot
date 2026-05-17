"""Backtest multi-timeframe (H1 y H4) con datos reales de Capital.com.

NO abre órdenes. Solo descarga datos y corre el backtest.

Uso:
  python scripts/run_multitf_capital.py              # backtest con defaults
  python scripts/run_multitf_capital.py --optimize   # optimiza H1 y H4
  python scripts/run_multitf_capital.py --trials 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import optuna
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from src.backtest.runner import BrokerConfig
from src.backtest.runner_advanced import run_backtest_advanced
from src.strategy.adaptive import AdaptiveParams, AdaptiveScalp15m
from src.strategy.regime import RegimeParams
from src.strategy.scalp_15m import build_h1_trend

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_URL = "https://api-capital.backend-capital.com/api/v1"

BROKER = BrokerConfig(
    spread_pips           = 0.50,
    commission_per_lot_rt = 0.0,
    slippage_pips         = 0.20,
    risk_per_trade_pct    = 0.005,
    initial_equity        = 10_000.0,
)

MONTHS = 20
DAYS   = int(MONTHS * 30.5)   # ~610 días


# ── Capital.com helpers ───────────────────────────────────────────────────────

def _open_session() -> dict:
    resp = requests.post(
        f"{BASE_URL}/session",
        json={"identifier": os.environ["CAPITAL_EMAIL"],
              "password":   os.environ["CAPITAL_PASSWORD"],
              "encryptedPassword": False},
        headers={"X-CAP-API-KEY": os.environ["CAPITAL_API_KEY"],
                 "Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return {"X-CAP-API-KEY":    os.environ["CAPITAL_API_KEY"],
            "CST":              resp.headers["CST"],
            "X-SECURITY-TOKEN": resp.headers["X-SECURITY-TOKEN"],
            "Content-Type":     "application/json"}


def _fetch_range(hdrs: dict, resolution: str, from_dt: datetime, to_dt: datetime) -> list:
    r = requests.get(f"{BASE_URL}/prices/GOLD",
        params={"resolution": resolution, "max": 1000,
                "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%S")},
        headers=hdrs, timeout=15)
    if r.status_code != 200:
        return []
    return r.json().get("prices", [])


def _paginate(hdrs: dict, resolution: str, total_days: int, chunk_days: int) -> list:
    all_prices: list = []
    to_dt = datetime.now(timezone.utc)
    fetched_days = 0
    while fetched_days < total_days:
        from_dt = to_dt - timedelta(days=chunk_days)
        chunk = _fetch_range(hdrs, resolution, from_dt, to_dt)
        if not chunk:
            break
        all_prices = chunk + all_prices
        to_dt = from_dt
        fetched_days += chunk_days
        time.sleep(0.25)
    return all_prices


def _to_df(prices: list) -> pd.DataFrame:
    rows = []
    for p in prices:
        mid = lambda s: (s["bid"] + s["ask"]) / 2
        rows.append({"time":   p["snapshotTimeUTC"],
                     "open":   mid(p["openPrice"]),
                     "high":   mid(p["highPrice"]),
                     "low":    mid(p["lowPrice"]),
                     "close":  mid(p["closePrice"]),
                     "volume": p.get("lastTradedVolume", 0)})
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


# ── Descarga ──────────────────────────────────────────────────────────────────

def download_all(hdrs: dict) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}

    configs = [
        ("15m",  "MINUTE_15", DAYS, 10),
        ("H1",   "HOUR",      DAYS, 40),
        ("H4",   "HOUR_4",    DAYS, 100),
        ("Daily","DAY",       DAYS + 200, 400),
    ]

    for label, res, total, chunk in configs:
        print(f"  Descargando {label} ({total//30:.0f} meses)…")
        raw = _paginate(hdrs, res, total, chunk)
        df  = _to_df(raw)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=total)
        df = df[df.index >= cutoff]
        print(f"    {len(df)} barras  {df.index[0].date()} → {df.index[-1].date()}")
        dfs[label] = df

    return dfs


def build_df(entry_df: pd.DataFrame, trend_df: pd.DataFrame,
             disable_session: bool = False) -> pd.DataFrame:
    h1 = build_h1_trend(trend_df)
    h1.index = h1.index.tz_localize("UTC") if h1.index.tzinfo is None else h1.index
    df = entry_df.copy()
    df["h1_trend"] = h1.reindex(entry_df.index, method="ffill").fillna(0).astype(int)
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(result, title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  Total return  : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe        : {result.sharpe:.3f}")
    print(f"  Calmar        : {result.calmar:.3f}")
    print(f"  Max DD        : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate      : {result.win_rate_pct:.1f}%")
    print(f"  Total trades  : {result.total_trades}")
    print(f"  Profit factor : {result.profit_factor:.2f}")


def _print_monthly(trades_df: pd.DataFrame, months: int = 20):
    if trades_df.empty:
        print("  (sin trades)")
        return
    td = trades_df[trades_df["exit_reason"] != "PARTIAL"].copy()
    td["month"] = pd.to_datetime(td["exit_time"]).dt.to_period("M")
    m = td.groupby("month").agg(
        trades  = ("pnl_usd", "count"),
        pnl_usd = ("pnl_usd", "sum"),
        win_pct = ("pnl_usd", lambda x: (x > 0).mean() * 100),
        avg     = ("pnl_usd", "mean"),
        mejor   = ("pnl_usd", "max"),
        peor    = ("pnl_usd", "min"),
    ).reset_index()
    m["month"] = m["month"].astype(str)
    print(f"\n{'─'*90}")
    print(f"  {'Mes':<10} {'Trades':>6} {'PnL USD':>10} {'Win%':>7} {'Avg':>9} {'Mejor':>10} {'Peor':>10}")
    print(f"{'─'*90}")
    for _, r in m.iterrows():
        mark = " ★" if r["pnl_usd"] > 0 else "  "
        print(f"  {r['month']:<10} {r['trades']:>6} {r['pnl_usd']:>10.2f}"
              f" {r['win_pct']:>7.1f} {r['avg']:>9.2f}"
              f" {r['mejor']:>10.2f} {r['peor']:>10.2f}{mark}")
    print(f"{'─'*90}")
    profitable = (m["pnl_usd"] > 0).sum()
    print(f"  {'TOTAL':<10} {m['trades'].sum():>6} {m['pnl_usd'].sum():>10.2f}"
          f"   Meses positivos: {profitable}/{len(m)}")


def _print_direction(trades_df: pd.DataFrame):
    if trades_df.empty or "direction" not in trades_df.columns:
        return
    td = trades_df[trades_df["exit_reason"] != "PARTIAL"]
    if td.empty:
        return
    print("\nPor dirección:")
    grp = td.groupby("direction").agg(
        trades   = ("pnl_usd", "count"),
        pnl_usd  = ("pnl_usd", "sum"),
        win_rate = ("pnl_usd", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        avg      = ("pnl_usd", "mean"),
    )
    print(grp.round(2).to_string())


# ── Optimización ──────────────────────────────────────────────────────────────

def _objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    params = AdaptiveParams(
        regime = RegimeParams(
            adx_trend_min   = trial.suggest_float("regime_adx",    20.0, 40.0, step=1.0),
            slope_threshold = trial.suggest_float("regime_slope",   0.01, 0.20, step=0.01),
        ),
        adx_min          = trial.suggest_float("adx_min",         18.0, 40.0, step=1.0),
        rsi_min          = trial.suggest_float("rsi_min",         25.0, 50.0, step=5.0),
        rsi_max          = trial.suggest_float("rsi_max",         50.0, 75.0, step=5.0),
        sl_atr_mult      = trial.suggest_float("sl_atr_mult",      0.8,  3.0, step=0.1),
        tp_atr_mult      = trial.suggest_float("tp_atr_mult",      1.5,  6.0, step=0.1),
        trail_start_mult = trial.suggest_float("trail_start",      1.5,  4.0, step=0.1),
        trail_dist_mult  = trial.suggest_float("trail_dist",       0.2,  1.5, step=0.1),
        partial_close_r  = trial.suggest_float("partial_close",    0.5,  2.5, step=0.25),
        use_session_filter = trial.suggest_categorical("session_filter", [True, False]),
    )
    result = run_backtest_advanced(df, AdaptiveScalp15m(params), BROKER)
    if result.total_trades < 8:
        return float("-inf")
    # Penalizar si MaxDD > 15%
    if result.max_drawdown_pct > 15.0:
        return float("-inf")
    score = result.calmar
    return float(score) if score > -10 else float("-inf")


def optimise(df: pd.DataFrame, label: str, n_trials: int) -> AdaptiveParams:
    print(f"\n  Optimizando {label} — {n_trials} trials…")
    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: _objective(t, df), n_trials=n_trials, show_progress_bar=True)
    b = study.best_params
    print(f"\n  Mejor Calmar {label}: {study.best_value:.3f}")
    for k, v in b.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
    return AdaptiveParams(
        regime = RegimeParams(
            adx_trend_min   = b["regime_adx"],
            slope_threshold = b["regime_slope"],
        ),
        adx_min          = b["adx_min"],
        rsi_min          = b["rsi_min"],
        rsi_max          = b["rsi_max"],
        sl_atr_mult      = b["sl_atr_mult"],
        tp_atr_mult      = b["tp_atr_mult"],
        trail_start_mult = b["trail_start"],
        trail_dist_mult  = b["trail_dist"],
        partial_close_r  = b["partial_close"],
        use_session_filter = b.get("session_filter", True),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--trials",   type=int, default=200)
    args = parser.parse_args()

    print("Conectando a Capital.com…")
    hdrs = _open_session()

    print(f"\nDescargando {MONTHS} meses de datos reales GOLD…")
    dfs = download_all(hdrs)

    timeframes = [
        ("15m", dfs["15m"], dfs["H1"],    False),
        ("H1",  dfs["H1"],  dfs["H4"],    False),
        ("H4",  dfs["H4"],  dfs["Daily"], True),
    ]

    for label, entry_df, trend_df, disable_session in timeframes:
        print(f"\n{'#'*62}")
        print(f"  TIMEFRAME: {label}")
        print(f"{'#'*62}")

        df = build_df(entry_df, trend_df, disable_session)

        if args.optimize:
            params = optimise(df, label, args.trials)
        else:
            params = AdaptiveParams(
                use_session_filter = not disable_session,
            )

        strategy = AdaptiveScalp15m(params)
        result   = run_backtest_advanced(df, strategy, BROKER)

        _print_summary(result, f"AdaptiveScalp {label} — Capital.com ({MONTHS}m)")
        _print_monthly(result.trades)
        if not result.trades.empty:
            print("\nSalidas:")
            print(result.trades["exit_reason"].value_counts().to_string())
            _print_direction(result.trades)


if __name__ == "__main__":
    main()
