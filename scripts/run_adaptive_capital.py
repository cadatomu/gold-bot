"""Backtest de AdaptiveScalp15m con datos reales de Capital.com.

Solo descarga datos — NO abre órdenes.

Uso:
  python scripts/run_adaptive_capital.py
  python scripts/run_adaptive_capital.py --optimize --trials 200
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
    spread_pips           = 0.50,   # spread real visto: 0.50
    commission_per_lot_rt = 0.0,    # Capital.com = spread only
    slippage_pips         = 0.20,
    risk_per_trade_pct    = 0.005,
    initial_equity        = 10_000.0,
)


# ── Sesión Capital.com ────────────────────────────────────────────────────────

def _open_session() -> dict:
    resp = requests.post(
        f"{BASE_URL}/session",
        json={
            "identifier":        os.environ["CAPITAL_EMAIL"],
            "password":          os.environ["CAPITAL_PASSWORD"],
            "encryptedPassword": False,
        },
        headers={"X-CAP-API-KEY": os.environ["CAPITAL_API_KEY"],
                 "Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return {
        "X-CAP-API-KEY":    os.environ["CAPITAL_API_KEY"],
        "CST":              resp.headers["CST"],
        "X-SECURITY-TOKEN": resp.headers["X-SECURITY-TOKEN"],
        "Content-Type":     "application/json",
    }


def _fetch_chunk(hdrs: dict, resolution: str, from_dt: datetime, to_dt: datetime) -> list:
    r = requests.get(
        f"{BASE_URL}/prices/GOLD",
        params={
            "resolution": resolution,
            "max":        1000,
            "from":       from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "to":         to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        headers=hdrs,
        timeout=15,
    )
    if r.status_code != 200:
        return []
    return r.json().get("prices", [])


def _prices_to_df(prices: list) -> pd.DataFrame:
    rows = []
    for p in prices:
        mid = lambda side: (side["bid"] + side["ask"]) / 2
        rows.append({
            "time":   p["snapshotTimeUTC"],
            "open":   mid(p["openPrice"]),
            "high":   mid(p["highPrice"]),
            "low":    mid(p["lowPrice"]),
            "close":  mid(p["closePrice"]),
            "volume": p.get("lastTradedVolume", 0),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── Descarga de datos ─────────────────────────────────────────────────────────

def download_capital_data(days_15m: int = 60, days_h1: int = 730) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Conectando a Capital.com…")
    hdrs = _open_session()
    print("Sesión abierta.")

    # ── 15m ──────────────────────────────────────────────────────────────────
    print(f"Descargando 15m ({days_15m} días)…")
    all_15m: list = []
    to_dt = datetime.now(timezone.utc)
    chunk_days = 10  # ~960 velas de 15m por chunk

    for i in range((days_15m // chunk_days) + 2):
        from_dt = to_dt - timedelta(days=chunk_days)
        chunk = _fetch_chunk(hdrs, "MINUTE_15", from_dt, to_dt)
        if not chunk:
            break
        all_15m = chunk + all_15m
        to_dt = from_dt
        time.sleep(0.3)
        if (to_dt) < (datetime.now(timezone.utc) - timedelta(days=days_15m)):
            break

    df_15m = _prices_to_df(all_15m)
    cutoff_15m = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_15m)
    df_15m = df_15m[df_15m.index >= cutoff_15m]
    print(f"  {len(df_15m)} barras 15m  {df_15m.index[0].date()} → {df_15m.index[-1].date()}")

    # ── H1 ───────────────────────────────────────────────────────────────────
    print(f"Descargando H1 ({days_h1} días)…")
    all_h1: list = []
    to_dt = datetime.now(timezone.utc)
    chunk_days_h1 = 40

    for i in range((days_h1 // chunk_days_h1) + 2):
        from_dt = to_dt - timedelta(days=chunk_days_h1)
        chunk = _fetch_chunk(hdrs, "HOUR", from_dt, to_dt)
        if not chunk:
            break
        all_h1 = chunk + all_h1
        to_dt = from_dt
        time.sleep(0.3)
        if to_dt < (datetime.now(timezone.utc) - timedelta(days=days_h1)):
            break

    df_h1 = _prices_to_df(all_h1)
    cutoff_h1 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_h1)
    df_h1 = df_h1[df_h1.index >= cutoff_h1]
    print(f"  {len(df_h1)} barras H1   {df_h1.index[0].date()} → {df_h1.index[-1].date()}")

    return df_15m, df_h1


def merge_data(df_15m: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    h1 = build_h1_trend(df_h1)
    h1.index = h1.index.tz_localize("UTC") if h1.index.tzinfo is None else h1.index
    df = df_15m.copy()
    df["h1_trend"] = h1.reindex(df_15m.index, method="ffill").fillna(0).astype(int)
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(result, title: str = "AdaptiveScalp 15m — Capital.com data"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total return  : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe        : {result.sharpe:.3f}")
    print(f"  Calmar        : {result.calmar:.3f}")
    print(f"  Max DD        : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate      : {result.win_rate_pct:.1f}%")
    print(f"  Total trades  : {result.total_trades}")
    print(f"  Profit factor : {result.profit_factor:.2f}")


def _print_monthly(trades_df: pd.DataFrame):
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
    print(f"\n{'─'*88}")
    print(f"  {'Mes':<10} {'Trades':>6} {'PnL USD':>10} {'Win%':>7} {'Avg':>9} {'Mejor':>10} {'Peor':>10}")
    print(f"{'─'*88}")
    for _, r in m.iterrows():
        print(f"  {r['month']:<10} {r['trades']:>6} {r['pnl_usd']:>10.2f} "
              f"{r['win_pct']:>7.1f} {r['avg']:>9.2f} {r['mejor']:>10.2f} {r['peor']:>10.2f}")
    print(f"{'─'*88}")
    print(f"  {'TOTAL':<10} {m['trades'].sum():>6} {m['pnl_usd'].sum():>10.2f}")


def _print_direction_breakdown(trades_df: pd.DataFrame):
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
            adx_trend_min   = trial.suggest_float("regime_adx",    20.0, 35.0, step=1.0),
            slope_threshold = trial.suggest_float("regime_slope",   0.01, 0.15, step=0.01),
        ),
        adx_min          = trial.suggest_float("adx_min",         20.0, 35.0, step=1.0),
        rsi_min          = trial.suggest_float("rsi_min",         25.0, 45.0, step=5.0),
        rsi_max          = trial.suggest_float("rsi_max",         55.0, 75.0, step=5.0),
        sl_atr_mult      = trial.suggest_float("sl_atr_mult",      1.0,  3.0, step=0.1),
        tp_atr_mult      = trial.suggest_float("tp_atr_mult",      2.0,  6.0, step=0.1),
        trail_start_mult = trial.suggest_float("trail_start",      1.5,  3.5, step=0.1),
        trail_dist_mult  = trial.suggest_float("trail_dist",       0.2,  1.5, step=0.1),
        partial_close_r  = trial.suggest_float("partial_close",    0.5,  2.0, step=0.25),
    )
    result = run_backtest_advanced(df, AdaptiveScalp15m(params), BROKER)
    if result.total_trades < 5:
        return float("-inf")
    return float(result.calmar) if result.calmar > -10 else float("-inf")


def run_optimisation(df: pd.DataFrame, n_trials: int = 100) -> AdaptiveParams:
    print(f"\nOptimizando {n_trials} trials…")
    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: _objective(t, df), n_trials=n_trials, show_progress_bar=True)
    b = study.best_params
    print(f"\nMejor Calmar: {study.best_value:.3f}")
    for k, v in b.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
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
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--trials",   type=int, default=100)
    args = parser.parse_args()

    df_15m, df_h1 = download_capital_data(days_15m=60, days_h1=730)
    df = merge_data(df_15m, df_h1)

    if args.optimize:
        params = run_optimisation(df, n_trials=args.trials)
    else:
        params = AdaptiveParams()

    strategy = AdaptiveScalp15m(params)
    result   = run_backtest_advanced(df, strategy, BROKER)

    _print_summary(result)
    _print_monthly(result.trades)

    if not result.trades.empty:
        print("\nSalidas por razón:")
        print(result.trades["exit_reason"].value_counts().to_string())
        _print_direction_breakdown(result.trades)


if __name__ == "__main__":
    main()
