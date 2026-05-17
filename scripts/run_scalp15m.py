"""Backtest + optimización para ScalpATR15m v2 (sesión + H1 + cierre parcial).

Uso:
  python scripts/run_scalp15m.py                  # backtest con params por defecto
  python scripts/run_scalp15m.py --optimize       # optimización Optuna (50 trials)
  python scripts/run_scalp15m.py --optimize --trials 200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yfinance as yf

# ── project root en sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.runner import BrokerConfig
from src.backtest.runner_advanced import run_backtest_advanced
from src.strategy.scalp_15m import ScalpATR15m, Scalp15mParams, build_h1_trend

optuna.logging.set_verbosity(optuna.logging.WARNING)

GOLD_TICKER = "GC=F"
# OANDA Europe — XAUUSD Standard account
# Sin comisión por operación; spread promedio ~$0.45/oz en horas activas
# Slippage conservador ~$0.30 para entradas en 15m
BROKER = BrokerConfig(
    spread_pips            = 0.45,   # $0.45 spread promedio OANDA XAUUSD
    commission_per_lot_rt  = 0.0,    # sin comisión en cuenta estándar OANDA
    slippage_pips          = 0.30,   # slippage conservador 15m
    risk_per_trade_pct     = 0.005,  # 0.5% equity por trade → apalancamiento efectivo ~0.3x
    initial_equity         = 10_000.0,
)


# ── Data helpers ─────────────────────────────────────────────────────────────

def _download(interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(GOLD_TICKER, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
    return raw[["open", "high", "low", "close", "volume"]].dropna()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_15m, df_1h). Yahoo caps 15m at ~60 days."""
    print("Descargando datos 15m (últimos 60d)…")
    df_15m = _download("15m", "60d")
    print(f"  15m: {len(df_15m)} barras  {df_15m.index[0].date()} → {df_15m.index[-1].date()}")

    print("Descargando datos H1 (últimos 730d)…")
    df_1h = _download("1h", "730d")
    print(f"  H1 : {len(df_1h)} barras  {df_1h.index[0].date()} → {df_1h.index[-1].date()}")
    return df_15m, df_1h


def merge_h1_trend(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columna h1_trend al DataFrame 15m.
    Se usa forward-fill para alinear la señal H1 a cada barra 15m.
    """
    h1_trend = build_h1_trend(df_1h)
    h1_trend.index = pd.DatetimeIndex(h1_trend.index).tz_localize("UTC") \
        if h1_trend.index.tzinfo is None else h1_trend.index
    trend_15m = h1_trend.reindex(df_15m.index, method="ffill")
    df = df_15m.copy()
    df["h1_trend"] = trend_15m.fillna(0).astype(int)
    return df


# ── Monthly breakdown ─────────────────────────────────────────────────────────

def _monthly_table(trades_df: pd.DataFrame, equity_curve: pd.Series) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    td = trades_df[trades_df["exit_reason"] != "PARTIAL"].copy()
    td["month"] = pd.to_datetime(td["exit_time"]).dt.to_period("M")
    monthly = td.groupby("month").agg(
        trades      = ("pnl_usd", "count"),
        pnl_usd     = ("pnl_usd", "sum"),
        win_rate    = ("pnl_usd", lambda x: (x > 0).mean() * 100),
        avg_trade   = ("pnl_usd", "mean"),
        best_trade  = ("pnl_usd", "max"),
        worst_trade = ("pnl_usd", "min"),
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly["pnl_usd"]    = monthly["pnl_usd"].round(2)
    monthly["win_rate"]   = monthly["win_rate"].round(1)
    monthly["avg_trade"]  = monthly["avg_trade"].round(2)
    monthly["best_trade"] = monthly["best_trade"].round(2)
    monthly["worst_trade"]= monthly["worst_trade"].round(2)
    return monthly


def _print_summary(result, title: str = "ScalpATR 15m v2"):
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")
    print(f"  Total return : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe       : {result.sharpe:.3f}")
    print(f"  Calmar       : {result.calmar:.3f}")
    print(f"  Max DD       : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate     : {result.win_rate_pct:.1f}%")
    print(f"  Trades       : {result.total_trades}")
    print(f"  Profit factor: {result.profit_factor:.2f}")


def _print_monthly(monthly: pd.DataFrame):
    if monthly.empty:
        print("  (sin trades)")
        return
    print(f"\n{'─'*84}")
    print(f"  {'Mes':<10} {'Trades':>6} {'PnL USD':>10} {'Win%':>7} {'Avg':>9} {'Best':>10} {'Worst':>10}")
    print(f"{'─'*84}")
    for _, r in monthly.iterrows():
        print(f"  {r['month']:<10} {r['trades']:>6} {r['pnl_usd']:>10.2f} "
              f"{r['win_rate']:>7.1f} {r['avg_trade']:>9.2f} "
              f"{r['best_trade']:>10.2f} {r['worst_trade']:>10.2f}")
    print(f"{'─'*84}")
    total_pnl = monthly["pnl_usd"].sum()
    total_tr  = monthly["trades"].sum()
    print(f"  {'TOTAL':<10} {total_tr:>6} {total_pnl:>10.2f}")


# ── Optimisation ─────────────────────────────────────────────────────────────

def _objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    params = Scalp15mParams(
        sl_atr_mult       = trial.suggest_float("sl_atr_mult",       1.0, 3.0, step=0.1),
        tp_atr_mult       = trial.suggest_float("tp_atr_mult",       2.0, 6.0, step=0.1),
        adx_min           = trial.suggest_float("adx_min",          15.0, 35.0, step=1.0),
        rsi_min           = trial.suggest_float("rsi_min",          25.0, 45.0, step=5.0),
        rsi_max           = trial.suggest_float("rsi_max",          55.0, 75.0, step=5.0),
        trail_start_mult  = trial.suggest_float("trail_start_mult",  1.5,  3.5, step=0.1),
        trail_dist_mult   = trial.suggest_float("trail_dist_mult",   0.3,  1.5, step=0.1),
        partial_close_r   = trial.suggest_float("partial_close_r",   0.5,  2.0, step=0.25),
        use_session_filter= True,
        use_h1_filter     = True,
    )
    result = run_backtest_advanced(df, ScalpATR15m(params), BROKER)
    if result.total_trades < 5:
        return float("-inf")
    if result.calmar == 0:
        return float("-inf")
    return float(result.calmar)


def run_optimisation(df: pd.DataFrame, n_trials: int = 50) -> Scalp15mParams:
    print(f"\nOptimizando {n_trials} trials (objetivo: Calmar)…")
    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: _objective(t, df),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"\nMejores params encontrados (Calmar={study.best_value:.3f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return Scalp15mParams(
        sl_atr_mult       = best["sl_atr_mult"],
        tp_atr_mult       = best["tp_atr_mult"],
        adx_min           = best["adx_min"],
        rsi_min           = best["rsi_min"],
        rsi_max           = best["rsi_max"],
        trail_start_mult  = best["trail_start_mult"],
        trail_dist_mult   = best["trail_dist_mult"],
        partial_close_r   = best["partial_close_r"],
        use_session_filter= True,
        use_h1_filter     = True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize",  action="store_true")
    parser.add_argument("--trials",    type=int, default=50)
    parser.add_argument("--no-h1",     action="store_true", help="Desactiva filtro H1")
    parser.add_argument("--no-session",action="store_true", help="Desactiva filtro sesión")
    args = parser.parse_args()

    df_15m, df_1h = load_data()
    df = merge_h1_trend(df_15m, df_1h)
    print(f"\nh1_trend cobertura: {df['h1_trend'].notna().sum()}/{len(df)} barras")

    if args.optimize:
        params = run_optimisation(df, n_trials=args.trials)
    else:
        params = Scalp15mParams(
            use_session_filter = not args.no_session,
            use_h1_filter      = not args.no_h1,
        )

    strategy = ScalpATR15m(params)
    result   = run_backtest_advanced(df, strategy, BROKER)

    _print_summary(result)
    monthly = _monthly_table(result.trades, result.equity_curve)
    _print_monthly(monthly)

    # Breakdown de razón de salida y dirección
    if not result.trades.empty:
        print("\nSalidas por razón:")
        print(result.trades["exit_reason"].value_counts().to_string())
        if "direction" in result.trades.columns:
            print("\nTrades por dirección:")
            dir_grp = result.trades[result.trades["exit_reason"] != "PARTIAL"].groupby("direction").agg(
                trades   = ("pnl_usd", "count"),
                pnl_usd  = ("pnl_usd", "sum"),
                win_rate = ("pnl_usd", lambda x: f"{(x > 0).mean()*100:.1f}%"),
            )
            print(dir_grp.to_string())

    # Comparativa: con/sin filtros
    print("\n--- Comparativa de filtros ---")
    for label, sess, h1 in [
        ("Sin filtros",        False, False),
        ("Solo sesión",        True,  False),
        ("Solo H1",            False, True),
        ("Sesión + H1 (v2)",   True,  True),
    ]:
        p = Scalp15mParams(use_session_filter=sess, use_h1_filter=h1)
        r = run_backtest_advanced(df, ScalpATR15m(p), BROKER)
        print(f"  {label:<22}: ret={r.total_return_pct:+6.2f}%  "
              f"Calmar={r.calmar:5.3f}  trades={r.total_trades}  "
              f"win={r.win_rate_pct:.1f}%")


if __name__ == "__main__":
    main()
