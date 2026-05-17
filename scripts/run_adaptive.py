"""Backtest + optimización para AdaptiveScalp15m v2 (sin lateral).

Uso:
  python scripts/run_adaptive.py
  python scripts/run_adaptive.py --optimize --trials 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.runner import BrokerConfig
from src.backtest.runner_advanced import run_backtest_advanced
from src.strategy.adaptive import AdaptiveParams, AdaptiveScalp15m
from src.strategy.regime import RegimeParams
from src.strategy.scalp_15m import build_h1_trend

optuna.logging.set_verbosity(optuna.logging.WARNING)

GOLD_TICKER = "GC=F"

BROKER = BrokerConfig(
    spread_pips           = 0.45,
    commission_per_lot_rt = 0.0,
    slippage_pips         = 0.30,
    risk_per_trade_pct    = 0.005,
    initial_equity        = 10_000.0,
)


# ── Data ─────────────────────────────────────────────────────────────────────

def _download(interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(GOLD_TICKER, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
    return raw[["open", "high", "low", "close", "volume"]].dropna()


def load_data() -> pd.DataFrame:
    print("Descargando 15m (60d) y H1 (730d)…")
    df_15m = _download("15m", "60d")
    df_1h  = _download("1h",  "730d")
    h1     = build_h1_trend(df_1h)
    h1.index = h1.index.tz_localize("UTC") if h1.index.tzinfo is None else h1.index
    df = df_15m.copy()
    df["h1_trend"] = h1.reindex(df_15m.index, method="ffill").fillna(0).astype(int)
    print(f"  {len(df)} barras  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(result, title: str = "AdaptiveScalp 15m v2"):
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


def _print_regime_stats(signals: pd.DataFrame):
    if "regime" not in signals.columns:
        return
    counts = signals["regime"].value_counts()
    total  = len(signals)
    print("\nDistribución de régimen:")
    for reg, cnt in counts.items():
        print(f"  {reg:<8}: {cnt:>5} barras ({cnt/total*100:.1f}%)")


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
            adx_trend_min   = trial.suggest_float("regime_adx",       20.0, 35.0, step=1.0),
            slope_threshold = trial.suggest_float("regime_slope",      0.01, 0.15, step=0.01),
        ),
        adx_min             = trial.suggest_float("adx_min",          20.0, 35.0, step=1.0),
        rsi_min             = trial.suggest_float("rsi_min",          25.0, 45.0, step=5.0),
        rsi_max             = trial.suggest_float("rsi_max",          55.0, 75.0, step=5.0),
        sl_atr_mult         = trial.suggest_float("sl_atr_mult",       1.0,  3.0, step=0.1),
        tp_atr_mult         = trial.suggest_float("tp_atr_mult",       2.0,  6.0, step=0.1),
        trail_start_mult    = trial.suggest_float("trail_start",       1.5,  3.5, step=0.1),
        trail_dist_mult     = trial.suggest_float("trail_dist",        0.2,  1.5, step=0.1),
        partial_close_r     = trial.suggest_float("partial_close",     0.5,  2.0, step=0.25),
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
    study.optimize(
        lambda t: _objective(t, df),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    b = study.best_params
    print(f"\nMejor Calmar: {study.best_value:.3f}")
    print("Parámetros:")
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

    df = load_data()

    if args.optimize:
        params = run_optimisation(df, n_trials=args.trials)
    else:
        params = AdaptiveParams()

    strategy = AdaptiveScalp15m(params)
    result   = run_backtest_advanced(df, strategy, BROKER)
    signals  = strategy.generate_signals(df)

    _print_summary(result)
    _print_monthly(result.trades)

    if not result.trades.empty:
        print("\nSalidas por razón:")
        print(result.trades["exit_reason"].value_counts().to_string())
        _print_direction_breakdown(result.trades)

    _print_regime_stats(signals)


if __name__ == "__main__":
    main()
