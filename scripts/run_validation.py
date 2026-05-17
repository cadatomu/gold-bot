#!/usr/bin/env python3
"""FASE 5 — Full validation script.

Runs:
  1. Full backtest on all available data
  2. Walk-forward (rolling, 8-month train / 2-month test)
  3. Stress test suite
  4. Optuna optimisation on first 70% of data
  5. HTML reports saved to results/

Usage (with live MT5):
    python scripts/run_validation.py --login 12345 --password pw --server Demo

Usage (with synthetic data for offline validation):
    python scripts/run_validation.py --synthetic --bars 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.optimizer import OptimiserConfig, run_optimisation
from src.backtest.report import (
    print_stress_summary,
    print_summary,
    save_html_report,
    save_wf_html_report,
)
from src.backtest.runner import BrokerConfig, run_backtest
from src.backtest.stress_test import run_stress_suite
from src.backtest.walk_forward import run_walk_forward
from src.monitoring.logger import configure_logging
from src.strategy.trend_atr import TrendATR, TrendATRParams

RESULTS_DIR = Path("results")


def make_synthetic_df(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    closes = 1800.0 + np.cumsum(rng.normal(0.5, 3, n))
    opens  = closes - rng.uniform(-2, 2, n)
    highs  = np.maximum(opens, closes) + rng.uniform(0, 5, n)
    lows   = np.minimum(opens, closes) - rng.uniform(0, 5, n)
    idx    = pd.date_range("2015-01-05 00:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": rng.integers(200, 800, n)},
        index=idx,
    )


def load_mt5_data(args: argparse.Namespace) -> pd.DataFrame:
    from src.data.historical import download_history
    from src.data.mt5_connector import MT5Connector
    with MT5Connector(
        login         = args.login,
        password      = args.password,
        server        = args.server,
        terminal_path = args.terminal_path or None,
    ) as conn:
        return download_history(
            conn, "XAUUSD", MT5Connector.TIMEFRAME_H4,
            years=10, cache_dir=Path("data/cache"),
        )


def run(df: pd.DataFrame, n_optuna_trials: int = 50) -> None:
    configure_logging(level="INFO", fmt="console")
    RESULTS_DIR.mkdir(exist_ok=True)

    broker = BrokerConfig(initial_equity=10_000.0)
    bars   = len(df)
    print(f"\n[DATA] {bars} H4 bars  ({df.index[0].date()} → {df.index[-1].date()})")

    # ------------------------------------------------------------------ #
    # 1. Full backtest (default params)
    # ------------------------------------------------------------------ #
    print("\n[1/4] Full backtest (default params) …")
    strategy = TrendATR()
    result   = run_backtest(df, strategy, broker)
    print_summary(result)
    save_html_report(result, RESULTS_DIR / "full_backtest.html",
                     title="TrendATR XAUUSD H4 — Full History")
    print(f"      → results/full_backtest.html")

    # ------------------------------------------------------------------ #
    # 2. Walk-forward
    # ------------------------------------------------------------------ #
    print("[2/4] Walk-forward (8-month train / 2-month test, rolling) …")
    train_bars = int(bars * 0.4)
    test_bars  = max(252, int(bars * 0.1))
    wf = run_walk_forward(df, strategy,
                          train_bars=train_bars,
                          test_bars=test_bars,
                          anchored=False,
                          broker=broker)
    print(f"      Windows: {len(wf.windows)}  |  OOS trades: {wf.oos_total_trades}")
    print(f"      OOS return:  {wf.oos_total_return:+.2f}%")
    print(f"      OOS Calmar:  {wf.oos_calmar:.3f}")
    print(f"      OOS Max DD:  {wf.oos_max_drawdown:.2f}%")
    save_wf_html_report(wf, RESULTS_DIR / "walk_forward.html")
    print(f"      → results/walk_forward.html")

    # ------------------------------------------------------------------ #
    # 3. Stress tests
    # ------------------------------------------------------------------ #
    print("[3/4] Stress tests …")
    suite = run_stress_suite(df, strategy, broker)
    print_stress_summary(suite)

    # ------------------------------------------------------------------ #
    # 4. Optuna optimisation (in-sample = first 70%)
    # ------------------------------------------------------------------ #
    cutoff    = int(bars * 0.70)
    df_train  = df.iloc[:cutoff]
    print(f"[4/4] Optuna optimisation ({n_optuna_trials} trials, "
          f"in-sample {df_train.index[0].date()} → {df_train.index[-1].date()}) …")

    opt_cfg = OptimiserConfig(n_trials=n_optuna_trials, objective="calmar", min_trades=5)
    opt     = run_optimisation(df_train, broker, opt_cfg)
    p       = opt.best_params
    print(f"\n  Best Calmar (IS):   {opt.best_value:.3f}")
    print(f"  sl_atr_mult:        {p.sl_atr_mult}")
    print(f"  tp_atr_mult:        {p.tp_atr_mult}")
    print(f"  atr_percentile_min: {p.atr_percentile_min}")
    print(f"  trailing_activate_r:{p.trailing_activate_r}")

    # OOS validation with optimised params
    df_test    = df.iloc[cutoff:]
    opt_result = run_backtest(df_test, TrendATR(p), broker)
    print(f"\n  OOS validation ({df_test.index[0].date()} → {df_test.index[-1].date()}):")
    print_summary(opt_result)
    save_html_report(opt_result, RESULTS_DIR / "optimised_oos.html",
                     title="TrendATR Optimised — OOS Validation")
    print(f"  → results/optimised_oos.html")

    # ------------------------------------------------------------------ #
    # Pass/Fail summary
    # ------------------------------------------------------------------ #
    print("\n" + "═" * 50)
    print("  VALIDATION TARGETS")
    print("═" * 50)
    checks = [
        ("CAGR > 15%",      result.cagr_pct > 15,      f"{result.cagr_pct:.1f}%"),
        ("Max DD ≤ 15%",    result.max_drawdown_pct >= -15, f"{result.max_drawdown_pct:.1f}%"),
        ("Calmar > 1.0",    result.calmar > 1.0,       f"{result.calmar:.3f}"),
        ("Sharpe > 1.0",    result.sharpe > 1.0,       f"{result.sharpe:.3f}"),
        ("Win rate > 45%",  result.win_rate_pct > 45,  f"{result.win_rate_pct:.1f}%"),
        ("OOS WF Calmar > 0.5", wf.oos_calmar > 0.5,  f"{wf.oos_calmar:.3f}"),
    ]
    all_pass = True
    for label, passed, val in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {label:<28}  {val}")
        if not passed:
            all_pass = False
    print("═" * 50)
    print(f"  {'ALL TARGETS MET' if all_pass else 'SOME TARGETS MISSED'}")
    print("═" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FASE 5 full validation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no MT5 required)")
    parser.add_argument("--bars", type=int, default=4000,
                        help="Number of synthetic bars (default 4000 ≈ 2.5 years H4)")
    parser.add_argument("--login",         type=int,   default=0)
    parser.add_argument("--password",      type=str,   default="")
    parser.add_argument("--server",        type=str,   default="")
    parser.add_argument("--terminal-path", type=str,   default="")
    parser.add_argument("--trials",        type=int,   default=50,
                        help="Optuna trials (default 50)")
    args = parser.parse_args()

    if args.synthetic:
        df = make_synthetic_df(n=args.bars)
    else:
        if not args.login or not args.password or not args.server:
            parser.error("--login, --password, --server required without --synthetic")
        df = load_mt5_data(args)

    run(df, n_optuna_trials=args.trials)


if __name__ == "__main__":
    main()
