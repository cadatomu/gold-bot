"""Optuna-based hyperparameter optimiser for TrendATR.

Optimisation target: maximise Calmar ratio on an in-sample window.
Walk-forward is used separately for OOS validation — do NOT optimise
on the full history to avoid overfitting.

Tunable parameters (ranges from strategy_params.yaml comments):
  - sl_atr_mult       [1.0, 2.5]
  - tp_atr_mult       [2.0, 5.0]
  - atr_percentile_min[10,  40 ]
  - trailing_activate_r[1.0, 2.5]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import optuna
import pandas as pd

from src.backtest.runner import BrokerConfig, run_backtest
from src.strategy.trend_atr import TrendATR, TrendATRParams

# Suppress Optuna's verbose per-trial output by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimiserConfig:
    n_trials:          int   = 100
    objective:         str   = "calmar"   # "calmar" | "sharpe" | "total_return_pct"
    min_trades:        int   = 10         # discard trials with fewer trades
    timeout_seconds:   Optional[float] = None
    n_jobs:            int   = 1          # parallel trials (1 = sequential)
    seed:              int   = 42


@dataclass
class OptimisationResult:
    best_params:  TrendATRParams
    best_value:   float
    study:        optuna.Study
    n_trials:     int


def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    broker: BrokerConfig,
    cfg: OptimiserConfig,
) -> float:
    params = TrendATRParams(
        sl_atr_mult         = trial.suggest_float("sl_atr_mult",          1.0,  3.0, step=0.1),
        tp_atr_mult         = trial.suggest_float("tp_atr_mult",          2.0,  6.0, step=0.1),
        atr_percentile_min  = trial.suggest_float("atr_percentile_min",  20.0, 60.0, step=5.0),
        adx_min_threshold   = trial.suggest_float("adx_min_threshold",   15.0, 35.0, step=1.0),
        trailing_activate_r = trial.suggest_float("trailing_activate_r",  1.0,  2.5, step=0.1),
    )
    result = run_backtest(df, TrendATR(params), broker)

    if result.total_trades < cfg.min_trades:
        return float("-inf")

    value = getattr(result, cfg.objective, None)
    if value is None:
        raise ValueError(f"Unknown objective: {cfg.objective}")

    # For drawdown-based objectives lower is worse; Calmar already handles sign
    return float(value)


def run_optimisation(
    df:     pd.DataFrame,
    broker: BrokerConfig | None = None,
    cfg:    OptimiserConfig | None = None,
) -> OptimisationResult:
    """
    Run Optuna optimisation on `df` (should be in-sample data only).

    Parameters
    ----------
    df     : In-sample OHLCV DataFrame
    broker : BrokerConfig (defaults if None)
    cfg    : OptimiserConfig (defaults if None)

    Returns
    -------
    OptimisationResult with best TrendATRParams and the full study object
    """
    if broker is None:
        broker = BrokerConfig()
    if cfg is None:
        cfg = OptimiserConfig()

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study   = optuna.create_study(
        direction = "maximize",
        sampler   = sampler,
    )

    study.optimize(
        lambda trial: _objective(trial, df, broker, cfg),
        n_trials         = cfg.n_trials,
        timeout          = cfg.timeout_seconds,
        n_jobs           = cfg.n_jobs,
        show_progress_bar= False,
    )

    best = study.best_params
    best_params = TrendATRParams(
        sl_atr_mult         = best["sl_atr_mult"],
        tp_atr_mult         = best["tp_atr_mult"],
        atr_percentile_min  = best["atr_percentile_min"],
        adx_min_threshold   = best["adx_min_threshold"],
        trailing_activate_r = best["trailing_activate_r"],
    )

    return OptimisationResult(
        best_params = best_params,
        best_value  = study.best_value,
        study       = study,
        n_trials    = len(study.trials),
    )
