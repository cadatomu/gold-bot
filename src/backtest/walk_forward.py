"""Walk-forward analysis for strategy validation.

Splits data into anchored or rolling train/test windows,
runs the backtest on each, and aggregates out-of-sample metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.backtest.runner import BacktestResult, BrokerConfig, run_backtest
from src.strategy.base import Strategy


@dataclass
class WFWindow:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    result:      Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    windows:           List[WFWindow]
    oos_equity:        pd.Series        # concatenated out-of-sample equity curve
    oos_total_return:  float
    oos_sharpe:        float
    oos_calmar:        float
    oos_max_drawdown:  float
    oos_win_rate:      float
    oos_profit_factor: float
    oos_total_trades:  int


def _split_windows(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    anchored: bool,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate (train_df, test_df) slices."""
    windows = []
    n = len(df)
    start = 0
    while start + train_bars + test_bars <= n:
        train_slice = df.iloc[start: start + train_bars]
        test_slice  = df.iloc[start + train_bars: start + train_bars + test_bars]
        windows.append((train_slice, test_slice))
        if anchored:
            start = 0
            train_bars += test_bars  # expand training window
            start = train_bars       # next train ends where last test did
            # reset: anchored means train always starts at 0
            # recalculate properly
            break  # anchored: rebuild below
        else:
            start += test_bars
    return windows


def _anchored_windows(
    df: pd.DataFrame,
    initial_train_bars: int,
    test_bars: int,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    windows = []
    n = len(df)
    train_end = initial_train_bars
    while train_end + test_bars <= n:
        train_slice = df.iloc[0:train_end]
        test_slice  = df.iloc[train_end: train_end + test_bars]
        windows.append((train_slice, test_slice))
        train_end += test_bars
    return windows


def _rolling_windows(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    windows = []
    n = len(df)
    start = 0
    while start + train_bars + test_bars <= n:
        train_slice = df.iloc[start: start + train_bars]
        test_slice  = df.iloc[start + train_bars: start + train_bars + test_bars]
        windows.append((train_slice, test_slice))
        start += test_bars
    return windows


def run_walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    train_bars: int = 1008,    # ~8 months of H4 bars
    test_bars: int = 252,      # ~2 months of H4 bars
    anchored: bool = False,
    broker: Optional[BrokerConfig] = None,
) -> WalkForwardResult:
    """
    Run walk-forward validation.

    Parameters
    ----------
    df         : Full OHLCV history (UTC DatetimeIndex)
    strategy   : Strategy instance (params fixed — no in-sample tuning here)
    train_bars : Number of bars in each training window
    test_bars  : Number of bars in each test (out-of-sample) window
    anchored   : True = expanding window; False = rolling window
    broker     : BrokerConfig

    Returns
    -------
    WalkForwardResult with per-window results and aggregated OOS metrics
    """
    if broker is None:
        broker = BrokerConfig()

    slices = (
        _anchored_windows(df, train_bars, test_bars)
        if anchored
        else _rolling_windows(df, train_bars, test_bars)
    )

    if not slices:
        raise ValueError(
            f"Not enough data for walk-forward: {len(df)} bars, "
            f"need at least {train_bars + test_bars}"
        )

    wf_windows: List[WFWindow] = []
    oos_equity_parts: List[pd.Series] = []

    for train_df, test_df in slices:
        # Strategy is evaluated on the test window only (OOS)
        # For proper Optuna-based WF, optimise on train_df here — stub for now
        result = run_backtest(test_df, strategy, broker)

        window = WFWindow(
            train_start = train_df.index[0],
            train_end   = train_df.index[-1],
            test_start  = test_df.index[0],
            test_end    = test_df.index[-1],
            result      = result,
        )
        wf_windows.append(window)
        oos_equity_parts.append(result.equity_curve)

    # Stitch OOS equity curves (each segment starts from previous end value)
    oos_eq = _stitch_equity(oos_equity_parts, broker.initial_equity)

    # Aggregate OOS metrics
    all_trades = pd.concat(
        [w.result.trades for w in wf_windows if w.result and len(w.result.trades) > 0],
        ignore_index=True,
    ) if any(len(w.result.trades) > 0 for w in wf_windows if w.result) else pd.DataFrame()

    initial  = broker.initial_equity
    final    = oos_eq.iloc[-1]
    ret      = (final / initial - 1) * 100

    rets = oos_eq.pct_change().dropna()
    bars_per_year = 1512
    sharpe = (rets.mean() / rets.std(ddof=1)) * (bars_per_year ** 0.5) if rets.std() > 0 else 0.0

    rolling_max = oos_eq.cummax()
    drawdowns   = (oos_eq - rolling_max) / rolling_max
    max_dd      = drawdowns.min() * 100
    n_years     = len(oos_eq) * 4 / (252 * 6)
    cagr        = ((final / initial) ** (1 / max(n_years, 1e-9)) - 1) * 100
    calmar      = cagr / abs(max_dd) if max_dd != 0 else 0.0

    if len(all_trades) > 0:
        winners = all_trades[all_trades["pnl_usd"] > 0]["pnl_usd"]
        losers  = all_trades[all_trades["pnl_usd"] <= 0]["pnl_usd"]
        win_rate = len(winners) / len(all_trades) * 100
        pf       = winners.sum() / abs(losers.sum()) if len(losers) > 0 else float("inf")
    else:
        win_rate = 0.0
        pf       = 0.0

    return WalkForwardResult(
        windows           = wf_windows,
        oos_equity        = oos_eq,
        oos_total_return  = ret,
        oos_sharpe        = sharpe,
        oos_calmar        = calmar,
        oos_max_drawdown  = max_dd,
        oos_win_rate      = win_rate,
        oos_profit_factor = pf,
        oos_total_trades  = len(all_trades),
    )


def _stitch_equity(parts: list[pd.Series], initial: float) -> pd.Series:
    """Scale each OOS equity segment so segments connect end-to-end."""
    if not parts:
        return pd.Series(dtype=float)
    stitched = []
    scale = 1.0
    for part in parts:
        if len(part) == 0:
            continue
        start_val = part.iloc[0]
        # rescale so this segment begins at previous end
        if stitched:
            scale = stitched[-1].iloc[-1] / start_val
        stitched.append(part * scale)
    return pd.concat(stitched)
