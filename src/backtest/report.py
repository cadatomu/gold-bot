"""quantstats-based HTML and console reporting for backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import quantstats as qs

from src.backtest.runner import BacktestResult
from src.backtest.walk_forward import WalkForwardResult


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    rets = equity.pct_change().dropna()
    rets.index = pd.DatetimeIndex(rets.index)
    return rets


def print_summary(result: BacktestResult) -> None:
    """Print a compact performance table to stdout."""
    trades = result.total_trades
    print(f"\n{'─'*50}")
    print(f"  Total return  : {result.total_return_pct:+.2f}%")
    print(f"  CAGR          : {result.cagr_pct:+.2f}%")
    print(f"  Sharpe        : {result.sharpe:.3f}")
    print(f"  Sortino       : {result.sortino:.3f}")
    print(f"  Calmar        : {result.calmar:.3f}")
    print(f"  Max drawdown  : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate      : {result.win_rate_pct:.1f}%")
    print(f"  Profit factor : {result.profit_factor:.3f}")
    print(f"  Total trades  : {trades}")
    if trades > 0:
        print(f"  Avg trade     : {result.avg_trade_pct:+.3f}%")
    print(f"{'─'*50}\n")


def save_html_report(
    result: BacktestResult,
    output_path: Path,
    title: str = "TrendATR — XAUUSD H4",
    benchmark: pd.Series | None = None,
) -> None:
    """
    Generate a full quantstats HTML tear-sheet.

    Parameters
    ----------
    result      : BacktestResult
    output_path : Where to write the .html file
    title       : Report title
    benchmark   : Optional benchmark returns series (same index)
    """
    rets = _returns_from_equity(result.equity_curve)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        rets,
        benchmark  = benchmark,
        output     = str(output_path),
        title      = title,
        download_filename = output_path.name,
    )


def save_wf_html_report(
    wf_result: WalkForwardResult,
    output_path: Path,
    title: str = "TrendATR Walk-Forward — XAUUSD H4",
) -> None:
    """Generate an HTML report for the concatenated OOS equity curve."""
    rets = _returns_from_equity(wf_result.oos_equity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        rets,
        output = str(output_path),
        title  = title,
        download_filename = output_path.name,
    )


def print_stress_summary(suite) -> None:  # type: ignore[no-untyped-def]
    """Print stress test comparison table."""
    from src.backtest.stress_test import StressSuiteResult
    assert isinstance(suite, StressSuiteResult)
    base = suite.baseline
    print(f"\n{'─'*70}")
    print(f"  {'Scenario':<22} {'Return%':>8} {'MaxDD%':>8} {'Calmar':>8} {'Trades':>7}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    print(f"  {'BASELINE':<22} {base.total_return_pct:>8.2f} "
          f"{base.max_drawdown_pct:>8.2f} {base.calmar:>8.3f} "
          f"{base.total_trades:>7}")
    for s in suite.scenarios:
        r = s.result
        print(f"  {s.scenario.name:<22} {r.total_return_pct:>8.2f} "
              f"{r.max_drawdown_pct:>8.2f} {r.calmar:>8.3f} "
              f"{r.total_trades:>7}  "
              f"(Δret={s.return_delta:+.1f}%)")
    print(f"{'─'*70}\n")
