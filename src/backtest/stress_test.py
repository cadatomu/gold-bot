"""Stress tests — run the backtest under adversarial cost/slippage scenarios.

Three standard stress scenarios:
  1. spread_shock  — spread × 3 (simulates illiquid session or news spike)
  2. commission_shock — commission × 2 (broker fee increase)
  3. slippage_shock — slippage × 5 (gapping market, partial fill)

For each scenario a full BacktestResult is produced and compared to the
baseline so degradation is quantified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtest.runner import BacktestResult, BrokerConfig, run_backtest
from src.strategy.base import Strategy


@dataclass
class StressScenario:
    name:        str
    description: str
    broker_fn:   Callable[[BrokerConfig], BrokerConfig]  # transforms baseline cfg


@dataclass
class StressResult:
    scenario:       StressScenario
    result:         BacktestResult
    # Degradation vs baseline (positive = worse)
    return_delta:   float   # baseline_return - stress_return  (pct points)
    drawdown_delta: float   # stress_dd - baseline_dd  (absolute pct, positive = deeper)
    calmar_delta:   float   # baseline_calmar - stress_calmar


@dataclass
class StressSuiteResult:
    baseline:  BacktestResult
    scenarios: list[StressResult]

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.scenarios:
            rows.append({
                "scenario":         s.scenario.name,
                "total_return_pct": round(s.result.total_return_pct, 2),
                "max_dd_pct":       round(s.result.max_drawdown_pct, 2),
                "calmar":           round(s.result.calmar, 3),
                "sharpe":           round(s.result.sharpe, 3),
                "win_rate_pct":     round(s.result.win_rate_pct, 2),
                "total_trades":     s.result.total_trades,
                "return_delta":     round(s.return_delta, 2),
                "drawdown_delta":   round(s.drawdown_delta, 2),
                "calmar_delta":     round(s.calmar_delta, 3),
            })
        return pd.DataFrame(rows)


# Built-in stress scenarios
SPREAD_SHOCK = StressScenario(
    name        = "spread_shock",
    description = "Spread × 3 — illiquid session or news spike",
    broker_fn   = lambda b: BrokerConfig(
        initial_equity       = b.initial_equity,
        spread_pips          = b.spread_pips * 3,
        commission_per_lot_rt= b.commission_per_lot_rt,
        slippage_pips        = b.slippage_pips,
        risk_per_trade_pct   = b.risk_per_trade_pct,
    ),
)

COMMISSION_SHOCK = StressScenario(
    name        = "commission_shock",
    description = "Commission × 2 — broker fee increase",
    broker_fn   = lambda b: BrokerConfig(
        initial_equity       = b.initial_equity,
        spread_pips          = b.spread_pips,
        commission_per_lot_rt= b.commission_per_lot_rt * 2,
        slippage_pips        = b.slippage_pips,
        risk_per_trade_pct   = b.risk_per_trade_pct,
    ),
)

SLIPPAGE_SHOCK = StressScenario(
    name        = "slippage_shock",
    description = "Slippage × 5 — gapping market, partial fill",
    broker_fn   = lambda b: BrokerConfig(
        initial_equity       = b.initial_equity,
        spread_pips          = b.spread_pips,
        commission_per_lot_rt= b.commission_per_lot_rt,
        slippage_pips        = b.slippage_pips * 5,
        risk_per_trade_pct   = b.risk_per_trade_pct,
    ),
)

DEFAULT_SCENARIOS = [SPREAD_SHOCK, COMMISSION_SHOCK, SLIPPAGE_SHOCK]


def run_stress_suite(
    df:        pd.DataFrame,
    strategy:  Strategy,
    baseline:  BrokerConfig | None = None,
    scenarios: list[StressScenario] | None = None,
) -> StressSuiteResult:
    """
    Run baseline + all stress scenarios and return comparative results.

    Parameters
    ----------
    df        : Full OHLCV history
    strategy  : Strategy instance (same params for all runs)
    baseline  : BrokerConfig to use as baseline (defaults if None)
    scenarios : List of StressScenario (uses DEFAULT_SCENARIOS if None)
    """
    if baseline is None:
        baseline = BrokerConfig()
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    base_result = run_backtest(df, strategy, baseline)
    stress_results = []

    for scenario in scenarios:
        stressed_broker = scenario.broker_fn(baseline)
        result = run_backtest(df, strategy, stressed_broker)

        stress_results.append(StressResult(
            scenario       = scenario,
            result         = result,
            return_delta   = base_result.total_return_pct - result.total_return_pct,
            drawdown_delta = result.max_drawdown_pct - base_result.max_drawdown_pct,
            calmar_delta   = base_result.calmar - result.calmar,
        ))

    return StressSuiteResult(baseline=base_result, scenarios=stress_results)
