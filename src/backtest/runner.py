"""vectorbt-based backtester for the gold bot strategies.

Simulates fixed-SL / fixed-TP trades on H4 OHLCV data.
Uses bar-level exits (SL/TP checked against bar high/low each bar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import SIGNAL_COLS, Strategy


@dataclass
class BrokerConfig:
    initial_equity: float = 10_000.0
    spread_pips: float = 0.30       # XAUUSD 1 pip = $0.10 per 0.01 lot
    commission_per_lot_rt: float = 7.0
    slippage_pips: float = 1.0
    risk_per_trade_pct: float = 0.005
    pip_value_per_lot: float = 10.0  # 1 pip = $10 per standard lot (100 oz)
    point_size: float = 0.01         # XAUUSD


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame            # one row per closed trade
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    avg_trade_pct: float


def _lot_size(equity: float, risk_pct: float, sl_pts: float, cfg: BrokerConfig) -> float:
    """Calculate lot size so that risk_pct of equity is lost if SL is hit."""
    if sl_pts <= 0:
        return 0.01
    risk_usd = equity * risk_pct
    # 1 lot loses sl_pts * pip_value_per_lot / point_size dollars
    pips = sl_pts / cfg.point_size
    loss_per_lot = pips * cfg.pip_value_per_lot
    if loss_per_lot <= 0:
        return 0.01
    lots = risk_usd / loss_per_lot
    return max(round(lots, 2), 0.01)


def _commission(lots: float, cfg: BrokerConfig) -> float:
    return lots * cfg.commission_per_lot_rt


def _spread_cost(lots: float, cfg: BrokerConfig) -> float:
    pips = cfg.spread_pips + cfg.slippage_pips
    return lots * pips * cfg.pip_value_per_lot


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    broker: Optional[BrokerConfig] = None,
) -> BacktestResult:
    """
    Run a single-pass backtest on `df` using `strategy`.

    Parameters
    ----------
    df      : OHLCV DataFrame, UTC index
    strategy: any Strategy subclass
    broker  : BrokerConfig (defaults used if None)

    Returns
    -------
    BacktestResult with equity curve, trade log, and summary metrics
    """
    if broker is None:
        broker = BrokerConfig()

    signals = strategy.generate_signals(df)

    equity = broker.initial_equity
    equity_curve = []
    trades = []

    in_trade = False
    entry_price = sl_price = tp_price = lots = 0.0
    entry_time = None
    entry_equity = 0.0

    for i, ts in enumerate(df.index):
        row = df.iloc[i]
        sig = signals.iloc[i]

        if in_trade:
            # Check SL / TP against bar high/low (pessimistic: SL first)
            hit_sl = row["low"] <= sl_price
            hit_tp = row["high"] >= tp_price

            if hit_sl or hit_tp:
                exit_price = sl_price if hit_sl else tp_price
                pnl_pips  = (exit_price - entry_price) / broker.point_size
                pnl_usd   = pnl_pips * broker.pip_value_per_lot * lots
                cost      = _commission(lots, broker) + _spread_cost(lots, broker)
                net_pnl   = pnl_usd - cost
                equity   += net_pnl

                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   ts,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "lots":        lots,
                    "pnl_usd":     net_pnl,
                    "pnl_pct":     net_pnl / entry_equity,
                    "exit_reason": "SL" if hit_sl else "TP",
                })
                in_trade = False

        elif sig[SIGNAL_COLS.ENTRY_LONG] and not in_trade:
            # Enter on next bar open is more realistic, but for simplicity
            # we use the signal bar's close (consistent with signal generation)
            sl_pt = sig[SIGNAL_COLS.SL_PRICE]
            tp_pt = sig[SIGNAL_COLS.TP_PRICE]
            if pd.isna(sl_pt) or pd.isna(tp_pt):
                pass
            else:
                entry_price  = row["close"]
                sl_price     = sl_pt
                tp_price     = tp_pt
                sl_dist      = entry_price - sl_price
                lots         = _lot_size(equity, broker.risk_per_trade_pct, sl_dist, broker)
                entry_time   = ts
                entry_equity = equity
                cost         = _commission(lots, broker) + _spread_cost(lots, broker)
                equity      -= cost  # pre-deduct entry cost
                in_trade     = True

        equity_curve.append(equity)

    # Close any open trade at last bar close
    if in_trade:
        exit_price = df.iloc[-1]["close"]
        pnl_pips   = (exit_price - entry_price) / broker.point_size
        pnl_usd    = pnl_pips * broker.pip_value_per_lot * lots
        equity    += pnl_usd
        equity_curve[-1] = equity
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   df.index[-1],
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "lots":        lots,
            "pnl_usd":     pnl_usd,
            "pnl_pct":     pnl_usd / entry_equity,
            "exit_reason": "EOD",
        })

    eq_series = pd.Series(equity_curve, index=df.index, name="equity")
    trades_df  = pd.DataFrame(trades) if trades else _empty_trades()

    return BacktestResult(
        equity_curve      = eq_series,
        trades            = trades_df,
        **_compute_metrics(eq_series, trades_df, broker.initial_equity),
    )


def _empty_trades() -> pd.DataFrame:
    cols = ["entry_time", "exit_time", "entry_price", "exit_price",
            "lots", "pnl_usd", "pnl_pct", "exit_reason"]
    return pd.DataFrame(columns=cols)


def _compute_metrics(
    eq: pd.Series,
    trades: pd.DataFrame,
    initial_equity: float,
) -> dict:
    final_equity   = eq.iloc[-1]
    total_return   = (final_equity / initial_equity - 1) * 100

    # CAGR
    n_years = len(eq) * 4 / (252 * 6.5)  # H4 bars per year ≈ 252*6 = 1512
    cagr    = ((final_equity / initial_equity) ** (1 / max(n_years, 1e-9)) - 1) * 100

    # Returns series (bar-by-bar)
    rets = eq.pct_change().dropna()

    # Sharpe (annualised, H4 bars → 1512 bars/year)
    bars_per_year = 1512
    sharpe = (rets.mean() / rets.std(ddof=1)) * (bars_per_year ** 0.5) if rets.std() > 0 else 0.0

    # Sortino (downside std)
    neg_rets = rets[rets < 0]
    sortino  = (rets.mean() / neg_rets.std(ddof=1)) * (bars_per_year ** 0.5) if len(neg_rets) > 1 else 0.0

    # Max drawdown
    rolling_max = eq.cummax()
    drawdowns   = (eq - rolling_max) / rolling_max
    max_dd      = drawdowns.min() * 100  # negative number

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Trade stats
    if len(trades) == 0:
        return dict(
            total_return_pct=total_return, cagr_pct=cagr, sharpe=sharpe,
            sortino=sortino, calmar=calmar, max_drawdown_pct=max_dd,
            win_rate_pct=0.0, profit_factor=0.0,
            total_trades=0, avg_trade_pct=0.0,
        )

    winners      = trades[trades["pnl_usd"] > 0]["pnl_usd"]
    losers       = trades[trades["pnl_usd"] <= 0]["pnl_usd"]
    win_rate     = len(winners) / len(trades) * 100
    gross_profit = winners.sum()
    gross_loss   = abs(losers.sum())
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_trade    = trades["pnl_pct"].mean() * 100

    return dict(
        total_return_pct=total_return, cagr_pct=cagr, sharpe=sharpe,
        sortino=sortino, calmar=calmar, max_drawdown_pct=max_dd,
        win_rate_pct=win_rate, profit_factor=pf,
        total_trades=len(trades), avg_trade_pct=avg_trade,
    )
