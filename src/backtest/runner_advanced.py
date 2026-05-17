"""Runner avanzado — LONG y SHORT con trailing stop y cierre parcial a 1R.

Lógica por barra con posición abierta (aplica a ambas direcciones):
  1. Cierre parcial: si precio alcanza partial_trigger → cierra 50%
  2. Activar trailing si precio alcanza trail_trigger
  3. Actualizar trailing (SL sigue el precio favorable)
  4. Verificar SL → cierre con pérdida
  5. Verificar TP hard → cierre si trailing no arrancó

LONG:  SL debajo del precio, TP arriba. Trailing sigue máximos.
SHORT: SL encima del precio, TP abajo. Trailing sigue mínimos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.runner import (
    BacktestResult, BrokerConfig,
    _commission, _compute_metrics, _empty_trades, _lot_size, _spread_cost,
)
from src.strategy.base import SIGNAL_COLS, Strategy

LONG  =  1
SHORT = -1


def run_backtest_advanced(
    df: pd.DataFrame,
    strategy: Strategy,
    broker: BrokerConfig | None = None,
) -> BacktestResult:
    """
    Backtest bidireccional con trailing stop y cierre parcial.

    Columnas extra que la estrategia debe incluir:
      - trail_trigger   : precio de activación del trailing
      - trail_distance  : distancia del trailing en puntos de precio
      - partial_trigger : precio de cierre parcial del 50%
    """
    if broker is None:
        broker = BrokerConfig()

    signals     = strategy.generate_signals(df)
    has_trail   = "trail_trigger"   in signals.columns and "trail_distance" in signals.columns
    has_partial = "partial_trigger" in signals.columns

    equity       = broker.initial_equity
    equity_curve = []
    trades       = []

    in_trade       = False
    direction      = LONG
    entry_price    = sl_price = tp_price = 0.0
    lots           = 0.0
    lots_remaining = 0.0
    entry_time     = None
    entry_equity   = 0.0
    trail_active   = False
    partial_done   = False
    extreme_price  = 0.0   # peak (long) o trough (short)
    trail_dist     = 0.0
    tr_trigger     = np.nan
    partial_trig   = np.nan

    for i, ts in enumerate(df.index):
        row = df.iloc[i]
        sig = signals.iloc[i]

        if in_trade:
            bar_high = row["high"]
            bar_low  = row["low"]

            active_lots = lots_remaining if partial_done else lots

            # ── 1. Cierre parcial ─────────────────────────────────────────
            if not partial_done and has_partial and not np.isnan(partial_trig):
                hit_partial = (
                    (direction == LONG  and bar_high >= partial_trig) or
                    (direction == SHORT and bar_low  <= partial_trig)
                )
                if hit_partial:
                    partial_lots = lots * 0.5
                    pnl_pips     = (partial_trig - entry_price) * direction / broker.point_size
                    pnl_usd      = pnl_pips * broker.pip_value_per_lot * partial_lots
                    net_partial  = pnl_usd - _commission(partial_lots, broker) - _spread_cost(partial_lots, broker)
                    equity      += net_partial
                    lots_remaining = lots * 0.5
                    partial_done   = True
                    active_lots    = lots_remaining
                    trades.append({
                        "entry_time":   entry_time,
                        "exit_time":    ts,
                        "entry_price":  entry_price,
                        "exit_price":   partial_trig,
                        "direction":    "LONG" if direction == LONG else "SHORT",
                        "lots":         partial_lots,
                        "pnl_usd":      net_partial,
                        "pnl_pct":      net_partial / entry_equity,
                        "exit_reason":  "PARTIAL",
                        "trail_active": False,
                    })

            # ── 2. Activar / actualizar trailing ──────────────────────────
            if trail_active:
                if direction == LONG:
                    if bar_high > extreme_price:
                        extreme_price = bar_high
                        sl_price      = extreme_price - trail_dist
                else:
                    if bar_low < extreme_price:
                        extreme_price = bar_low
                        sl_price      = extreme_price + trail_dist

            elif has_trail and not np.isnan(tr_trigger):
                activated = (
                    (direction == LONG  and bar_high >= tr_trigger) or
                    (direction == SHORT and bar_low  <= tr_trigger)
                )
                if activated:
                    trail_active  = True
                    extreme_price = bar_high if direction == LONG else bar_low
                    sl_price      = (extreme_price - trail_dist if direction == LONG
                                     else extreme_price + trail_dist)

            # ── 3. Verificar SL / TP ──────────────────────────────────────
            if direction == LONG:
                hit_sl = bar_low  <= sl_price
                hit_tp = bar_high >= tp_price and not trail_active
            else:
                hit_sl = bar_high >= sl_price
                hit_tp = bar_low  <= tp_price and not trail_active

            if hit_sl or hit_tp:
                exit_price  = sl_price if hit_sl else tp_price
                exit_reason = ("TRAIL" if (hit_sl and trail_active)
                               else "SL"    if hit_sl
                               else "TP")

                pnl_pips = (exit_price - entry_price) * direction / broker.point_size
                pnl_usd  = pnl_pips * broker.pip_value_per_lot * active_lots
                net_pnl  = pnl_usd - _commission(active_lots, broker) - _spread_cost(active_lots, broker)
                equity  += net_pnl

                trades.append({
                    "entry_time":   entry_time,
                    "exit_time":    ts,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "direction":    "LONG" if direction == LONG else "SHORT",
                    "lots":         active_lots,
                    "pnl_usd":      net_pnl,
                    "pnl_pct":      net_pnl / entry_equity,
                    "exit_reason":  exit_reason,
                    "trail_active": trail_active,
                })
                in_trade     = False
                trail_active = False
                partial_done = False

        else:
            # ── Nueva entrada ─────────────────────────────────────────────
            is_long  = bool(sig[SIGNAL_COLS.ENTRY_LONG])
            is_short = bool(sig[SIGNAL_COLS.ENTRY_SHORT])

            if is_long or is_short:
                sl_pt = sig[SIGNAL_COLS.SL_PRICE]
                tp_pt = sig[SIGNAL_COLS.TP_PRICE]
                if not pd.isna(sl_pt) and not pd.isna(tp_pt):
                    direction      = LONG if is_long else SHORT
                    entry_price    = row["close"]
                    sl_price       = float(sl_pt)
                    tp_price       = float(tp_pt)
                    sl_dist_pts    = abs(entry_price - sl_price)
                    lots           = _lot_size(equity, broker.risk_per_trade_pct,
                                               sl_dist_pts, broker)
                    lots_remaining = lots
                    entry_time     = ts
                    entry_equity   = equity
                    equity        -= _commission(lots, broker) + _spread_cost(lots, broker)
                    tr_trigger     = float(sig.get("trail_trigger",   np.nan)) if has_trail   else np.nan
                    trail_dist     = float(sig.get("trail_distance",  0.0))    if has_trail   else 0.0
                    partial_trig   = float(sig.get("partial_trigger", np.nan)) if has_partial else np.nan
                    trail_active   = False
                    partial_done   = False
                    extreme_price  = entry_price
                    in_trade       = True

        equity_curve.append(equity)

    # ── Cierre forzado al final del periodo ───────────────────────────────
    if in_trade:
        active_lots = lots_remaining if partial_done else lots
        exit_price  = df.iloc[-1]["close"]
        pnl_pips    = (exit_price - entry_price) * direction / broker.point_size
        pnl_usd     = pnl_pips * broker.pip_value_per_lot * active_lots
        equity     += pnl_usd
        equity_curve[-1] = equity
        trades.append({
            "entry_time":   entry_time,
            "exit_time":    df.index[-1],
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "direction":    "LONG" if direction == LONG else "SHORT",
            "lots":         active_lots,
            "pnl_usd":      pnl_usd,
            "pnl_pct":      pnl_usd / entry_equity,
            "exit_reason":  "EOD",
            "trail_active": trail_active,
        })

    eq_series = pd.Series(equity_curve, index=df.index, name="equity")
    trades_df = pd.DataFrame(trades) if trades else _empty_trades()

    return BacktestResult(
        equity_curve = eq_series,
        trades       = trades_df,
        **_compute_metrics(eq_series, trades_df, broker.initial_equity),
    )
