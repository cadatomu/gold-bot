"""Estrategia de reversión a la media — mercado LATERAL XAUUSD 15m.

Lógica:
  - Bollinger Bands(bb_period, bb_std) delimitan el rango
  - LONG: barra toca banda inferior y cierra adentro + RSI < rsi_os
  - SHORT: barra toca banda superior y cierra adentro + RSI > rsi_ob
  - SL fijo: sl_atr_mult × ATR desde la entrada
  - TP en la banda media (EMA del BB): mercado en rango, no se espera
    que el precio viaje más allá del centro
  - Sin trailing stop: en rango se toman ganancias rápido
  - Cierre parcial a partial_close_r × sl_dist (más conservador que trend)

Justificación de parámetros:
  bb_period=20 / bb_std=2.0 — estándar Bollinger clásico (Wilder 1983)
  rsi_os=38 / rsi_ob=62    — umbrales más estrechos que extremos para
                             capturar sobrecompra/sobreventa moderada en rango
  sl_atr_mult=1.0           — SL más ajustado porque el rango es acotado
  tp_band_middle=True       — TP en EMA20 (media del BB), objetivo natural
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy.base import SIGNAL_COLS
from src.strategy.trend_atr import _atr, _ema
from src.strategy.scalp_15m import _rsi


@dataclass
class MeanReversionParams:
    bb_period:       int   = 20
    bb_std:          float = 2.0
    rsi_period:      int   = 14
    rsi_os:          float = 38.0   # oversold threshold (long entry)
    rsi_ob:          float = 62.0   # overbought threshold (short entry)
    atr_period:      int   = 14
    sl_atr_mult:     float = 1.0    # SL ajustado: rango más estrecho
    partial_close_r: float = 0.75   # cierra 50% a 0.75× SL_dist
    use_session_filter: bool = True


def _bollinger(
    series: pd.Series,
    period: int,
    n_std:  float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Retorna (upper, middle, lower) para Bollinger Bands."""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std(ddof=0)
    return mid + n_std * sigma, mid, mid - n_std * sigma


def generate_mr_signals(
    df:     pd.DataFrame,
    params: MeanReversionParams,
    warmup: int,
) -> pd.DataFrame:
    """
    Genera señales de reversión a la media.
    No es una Strategy completa — la llama AdaptiveScalp15m
    para las barras con régimen LATERAL.

    Retorna DataFrame con las mismas columnas que generate_signals().
    """
    p   = params
    out = pd.DataFrame(index=df.index, dtype=object)

    atr = _atr(df, p.atr_period)
    rsi = _rsi(df["close"], p.rsi_period)
    upper_bb, mid_bb, lower_bb = _bollinger(df["close"], p.bb_period, p.bb_std)

    # ── Sesión ────────────────────────────────────────────────────────────
    from src.strategy.scalp_15m import _session_mask, LONDON_OPEN_START, LONDON_OPEN_END, NY_SESSION_START, NY_SESSION_END
    if p.use_session_filter:
        session_ok = _session_mask(df.index)
    else:
        session_ok = pd.Series(True, index=df.index)

    # ── Entry LONG: toca banda inferior y rebota ──────────────────────────
    prev_low   = df["low"].shift(1)
    touched_lo = prev_low <= lower_bb
    close_back = df["close"] > lower_bb   # cerró dentro del rango
    rsi_os_ok  = rsi < p.rsi_os

    raw_long = touched_lo & close_back & rsi_os_ok & session_ok

    # ── Entry SHORT: toca banda superior y rebota ─────────────────────────
    prev_high  = df["high"].shift(1)
    touched_hi = prev_high >= upper_bb
    close_back_s = df["close"] < upper_bb
    rsi_ob_ok  = rsi > p.rsi_ob

    raw_short = touched_hi & close_back_s & rsi_ob_ok & session_ok

    entry_long  = raw_long.copy()
    entry_short = raw_short.copy()
    entry_long.iloc[:warmup]  = False
    entry_short.iloc[:warmup] = False

    conflict = entry_long & entry_short
    entry_long[conflict]  = False
    entry_short[conflict] = False

    # ── SL / TP ───────────────────────────────────────────────────────────
    sl_dist = p.sl_atr_mult * atr
    close   = df["close"]

    # TP en la banda media (EMA20) — objetivo natural del rango
    tp_long  = mid_bb
    tp_short = mid_bb

    # Asegurarse que el TP esté al menos sl_dist/2 alejado (TP mínimo viable)
    tp_long  = np.maximum(tp_long,  close + sl_dist * 0.5)
    tp_short = np.minimum(tp_short, close - sl_dist * 0.5)

    out[SIGNAL_COLS.EMA_FAST]   = mid_bb      # reutilizamos como referencia BB medio
    out[SIGNAL_COLS.EMA_MEDIUM] = upper_bb
    out[SIGNAL_COLS.EMA_SLOW]   = lower_bb
    out[SIGNAL_COLS.ATR]        = atr

    out[SIGNAL_COLS.ENTRY_LONG]  = entry_long
    out[SIGNAL_COLS.ENTRY_SHORT] = entry_short

    out[SIGNAL_COLS.SL_PRICE] = np.where(
        entry_long,  close - sl_dist,
        np.where(entry_short, close + sl_dist, np.nan),
    )
    out[SIGNAL_COLS.TP_PRICE] = np.where(
        entry_long,  tp_long,
        np.where(entry_short, tp_short, np.nan),
    )

    # Sin trailing en mean reversion — se toman ganancias al llegar a la media
    out["trail_trigger"]  = np.nan
    out["trail_distance"] = np.nan

    # Cierre parcial más conservador
    out["partial_trigger"] = np.where(
        entry_long,  close + p.partial_close_r * sl_dist,
        np.where(entry_short, close - p.partial_close_r * sl_dist, np.nan),
    )

    return out
