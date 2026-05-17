"""ScalpATR15m v3 — XAUUSD 15m bidireccional (LONG + SHORT).

Filtros comunes (ambas direcciones):
  - Sesión activa: London 07-10 UTC o NY 13-17 UTC
  - ADX(14) >= adx_min  — mercado con tendencia
  - ATR > mediana 50 barras  — volatilidad suficiente
  - RSI entre rsi_min y rsi_max  — momentum no extremo

Entry LONG:
  - EMA(9) > EMA(21) > EMA(50)  → tendencia alcista 15m
  - H1: EMA(9) > EMA(21)  → confluencia superior (h1_trend == 1)
  - Barra previa tocó EMA(9) por debajo y cerró cerca
  - Barra actual cierra por encima de EMA(9)

Entry SHORT (espejo):
  - EMA(9) < EMA(21) < EMA(50)  → tendencia bajista 15m
  - H1: EMA(9) < EMA(21)  → confluencia superior (h1_trend == 0)
  - Barra previa tocó EMA(9) por arriba y cerró cerca
  - Barra actual cierra por debajo de EMA(9)

Exit (trailing + cierre parcial):
  - SL fijo inicial
  - Trailing activa a trail_start_mult × ATR de ganancia
  - Cierre parcial 50% a partial_close_r × SL_dist de ganancia
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from src.strategy.base import SIGNAL_COLS, Strategy
from src.strategy.trend_atr import _adx, _atr, _ema


LONDON_OPEN_START = 7
LONDON_OPEN_END   = 10
NY_SESSION_START  = 13
NY_SESSION_END    = 17


@dataclass
class Scalp15mParams:
    ema_fast:           int   = 9
    ema_mid:            int   = 21
    ema_slow:           int   = 50
    atr_period:         int   = 14
    atr_lookback:       int   = 50
    adx_period:         int   = 14
    adx_min:            float = 20.0
    rsi_period:         int   = 14
    rsi_min:            float = 35.0
    rsi_max:            float = 70.0
    sl_atr_mult:        float = 1.5
    tp_atr_mult:        float = 4.0
    trail_start_mult:   float = 2.0
    trail_dist_mult:    float = 1.0
    partial_close_r:    float = 1.0   # cierra 50% cuando profit = 1×SL_dist
    use_session_filter: bool  = True
    use_h1_filter:      bool  = True  # requiere columna h1_trend en df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _session_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True durante London open y NY session (UTC)."""
    h = index.hour
    london = (h >= LONDON_OPEN_START) & (h < LONDON_OPEN_END)
    ny     = (h >= NY_SESSION_START)  & (h < NY_SESSION_END)
    return pd.Series(london | ny, index=index)


def build_h1_trend(df_1h: pd.DataFrame, fast: int = 9, mid: int = 21) -> pd.Series:
    """
    Calcula EMA(fast) > EMA(mid) en H1 y lo alinea al índice 15m.
    Llamar antes de generate_signals y pasar el resultado como columna h1_trend.
    """
    ema_f = _ema(df_1h["close"], fast)
    ema_m = _ema(df_1h["close"], mid)
    trend = (ema_f > ema_m).astype(int)
    return trend


class ScalpATR15m(Strategy):
    """Scalping 15m con filtro de sesión, confluencia H1 y cierre parcial."""

    def __init__(self, params: Scalp15mParams | None = None) -> None:
        self._p = params or Scalp15mParams()

    @property
    def name(self) -> str:
        return "ScalpATR_15m_v2"

    @property
    def min_warmup_bars(self) -> int:
        return self._p.ema_slow + self._p.atr_lookback + self._p.rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p   = self._p
        out = pd.DataFrame(index=df.index)

        ema_f = _ema(df["close"], p.ema_fast)
        ema_m = _ema(df["close"], p.ema_mid)
        ema_s = _ema(df["close"], p.ema_slow)
        atr   = _atr(df, p.atr_period)
        adx   = _adx(df, p.adx_period)
        rsi   = _rsi(df["close"], p.rsi_period)

        out[SIGNAL_COLS.EMA_FAST]   = ema_f
        out[SIGNAL_COLS.EMA_MEDIUM] = ema_m
        out[SIGNAL_COLS.EMA_SLOW]   = ema_s
        out[SIGNAL_COLS.ATR]        = atr

        atr_median = atr.rolling(p.atr_lookback).quantile(0.50)
        vol_ok     = atr > atr_median
        adx_ok     = adx >= p.adx_min
        rsi_ok     = (rsi >= p.rsi_min) & (rsi <= p.rsi_max)

        # ── Filtro de sesión ─────────────────────────────────────────────
        if p.use_session_filter:
            session_ok = _session_mask(df.index)
        else:
            session_ok = pd.Series(True, index=df.index)

        # ── Filtro H1 ────────────────────────────────────────────────────
        h1_available = p.use_h1_filter and "h1_trend" in df.columns
        if h1_available:
            h1_bull = df["h1_trend"].astype(bool)
            h1_bear = ~h1_bull
        else:
            h1_bull = pd.Series(True, index=df.index)
            h1_bear = pd.Series(True, index=df.index)

        # ── Condiciones LONG ──────────────────────────────────────────────
        trend_up      = (ema_f > ema_m) & (ema_m > ema_s)
        prev_low      = df["low"].shift(1)
        prev_close    = df["close"].shift(1)
        touched_long  = (prev_low <= ema_f) & (prev_close >= ema_f * 0.9995)
        bounced_long  = df["close"] > ema_f

        raw_long   = trend_up & adx_ok & vol_ok & rsi_ok & touched_long & bounced_long & session_ok & h1_bull
        entry_long = raw_long.copy()
        entry_long.iloc[:self.min_warmup_bars] = False

        # ── Condiciones SHORT (espejo de LONG) ────────────────────────────
        trend_down    = (ema_f < ema_m) & (ema_m < ema_s)
        prev_high     = df["high"].shift(1)
        touched_short = (prev_high >= ema_f) & (prev_close <= ema_f * 1.0005)
        bounced_short = df["close"] < ema_f

        raw_short   = trend_down & adx_ok & vol_ok & rsi_ok & touched_short & bounced_short & session_ok & h1_bear
        entry_short = raw_short.copy()
        entry_short.iloc[:self.min_warmup_bars] = False

        # Si ambas señales coinciden en la misma barra, ignorar esa barra
        conflict = entry_long & entry_short
        entry_long[conflict]  = False
        entry_short[conflict] = False

        out[SIGNAL_COLS.ENTRY_LONG]  = entry_long
        out[SIGNAL_COLS.ENTRY_SHORT] = entry_short

        # ── SL / TP absolutos ─────────────────────────────────────────────
        sl_dist = p.sl_atr_mult * atr
        tp_dist = p.tp_atr_mult * atr
        close   = df["close"]

        out[SIGNAL_COLS.SL_PRICE] = np.where(
            entry_long,  close - sl_dist,
            np.where(entry_short, close + sl_dist, np.nan),
        )
        out[SIGNAL_COLS.TP_PRICE] = np.where(
            entry_long,  close + tp_dist,
            np.where(entry_short, close - tp_dist, np.nan),
        )

        # ── Trailing: trigger y distancia ────────────────────────────────
        out["trail_trigger"] = np.where(
            entry_long,  close + p.trail_start_mult * atr,
            np.where(entry_short, close - p.trail_start_mult * atr, np.nan),
        )
        out["trail_distance"] = np.where(
            entry_long | entry_short, p.trail_dist_mult * atr, np.nan,
        )

        # ── Cierre parcial al 50% ─────────────────────────────────────────
        out["partial_trigger"] = np.where(
            entry_long,  close + p.partial_close_r * sl_dist,
            np.where(entry_short, close - p.partial_close_r * sl_dist, np.nan),
        )

        return out
