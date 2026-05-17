"""AdaptiveScalp15m — detecta régimen y opera solo en tendencia.

  BULL   → entrada LONG  (pullback a EMA alcista)
  BEAR   → entrada SHORT (pullback a EMA bajista)
  LATERAL → sin operaciones
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.strategy.base import SIGNAL_COLS, Strategy
from src.strategy.trend_atr import _atr, _adx, _ema
from src.strategy.scalp_15m import _rsi, _session_mask
from src.strategy.regime import RegimeParams, classify_regime, BULL, BEAR


@dataclass
class AdaptiveParams:
    # Detector de régimen
    regime: RegimeParams = field(default_factory=RegimeParams)

    # Indicadores
    ema_fast:       int   = 9
    ema_mid:        int   = 21
    ema_slow:       int   = 50
    atr_period:     int   = 14
    atr_lookback:   int   = 50
    adx_period:     int   = 14
    adx_min:        float = 25.0
    rsi_period:     int   = 14
    rsi_min:        float = 35.0
    rsi_max:        float = 65.0

    # Risk / exit
    sl_atr_mult:        float = 1.7
    tp_atr_mult:        float = 4.0
    trail_start_mult:   float = 2.0
    trail_dist_mult:    float = 0.4
    partial_close_r:    float = 1.25

    # Filtros
    use_session_filter: bool = True
    use_h1_filter:      bool = True


class AdaptiveScalp15m(Strategy):
    """Opera LONG en BULL, SHORT en BEAR, quieto en LATERAL."""

    def __init__(self, params: AdaptiveParams | None = None) -> None:
        self._p = params or AdaptiveParams()

    @property
    def name(self) -> str:
        return "AdaptiveScalp_15m_v2"

    @property
    def min_warmup_bars(self) -> int:
        p = self._p
        return p.ema_slow + p.atr_lookback + p.rsi_period + p.regime.slope_lookback

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

        regime = classify_regime(adx, ema_s, p.regime)
        out["regime"] = regime

        # Filtros comunes
        atr_median = atr.rolling(p.atr_lookback).quantile(0.50)
        vol_ok     = atr > atr_median
        adx_ok     = adx >= p.adx_min
        rsi_ok     = (rsi >= p.rsi_min) & (rsi <= p.rsi_max)

        session_ok = _session_mask(df.index) if p.use_session_filter else pd.Series(True, index=df.index)

        h1_available = p.use_h1_filter and "h1_trend" in df.columns
        h1_bull = df["h1_trend"].astype(bool)          if h1_available else pd.Series(True,  index=df.index)
        h1_bear = ~df["h1_trend"].astype(bool)         if h1_available else pd.Series(True,  index=df.index)

        prev_low   = df["low"].shift(1)
        prev_high  = df["high"].shift(1)
        prev_close = df["close"].shift(1)
        close      = df["close"]

        # LONG — solo en régimen BULL
        trend_up     = (ema_f > ema_m) & (ema_m > ema_s)
        touch_long   = (prev_low <= ema_f) & (prev_close >= ema_f * 0.9995)
        bounce_long  = close > ema_f
        entry_long   = (
            (regime == BULL) & trend_up & adx_ok & vol_ok & rsi_ok
            & touch_long & bounce_long & session_ok & h1_bull
        )

        # SHORT — solo en régimen BEAR
        trend_down   = (ema_f < ema_m) & (ema_m < ema_s)
        touch_short  = (prev_high >= ema_f) & (prev_close <= ema_f * 1.0005)
        bounce_short = close < ema_f
        entry_short  = (
            (regime == BEAR) & trend_down & adx_ok & vol_ok & rsi_ok
            & touch_short & bounce_short & session_ok & h1_bear
        )

        # Warmup y conflictos
        entry_long  = entry_long.copy()
        entry_short = entry_short.copy()
        entry_long.iloc[:self.min_warmup_bars]  = False
        entry_short.iloc[:self.min_warmup_bars] = False
        conflict = entry_long & entry_short
        entry_long[conflict]  = False
        entry_short[conflict] = False

        out[SIGNAL_COLS.ENTRY_LONG]  = entry_long
        out[SIGNAL_COLS.ENTRY_SHORT] = entry_short

        # SL / TP
        sl_dist = p.sl_atr_mult * atr
        tp_dist = p.tp_atr_mult * atr

        out[SIGNAL_COLS.SL_PRICE] = np.where(
            entry_long,  close - sl_dist,
            np.where(entry_short, close + sl_dist, np.nan),
        )
        out[SIGNAL_COLS.TP_PRICE] = np.where(
            entry_long,  close + tp_dist,
            np.where(entry_short, close - tp_dist, np.nan),
        )

        out["trail_trigger"] = np.where(
            entry_long,  close + p.trail_start_mult * atr,
            np.where(entry_short, close - p.trail_start_mult * atr, np.nan),
        )
        out["trail_distance"] = np.where(
            entry_long | entry_short, p.trail_dist_mult * atr, np.nan,
        )
        out["partial_trigger"] = np.where(
            entry_long,  close + p.partial_close_r * sl_dist,
            np.where(entry_short, close - p.partial_close_r * sl_dist, np.nan),
        )

        return out
