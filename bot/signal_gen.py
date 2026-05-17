"""Generador de señales H4 usando AdaptiveScalp15m con params optimizados."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategy.adaptive import AdaptiveParams, AdaptiveScalp15m
from src.strategy.regime import RegimeParams
from src.strategy.scalp_15m import build_h1_trend

log = logging.getLogger(__name__)

# Parámetros H4 optimizados — backtest 20 meses, retorno 17.69%/mes
# regime_adx=21, regime_slope=0.15, adx_min=25, sl=2.2ATR, tp=2.6ATR
# trail_start=3.8ATR, trail_dist=0.8ATR, partial=1.5R, risk=8%
H4_PARAMS = AdaptiveParams(
    regime           = RegimeParams(adx_trend_min=21, slope_threshold=0.150),
    adx_min          = 25.0,
    rsi_min          = 25.0,
    rsi_max          = 70.0,
    sl_atr_mult      = 2.2,
    tp_atr_mult      = 2.6,
    trail_start_mult = 3.8,
    trail_dist_mult  = 0.8,
    partial_close_r  = 1.5,
    use_session_filter = False,
    use_h1_filter    = True,
)

_STRATEGY = AdaptiveScalp15m(H4_PARAMS)


def get_signal(h4_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    """
    Calcula la señal de la última barra H4 completa.

    Retorna dict con:
      direction        : 'LONG' | 'SHORT' | 'FLAT'
      close            : precio de cierre de la barra
      atr              : ATR de la barra
      sl_price         : precio del stop-loss
      tp_price         : precio del take-profit
      trail_trigger    : precio al que activa el trailing stop
      trail_distance   : distancia del trail en $/oz
      partial_trigger  : precio para cierre parcial (50%)
      bar_time         : timestamp de la barra (str)
    """
    if h4_df.empty or daily_df.empty:
        return {"direction": "FLAT", "bar_time": ""}

    # Filtro de tendencia usando Daily como TF superior (equivale al H1 en el backtest)
    daily_idx = daily_df.index
    if daily_idx.tzinfo is None:
        daily_idx = daily_idx.tz_localize("UTC")
    daily_trend = build_h1_trend(daily_df)
    daily_trend.index = daily_idx

    df = h4_df.copy()
    df["h1_trend"] = daily_trend.reindex(df.index, method="ffill").fillna(0).astype(int)

    signals = _STRATEGY.generate_signals(df)

    last      = signals.iloc[-1]
    close     = float(df["close"].iloc[-1])
    atr       = float(last.get("atr", 0.0) or 0.0)
    bar_time  = str(df.index[-1])

    if last["entry_long"]:
        direction = "LONG"
    elif last["entry_short"]:
        direction = "SHORT"
    else:
        direction = "FLAT"

    log.info(f"Señal H4 [{bar_time}]: {direction} | close={close:.2f} ATR={atr:.2f}")

    if direction == "FLAT":
        return {"direction": "FLAT", "close": close, "atr": atr, "bar_time": bar_time}

    return {
        "direction":       direction,
        "close":           close,
        "atr":             atr,
        "sl_price":        float(last["sl_price"]),
        "tp_price":        float(last["tp_price"]),
        "trail_trigger":   float(last["trail_trigger"]),
        "trail_distance":  float(last["trail_distance"]),
        "partial_trigger": float(last["partial_trigger"]),
        "bar_time":        bar_time,
    }
