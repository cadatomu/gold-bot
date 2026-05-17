"""Detector de régimen de mercado para XAUUSD 15m.

Clasifica cada barra en uno de tres estados:
  BULL   — ADX >= umbral Y EMA(50) con pendiente positiva
  BEAR   — ADX >= umbral Y EMA(50) con pendiente negativa
  LATERAL — ADX < umbral O pendiente plana (mercado en rango)

La clasificación se hace barra a barra sin lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


BULL    = "BULL"
BEAR    = "BEAR"
LATERAL = "LATERAL"


@dataclass
class RegimeParams:
    adx_trend_min:    float = 25.0  # ADX mínimo para considerar tendencia
    slope_ema_period: int   = 50    # EMA cuya pendiente mide la dirección
    slope_lookback:   int   = 10    # barras para calcular el cambio de pendiente
    slope_threshold:  float = 0.03  # cambio % mínimo para confirmar dirección


def classify_regime(
    adx:     pd.Series,
    ema_slow: pd.Series,
    params:  RegimeParams,
) -> pd.Series:
    """
    Devuelve una Series con valores 'BULL', 'BEAR' o 'LATERAL' por barra.

    Parameters
    ----------
    adx      : ADX calculado externamente
    ema_slow : EMA lenta (ej. EMA50) calculada externamente
    params   : RegimeParams

    Returns
    -------
    pd.Series[str] con el régimen de cada barra
    """
    p = params

    # Pendiente de la EMA lenta: cambio porcentual en slope_lookback barras
    slope_pct = ema_slow.pct_change(p.slope_lookback) * 100  # en %

    trending = adx >= p.adx_trend_min
    bull     = trending & (slope_pct >  p.slope_threshold)
    bear     = trending & (slope_pct < -p.slope_threshold)

    regime = pd.Series(LATERAL, index=adx.index, dtype=object)
    regime[bull] = BULL
    regime[bear] = BEAR

    return regime
