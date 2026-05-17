"""TrendATR v1 — XAUUSD H4 trend-following strategy.

Entry logic:
  - Macro trend: EMA(50) > EMA(200)
  - ADX(14) > adx_min_threshold  — confirms trend strength, filters ranging markets
  - Pullback: previous bar's low touched EMA(fast) and current close > EMA(fast)
  - Volatility gate: ATR(14) > atr_percentile_min-th pct of last atr_lookback bars
  - News blackout: ±news_blackout_minutes around FOMC/NFP/CPI/FED events

SL / TP:
  - SL = entry_close - sl_atr_mult × ATR
  - TP = entry_close + tp_atr_mult × ATR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from src.strategy.base import SIGNAL_COLS, Strategy


@dataclass
class TrendATRParams:
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_percentile_min: float = 40.0   # raised from 25 → filters more chop
    atr_lookback: int = 100
    adx_period: int = 14
    adx_min_threshold: float = 22.0    # new: ADX gate — no entry in ranging market
    sl_atr_mult: float = 2.0           # raised from 1.5 → more room per trade
    tp_atr_mult: float = 4.0           # raised from 3.0 → R:R 1:2 with wider SL
    trailing_activate_r: float = 1.5
    news_blackout_minutes: int = 30
    news_events: Sequence[datetime] = field(default_factory=list)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR (same as MT5 default)."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ADX — measures trend strength independent of direction."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    idx   = df.index

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=idx)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=idx)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    alpha    = 1 / period
    atr_w    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w * 100
    minus_di = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w * 100

    di_sum  = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx      = (di_diff / di_sum.replace(0, np.nan) * 100).fillna(0)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def _in_news_blackout(
    ts: pd.Timestamp,
    events: Sequence[datetime],
    window_minutes: int,
) -> bool:
    if not events:
        return False
    window = timedelta(minutes=window_minutes)
    for ev in events:
        ev_utc = ev if ev.tzinfo else ev.replace(tzinfo=timezone.utc)
        if abs(ts.to_pydatetime() - ev_utc) <= window:
            return True
    return False


class TrendATR(Strategy):
    """H4 trend-following strategy for XAUUSD — v1.1 with ADX filter."""

    def __init__(self, params: TrendATRParams | None = None) -> None:
        self._p = params or TrendATRParams()

    @property
    def name(self) -> str:
        return "TrendATR_v1"

    @property
    def min_warmup_bars(self) -> int:
        return self._p.ema_slow + self._p.atr_lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self._p
        out = pd.DataFrame(index=df.index)

        ema_f = _ema(df["close"], p.ema_fast)
        ema_m = _ema(df["close"], p.ema_medium)
        ema_s = _ema(df["close"], p.ema_slow)
        atr   = _atr(df, p.atr_period)
        adx   = _adx(df, p.adx_period)

        out[SIGNAL_COLS.EMA_FAST]   = ema_f
        out[SIGNAL_COLS.EMA_MEDIUM] = ema_m
        out[SIGNAL_COLS.EMA_SLOW]   = ema_s
        out[SIGNAL_COLS.ATR]        = atr

        atr_pct_min = atr.rolling(p.atr_lookback).quantile(p.atr_percentile_min / 100)

        trend_up = ema_m > ema_s
        adx_ok   = adx >= p.adx_min_threshold

        prev_low   = df["low"].shift(1)
        prev_close = df["close"].shift(1)
        touched_ema = (prev_low <= ema_f) & (prev_close >= ema_f)
        touched_ema = touched_ema | ((prev_close - ema_f).abs() <= 0.5 * atr)

        re_entry = df["close"] > ema_f
        vol_ok   = atr > atr_pct_min

        raw_signal = trend_up & adx_ok & touched_ema & re_entry & vol_ok

        news_mask = pd.Series(False, index=df.index)
        if p.news_events:
            for ts in df.index:
                if _in_news_blackout(ts, p.news_events, p.news_blackout_minutes):
                    news_mask.loc[ts] = True

        entry_long = raw_signal & ~news_mask
        entry_long.iloc[:self.min_warmup_bars] = False

        out[SIGNAL_COLS.ENTRY_LONG]  = entry_long
        out[SIGNAL_COLS.ENTRY_SHORT] = False

        sl_dist = p.sl_atr_mult * atr
        tp_dist = p.tp_atr_mult * atr
        out[SIGNAL_COLS.SL_PRICE] = np.where(entry_long, df["close"] - sl_dist, np.nan)
        out[SIGNAL_COLS.TP_PRICE] = np.where(entry_long, df["close"] + tp_dist, np.nan)

        return out
