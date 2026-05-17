"""Yahoo Finance data source — fallback for Linux environments without MT5.

Downloads GC=F (Gold Futures) 1h OHLCV data and resamples to H4.
GC=F tracks XAUUSD with >99% correlation; prices differ by ~0-3 USD
due to futures basis, which does not materially affect signal generation.

Limitations:
  - Yahoo Finance caps 1h history at ~730 days (~17 months of H4 bars)
  - For longer histories, daily data is available but not suitable for H4 strategy
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


GOLD_TICKER  = "GC=F"
_CACHE_FILE  = "gcf_h4.parquet"


def _download_raw(period: str = "730d") -> pd.DataFrame:
    raw = yf.download(
        GOLD_TICKER,
        period   = period,
        interval = "1h",
        progress = False,
        auto_adjust = True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.DatetimeIndex(raw.index).tz_convert("UTC")
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    return raw


def _resample_to_h4(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h bars to H4, aligning to 00:00 UTC origin."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    h4 = df_1h.resample("4h", origin="start_day").agg(agg).dropna()
    # Drop bars with zero range (gaps / holiday artifacts)
    h4 = h4[h4["high"] > h4["low"]]
    return h4


def load_h4(
    cache_dir: Path | None = None,
    force_refresh: bool    = False,
) -> pd.DataFrame:
    """
    Return XAUUSD-equivalent H4 OHLCV data from Yahoo Finance (GC=F).

    Caches to parquet so repeated calls don't hit the network.

    Parameters
    ----------
    cache_dir     : Directory for the parquet cache (None = no cache)
    force_refresh : Ignore cache and re-download

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume], UTC index.
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / _CACHE_FILE
        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
            # Refresh if last bar is older than 8h
            age_hours = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 3600
            if age_hours < 8:
                return df

    raw = _download_raw()
    h4  = _resample_to_h4(raw)

    if cache_dir is not None:
        cache_path = Path(cache_dir) / _CACHE_FILE
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        h4.to_parquet(cache_path)

    return h4
