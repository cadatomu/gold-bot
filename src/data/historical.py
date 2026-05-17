"""Historical data download and cache management.

Downloads XAUUSD H4 from MT5, persists as Parquet, validates data quality.

Gap detection logic:
  XAUUSD trades Sunday 22:00 UTC → Friday 22:00 UTC (no gaps Mon–Fri).
  Expected H4 bars per week: ~30 (5 days × ~6 bars/day, minus weekend).
  A "gap" is any missing bar during expected trading hours.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.mt5_connector import MT5Connector, MT5DataError
from src.monitoring.logger import get_logger

log = get_logger(__name__)

# XAUUSD H4 — Forex/CFD market hours (UTC)
_MARKET_OPEN_WEEKDAY  = 4   # Sunday = 6 in Python, but pandas uses Mon=0 convention
_MARKET_CLOSE_WEEKDAY = 4   # Friday
_WEEKEND_DAYS = {5, 6}      # Saturday=5, Sunday=6 in datetime.weekday()
_H4_FREQ = "4h"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_history(
    connector: MT5Connector,
    symbol: str,
    timeframe: int,
    years: int = 10,
    cache_dir: str | Path = "data/cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download `years` of H4 history for `symbol`, cache as Parquet.

    On subsequent calls, loads from cache and fetches only the missing tail
    (incremental update).

    Args:
        connector: Connected MT5Connector instance.
        symbol: e.g. "XAUUSD"
        timeframe: MT5Connector.TIMEFRAME_H4
        years: Years of history to target.
        cache_dir: Directory for Parquet files.
        force_refresh: Ignore cache and re-download everything.

    Returns:
        DataFrame [open, high, low, close, volume] indexed by UTC datetime,
        validated and cleaned.

    Raises:
        MT5DataError: If download fails and no cache exists.
    """
    cache_path = _cache_path(cache_dir, symbol, timeframe)

    if not force_refresh and cache_path.exists():
        cached = _load_parquet(cache_path)
        tail_start = cached.index[-1] + timedelta(hours=4)
        if _needs_update(cached.index[-1]):
            log.info("cache_update", symbol=symbol, from_=str(tail_start))
            fresh = _fetch_range(connector, symbol, timeframe, tail_start, _now_utc())
            df = pd.concat([cached, fresh]).pipe(_deduplicate)
        else:
            log.info("cache_hit", symbol=symbol, bars=len(cached))
            df = cached
    else:
        date_from = _now_utc() - timedelta(days=years * 365)
        log.info("full_download", symbol=symbol, from_=str(date_from), years=years)
        df = _fetch_range(connector, symbol, timeframe, date_from, _now_utc())

    df = _clean(df)
    gaps = find_gaps(df, timeframe)
    if gaps:
        log.warning("data_gaps_detected", symbol=symbol, count=len(gaps), gaps=gaps[:5])

    _save_parquet(df, cache_path)
    log.info(
        "history_ready",
        symbol=symbol,
        bars=len(df),
        from_=str(df.index[0].date()),
        to=str(df.index[-1].date()),
        gaps=len(gaps),
    )
    return df


def find_gaps(df: pd.DataFrame, timeframe: int) -> list[tuple[str, str]]:
    """Detect missing bars in OHLCV data during expected trading hours.

    A gap is a period longer than 2× the bar duration with no data,
    excluding weekends (Sat 22:00 → Sun 22:00 UTC for XAUUSD).

    Args:
        df: DataFrame indexed by UTC datetime.
        timeframe: MT5Connector.TIMEFRAME_* constant (used to infer bar duration).

    Returns:
        List of (gap_start, gap_end) string tuples.
    """
    if len(df) < 2:
        return []

    bar_hours = _timeframe_hours(timeframe)
    max_gap = timedelta(hours=bar_hours * 2)
    gaps: list[tuple[str, str]] = []

    diffs = df.index.to_series().diff().dropna()
    for ts, delta in diffs.items():
        if delta <= max_gap:
            continue
        # Allow weekend gap: Friday close to Sunday open (~48h)
        prev_ts = ts - delta
        if _is_weekend_gap(prev_ts, ts):
            continue
        gaps.append((str(prev_ts), str(ts)))

    return gaps


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return list of data quality issues found in df.

    Checks: missing values, negative prices, high < low, zero volume.
    """
    issues: list[str] = []

    null_counts = df[["open", "high", "low", "close"]].isnull().sum()
    for col, n in null_counts.items():
        if n > 0:
            issues.append(f"{n} null values in column '{col}'")

    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        issues.append(f"{bad_hl} bars where high < low")

    for col in ["open", "high", "low", "close"]:
        neg = (df[col] <= 0).sum()
        if neg > 0:
            issues.append(f"{neg} non-positive prices in '{col}'")

    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"{zero_vol} bars with zero volume")

    return issues


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fetch_range(
    connector: MT5Connector,
    symbol: str,
    timeframe: int,
    date_from: datetime,
    date_to: datetime,
) -> pd.DataFrame:
    """Fetch via get_rates_range, chunking if needed."""
    chunks: list[pd.DataFrame] = []
    # MT5 caps at ~100k bars per call; H4 = 6 bars/day = ~21 900 bars/10yr — fits in one call.
    chunk = connector.get_rates_range(symbol, timeframe, date_from, date_to)
    chunks.append(chunk)
    return pd.concat(chunks).pipe(_deduplicate) if len(chunks) > 1 else chunks[0]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="last")].sort_index()


def _cache_path(cache_dir: str | Path, symbol: str, timeframe: int) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{symbol}_TF{timeframe}.parquet"


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    log.debug("parquet_saved", path=str(path), rows=len(df))


def _needs_update(last_ts: pd.Timestamp, staleness_hours: int = 4) -> bool:
    """True if the cache tail is older than one bar duration."""
    age = _now_utc() - last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return age > timedelta(hours=staleness_hours)


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _timeframe_hours(timeframe: int) -> int:
    mapping = {1: 0, 5: 0, 15: 0, 30: 0, 16385: 1, 16388: 4, 16408: 24}
    return mapping.get(timeframe, 4)


def _is_weekend_gap(prev_ts: pd.Timestamp, next_ts: pd.Timestamp) -> bool:
    """True if the gap spans a weekend (Fri close → Sun open for XAUUSD)."""
    delta = next_ts - prev_ts
    if delta > timedelta(hours=72):
        return False
    prev_wd = prev_ts.weekday()   # Mon=0 … Fri=4, Sat=5, Sun=6
    next_wd = next_ts.weekday()
    # Gap is weekend if prev is Fri/Sat and next is Sun/Mon
    return prev_wd in {4, 5} and next_wd in {6, 0}
