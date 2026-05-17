"""Abstract base class for all strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalColumns:
    """Column names written into the signals DataFrame."""
    ENTRY_LONG:  str = "entry_long"   # bool — bar where a long entry fires
    ENTRY_SHORT: str = "entry_short"  # bool — bar where a short entry fires
    SL_PRICE:    str = "sl_price"     # float — stop-loss price for that bar's trade
    TP_PRICE:    str = "tp_price"     # float — take-profit price for that bar's trade
    ATR:         str = "atr"          # float — ATR value used for sizing
    EMA_FAST:    str = "ema_fast"
    EMA_MEDIUM:  str = "ema_medium"
    EMA_SLOW:    str = "ema_slow"


SIGNAL_COLS = SignalColumns()


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Subclasses must implement `generate_signals`. The returned DataFrame must
    share the same DatetimeIndex as `df` and contain at least the columns
    defined in SIGNAL_COLS.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry/exit signals from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns [open, high, low, close, volume].
            Index must be a UTC-aware DatetimeIndex sorted ascending.

        Returns
        -------
        pd.DataFrame
            Same index as `df`. Columns as per SIGNAL_COLS.
            Rows before indicator warmup will have NaN in indicator columns
            and False in entry columns.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @property
    @abstractmethod
    def min_warmup_bars(self) -> int:
        """Minimum number of bars required before the first valid signal."""
