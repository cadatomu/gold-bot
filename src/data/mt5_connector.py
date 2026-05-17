"""MT5Connector — thin wrapper around the MetaTrader5 Python package.

Design notes:
- Uses dependency injection for the mt5 module so tests never need a live terminal.
- All public methods raise typed exceptions; callers should NOT catch bare Exception.
- The context manager guarantees shutdown() even on crash.
- MetaTrader5 on Linux requires the terminal to be running (not just installed).
  On a headless VPS use MetaTrader5 Linux build or Wine + Windows build.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import the real MT5 package. Tests inject a mock via the constructor.
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as _mt5_default
except ImportError:  # pragma: no cover — only missing on CI without MT5
    _mt5_default = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MT5Error(Exception):
    """Base class for all MT5 errors."""


class MT5ConnectionError(MT5Error):
    """Terminal unreachable or login rejected."""


class MT5DataError(MT5Error):
    """Data retrieval failed (symbol not found, empty result, etc.)."""


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class MT5Connector:
    """Context-manager wrapper around MetaTrader5.

    Args:
        login: MT5 account number.
        password: Account password.
        server: Broker server name (e.g. "ICMarkets-Demo").
        terminal_path: Path to terminal64.exe / metatrader5 binary.
                       Pass None to let MT5 auto-detect.
        timeout_ms: Connection timeout in milliseconds.
        mt5_module: Injected MT5 module. Defaults to the real MetaTrader5 package.
                    Pass a mock in tests.

    Example::

        with MT5Connector(login=123, password="x", server="ICMarkets-Demo") as conn:
            df = conn.get_rates("XAUUSD", conn.TIMEFRAME_H4, count=500)
    """

    # Expose common timeframe constants so callers don't import MT5 directly.
    TIMEFRAME_M1:  int = 1
    TIMEFRAME_M5:  int = 5
    TIMEFRAME_M15: int = 15
    TIMEFRAME_M30: int = 30
    TIMEFRAME_H1:  int = 16385
    TIMEFRAME_H4:  int = 16388
    TIMEFRAME_D1:  int = 16408

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        terminal_path: Optional[str] = None,
        timeout_ms: int = 60_000,
        mt5_module: Any = None,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._timeout_ms = timeout_ms
        self._mt5 = mt5_module if mt5_module is not None else _mt5_default
        self._connected = False

        if self._mt5 is None:
            raise MT5ConnectionError(
                "MetaTrader5 package not available. "
                "Install it with: pip install MetaTrader5"
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialize MT5 terminal and log in.

        Raises:
            MT5ConnectionError: If terminal init or login fails.
        """
        init_kwargs: dict[str, Any] = {"timeout": self._timeout_ms}
        if self._terminal_path:
            init_kwargs["path"] = self._terminal_path

        if not self._mt5.initialize(**init_kwargs):
            code, msg = self._mt5.last_error()
            raise MT5ConnectionError(f"MT5 initialize() failed [{code}]: {msg}")

        authorized = self._mt5.login(
            login=self._login,
            password=self._password,
            server=self._server,
        )
        if not authorized:
            code, msg = self._mt5.last_error()
            self._mt5.shutdown()
            raise MT5ConnectionError(
                f"MT5 login failed for account {self._login} on {self._server} "
                f"[{code}]: {msg}"
            )

        self._connected = True
        info = self._mt5.account_info()
        log.info(
            "mt5_connected",
            account=self._login,
            server=self._server,
            balance=getattr(info, "balance", None),
            currency=getattr(info, "currency", None),
        )

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if self._connected:
            self._mt5.shutdown()
            self._connected = False
            log.info("mt5_disconnected", account=self._login)

    # ------------------------------------------------------------------
    # Symbol info
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Return key specs for a symbol.

        Args:
            symbol: e.g. "XAUUSD"

        Returns:
            Dict with point, digits, trade_contract_size, volume_min, etc.

        Raises:
            MT5DataError: Symbol not found or not visible.
        """
        self._assert_connected()
        info = self._mt5.symbol_info(symbol)
        if info is None:
            code, msg = self._mt5.last_error()
            raise MT5DataError(f"symbol_info({symbol!r}) failed [{code}]: {msg}")

        return {
            "symbol":              info.name,
            "digits":              info.digits,
            "point":               info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min":          info.volume_min,
            "volume_max":          info.volume_max,
            "volume_step":         info.volume_step,
            "spread":              info.spread,
        }

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_rates(
        self,
        symbol: str,
        timeframe: int,
        count: int,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """Download the most recent `count` bars from position `start_pos`.

        Args:
            symbol: Instrument (e.g. "XAUUSD").
            timeframe: One of the TIMEFRAME_* constants.
            count: Number of bars to retrieve.
            start_pos: 0 = most recent bar.

        Returns:
            DataFrame with columns [open, high, low, close, tick_volume, spread, real_volume]
            indexed by UTC datetime.

        Raises:
            MT5DataError: Empty result or MT5 error.
        """
        self._assert_connected()
        rates = self._mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        return self._rates_to_df(rates, symbol, timeframe)

    def get_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        """Download bars within a date range (UTC).

        Args:
            symbol: Instrument (e.g. "XAUUSD").
            timeframe: One of the TIMEFRAME_* constants.
            date_from: Start datetime (UTC, inclusive).
            date_to: End datetime (UTC, inclusive).

        Returns:
            DataFrame indexed by UTC datetime.

        Raises:
            MT5DataError: Empty result or MT5 error.
        """
        self._assert_connected()
        rates = self._mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        return self._rates_to_df(rates, symbol, timeframe)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self._connected:
            raise MT5ConnectionError("Not connected. Use as context manager or call connect() first.")

    def _rates_to_df(self, rates: Any, symbol: str, timeframe: int) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            code, msg = self._mt5.last_error()
            raise MT5DataError(
                f"No data returned for {symbol} tf={timeframe} [{code}]: {msg}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})

        # Keep only OHLCV + spread; drop real_volume (often 0 for CFDs)
        keep = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in df.columns]
        df = df[keep].copy()
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(np.float64)

        log.debug(
            "rates_fetched",
            symbol=symbol,
            bars=len(df),
            from_=str(df.index[0]),
            to=str(df.index[-1]),
        )
        return df
