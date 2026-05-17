"""Live trading main loop for XAUUSD H4.

Scheduling logic:
  - H4 bars close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
  - After each bar closes, wait BAR_CLOSE_DELAY_SECONDS then fetch the
    latest data, generate signals, and act if warranted
  - The scheduler runs in a tight poll loop and sleeps for short
    intervals to remain responsive to shutdown signals

Entry flow per bar:
  1. Download latest N bars from MT5 (refreshes cache)
  2. Run strategy.generate_signals()
  3. If entry_long on the LAST completed bar AND no position open:
       → OrderManager.open_trade()
  4. If position open: check trailing SL logic (post-FASE 5)
  5. Update CircuitBreaker with current equity
  6. Log state
"""

from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from src.data.historical import download_history
from src.data.mt5_connector import MT5Connector
from src.execution.order_manager import OrderManager, OrderManagerConfig, TradeRequest
from src.monitoring.logger import configure_logging
from src.risk.position_sizing import SizingConfig
from src.strategy.base import SIGNAL_COLS
from src.strategy.trend_atr import TrendATR, TrendATRParams

log = structlog.get_logger(__name__)

H4_SECONDS      = 4 * 3600
BAR_CLOSE_DELAY = 5   # seconds after bar close before fetching data
POLL_INTERVAL   = 30  # seconds between scheduler ticks
HISTORY_BARS    = 350  # bars to load for indicator warmup + signal


def _next_h4_close(now: datetime) -> datetime:
    """Return the next H4 bar-close timestamp after `now` (UTC)."""
    ts = now.timestamp()
    remainder = ts % H4_SECONDS
    next_close = ts + (H4_SECONDS - remainder)
    return datetime.fromtimestamp(next_close, tz=timezone.utc)


def _seconds_until(target: datetime) -> float:
    now = datetime.now(tz=timezone.utc)
    return max(0.0, (target - now).total_seconds())


class LiveLoop:
    """
    Encapsulates the live trading loop so it can be unit-tested
    without running the actual blocking loop.

    Parameters
    ----------
    connector    : Connected MT5Connector
    order_mgr    : OrderManager (already initialised)
    strategy     : Strategy instance
    symbol       : Trading symbol
    timeframe    : MT5 timeframe constant
    cache_dir    : Path for OHLCV cache
    """

    def __init__(
        self,
        connector:  MT5Connector,
        order_mgr:  OrderManager,
        strategy:   TrendATR,
        symbol:     str = "XAUUSD",
        timeframe:  int = MT5Connector.TIMEFRAME_H4,
        cache_dir:  Path = Path("data/cache"),
    ) -> None:
        self._conn      = connector
        self._om        = order_mgr
        self._strategy  = strategy
        self._symbol    = symbol
        self._tf        = timeframe
        self._cache_dir = cache_dir
        self._running   = False

    def run(self) -> None:
        """Block forever, processing one H4 bar at a time."""
        self._running = True

        def _stop(sig, frame):  # noqa: ANN001
            log.info("shutdown_signal_received", signal=sig)
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        log.info("live_loop_started", symbol=self._symbol)

        while self._running:
            now       = datetime.now(tz=timezone.utc)
            next_bar  = _next_h4_close(now)
            wait_secs = _seconds_until(next_bar)

            log.debug("waiting_for_bar_close",
                      next_bar=next_bar.isoformat(),
                      wait_seconds=round(wait_secs, 1))

            # Sleep in POLL_INTERVAL chunks so SIGTERM is handled promptly
            slept = 0.0
            while slept < wait_secs and self._running:
                chunk = min(POLL_INTERVAL, wait_secs - slept)
                time.sleep(chunk)
                slept += chunk

            if not self._running:
                break

            # Brief delay after bar close so MT5 finalises the bar
            time.sleep(BAR_CLOSE_DELAY)

            try:
                self.process_bar()
            except Exception:
                log.exception("bar_processing_error")

        log.info("live_loop_stopped")

    def process_bar(self) -> None:
        """
        Fetch data, generate signals, and act on the latest completed bar.
        Called once per H4 bar close. Also callable directly in tests.
        """
        df = download_history(
            self._conn, self._symbol, self._tf,
            years=1, cache_dir=self._cache_dir,
        )

        if len(df) < self._strategy.min_warmup_bars:
            log.warning("insufficient_bars", have=len(df),
                        need=self._strategy.min_warmup_bars)
            return

        signals = self._strategy.generate_signals(df)

        # Act on the last COMPLETED bar (index -1; the current forming bar
        # is not yet in the H4 dataset because we fetch after close)
        last_sig = signals.iloc[-1]
        last_bar = df.iloc[-1]

        account = self._conn._mt5.account_info()
        equity  = account.balance if account else 0.0

        if last_sig[SIGNAL_COLS.ENTRY_LONG] and self._om.position is None:
            req = TradeRequest(
                symbol      = self._symbol,
                direction   = 1,
                entry_price = last_bar["close"],
                sl_price    = last_sig[SIGNAL_COLS.SL_PRICE],
                tp_price    = last_sig[SIGNAL_COLS.TP_PRICE],
                comment     = f"TrendATR H4 {datetime.now(tz=timezone.utc).date()}",
            )
            self._om.open_trade(req, equity)

        log.info(
            "bar_processed",
            bar_time   = str(df.index[-1]),
            entry_long = bool(last_sig[SIGNAL_COLS.ENTRY_LONG]),
            has_pos    = self._om.position is not None,
            halted     = self._om.is_halted,
            equity     = round(equity, 2),
        )


def create_live_loop(
    login:        int,
    password:     str,
    server:       str,
    terminal_path: Optional[str] = None,
    cache_dir:    Path = Path("data/cache"),
    state_dir:    Path = Path("live_state"),
    log_level:    str  = "INFO",
    log_fmt:      str  = "json",
) -> LiveLoop:
    """
    Factory that wires up all components and returns a ready LiveLoop.
    The caller must call loop.run() inside a connected MT5Connector context.
    """
    configure_logging(level=log_level, fmt=log_fmt)

    connector = MT5Connector(
        login         = login,
        password      = password,
        server        = server,
        terminal_path = terminal_path,
    )
    connector.connect()

    account = connector._mt5.account_info()
    equity  = account.balance if account else 10_000.0

    om_cfg = OrderManagerConfig(
        symbol    = "XAUUSD",
        sizing    = SizingConfig(),
        state_dir = state_dir,
    )
    order_mgr = OrderManager(connector, equity, om_cfg)
    strategy  = TrendATR()

    return LiveLoop(
        connector = connector,
        order_mgr = order_mgr,
        strategy  = strategy,
        cache_dir = cache_dir,
    )
