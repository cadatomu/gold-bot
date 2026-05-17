"""Order manager — wraps MT5Connector for live trading.

Responsibilities:
  - Gate every trade through CircuitBreaker and position sizing
  - Open / close / modify orders via MT5
  - Track the single open position (max_concurrent_trades = 1)
  - Log every action with structlog

This module never calculates signals. It receives a pre-computed
entry request and translates it into MT5 orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from src.data.mt5_connector import MT5Connector, MT5DataError
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizing import SizingConfig, calc_lot_size

log = structlog.get_logger(__name__)


@dataclass
class TradeRequest:
    symbol:      str
    direction:   int        # +1 long, -1 short (v1: always +1)
    entry_price: float      # theoretical fill (signal bar close)
    sl_price:    float
    tp_price:    float
    comment:     str = ""


@dataclass
class OpenPosition:
    ticket:      int
    symbol:      str
    direction:   int
    lots:        float
    entry_price: float
    sl_price:    float
    tp_price:    float
    comment:     str = ""


@dataclass
class OrderManagerConfig:
    symbol:              str   = "XAUUSD"
    magic:               int   = 20240101
    slippage_points:     int   = 30         # MT5 max slippage
    sizing:              SizingConfig = field(default_factory=SizingConfig)
    max_daily_dd_pct:    float = 0.03
    max_total_dd_pct:    float = 0.15
    state_dir:           Path  = Path("live_state")


# MT5 order/type constants (mirror the MetaTrader5 module values)
_ORDER_TYPE_BUY  = 0
_ORDER_TYPE_SELL = 1
_TRADE_ACTION_DEAL   = 1
_TRADE_ACTION_SLTP   = 6
_TRADE_ACTION_REMOVE = 8
_ORDER_FILLING_IOC   = 1


class OrderManager:
    """
    Manages live order lifecycle against an MT5 terminal.

    Parameters
    ----------
    connector : Connected MT5Connector instance
    equity    : Starting account equity (used to initialise CircuitBreaker)
    cfg       : OrderManagerConfig
    """

    def __init__(
        self,
        connector: MT5Connector,
        equity: float,
        cfg: OrderManagerConfig | None = None,
    ) -> None:
        self._conn  = connector
        self._cfg   = cfg or OrderManagerConfig()
        self._pos:  Optional[OpenPosition] = None

        cb_path = self._cfg.state_dir / "circuit_breaker.json"
        self._cb = CircuitBreaker(
            cb_path,
            max_daily_dd_pct = self._cfg.max_daily_dd_pct,
            max_total_dd_pct = self._cfg.max_total_dd_pct,
        )
        self._cb.load_or_init(equity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_trade(self, request: TradeRequest, equity: float) -> Optional[OpenPosition]:
        """
        Send a market order if risk checks pass.

        Returns the OpenPosition on success, None if blocked by circuit
        breaker or if a position is already open.
        """
        self._cb.update(equity)

        if self._cb.is_halted():
            log.warning("order_blocked_by_circuit_breaker",
                        daily=self._cb.is_daily_halted(),
                        total=self._cb.is_total_halted())
            return None

        if self._pos is not None:
            log.warning("open_trade_rejected_position_already_open",
                        ticket=self._pos.ticket)
            return None

        lots = calc_lot_size(
            equity      = equity,
            sl_distance = abs(request.entry_price - request.sl_price),
            entry_price = request.entry_price,
            cfg         = self._cfg.sizing,
        )

        order_type = _ORDER_TYPE_BUY if request.direction == 1 else _ORDER_TYPE_SELL

        result = self._conn._mt5.order_send({
            "action":   _TRADE_ACTION_DEAL,
            "symbol":   request.symbol,
            "volume":   lots,
            "type":     order_type,
            "price":    request.entry_price,
            "sl":       request.sl_price,
            "tp":       request.tp_price,
            "deviation":self._cfg.slippage_points,
            "magic":    self._cfg.magic,
            "comment":  request.comment,
            "type_filling": _ORDER_FILLING_IOC,
        })

        if result is None or result.retcode != 10009:  # TRADE_RETCODE_DONE
            log.error("order_send_failed",
                      retcode=getattr(result, "retcode", None),
                      comment=getattr(result, "comment", ""))
            return None

        self._pos = OpenPosition(
            ticket      = result.order,
            symbol      = request.symbol,
            direction   = request.direction,
            lots        = lots,
            entry_price = result.price,
            sl_price    = request.sl_price,
            tp_price    = request.tp_price,
            comment     = request.comment,
        )
        log.info("trade_opened", ticket=self._pos.ticket, lots=lots,
                 entry=result.price, sl=request.sl_price, tp=request.tp_price)
        return self._pos

    def close_trade(self, equity: float) -> bool:
        """
        Close the current open position at market.
        Returns True on success.
        """
        if self._pos is None:
            log.warning("close_trade_no_position")
            return False

        close_type = _ORDER_TYPE_SELL if self._pos.direction == 1 else _ORDER_TYPE_BUY

        result = self._conn._mt5.order_send({
            "action":    _TRADE_ACTION_DEAL,
            "symbol":    self._pos.symbol,
            "volume":    self._pos.lots,
            "type":      close_type,
            "position":  self._pos.ticket,
            "deviation": self._cfg.slippage_points,
            "magic":     self._cfg.magic,
            "comment":   "close",
            "type_filling": _ORDER_FILLING_IOC,
        })

        if result is None or result.retcode != 10009:
            log.error("close_order_failed",
                      retcode=getattr(result, "retcode", None))
            return False

        log.info("trade_closed", ticket=self._pos.ticket,
                 close_price=result.price)
        self._pos = None
        self._cb.update(equity)
        return True

    def modify_sl(self, new_sl: float) -> bool:
        """
        Modify the SL of the open position (e.g. trailing or breakeven).
        Returns True on success.
        """
        if self._pos is None:
            log.warning("modify_sl_no_position")
            return False

        result = self._conn._mt5.order_send({
            "action":   _TRADE_ACTION_SLTP,
            "symbol":   self._pos.symbol,
            "position": self._pos.ticket,
            "sl":       new_sl,
            "tp":       self._pos.tp_price,
        })

        if result is None or result.retcode != 10009:
            log.error("modify_sl_failed",
                      retcode=getattr(result, "retcode", None))
            return False

        old_sl          = self._pos.sl_price
        self._pos.sl_price = new_sl
        log.info("sl_modified", ticket=self._pos.ticket,
                 old_sl=old_sl, new_sl=new_sl)
        return True

    @property
    def position(self) -> Optional[OpenPosition]:
        return self._pos

    @property
    def is_halted(self) -> bool:
        return self._cb.is_halted()
