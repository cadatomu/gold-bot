"""Conector para Interactive Brokers via ib_insync.

Requiere TWS o IB Gateway corriendo en el mismo equipo con API habilitada.

Credenciales en .env:
  IB_HOST      — host de TWS/Gateway (default: 127.0.0.1)
  IB_PORT      — 7497=TWS paper | 7496=TWS live | 4002=Gateway paper | 4001=Gateway live
  IB_CLIENT_ID — ID de cliente (cualquier número, default: 1)
"""

from __future__ import annotations

import os
from typing import Optional

from ib_insync import IB, Contract, MarketOrder, LimitOrder, StopOrder, util

util.logToConsole(level=30)   # solo warnings


GOLD_SYMBOL   = "GC"
GOLD_SECTYPE  = "CONTFUT"     # contrato continuo de futuros oro COMEX
GOLD_EXCHANGE = "COMEX"
GOLD_CURRENCY = "USD"


def _gold_contract() -> Contract:
    c = Contract()
    c.symbol   = GOLD_SYMBOL
    c.secType  = GOLD_SECTYPE
    c.exchange = GOLD_EXCHANGE
    c.currency = GOLD_CURRENCY
    return c


class IBConnector:
    """
    Conector IB para XAUUSD spot (IDEALPRO).

    Uso:
        conn = IBConnector.from_env()
        conn.connect()
        price = conn.get_price()
        conn.place_market_order("BUY", quantity=100, sl_price=3200.0, tp_price=3400.0)
        conn.disconnect()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1) -> None:
        self._host      = host
        self._port      = port
        self._client_id = client_id
        self._ib        = IB()
        self._contract  = _gold_contract()

    @classmethod
    def from_env(cls) -> "IBConnector":
        return cls(
            host      = os.environ.get("IB_HOST",      "127.0.0.1"),
            port      = int(os.environ.get("IB_PORT",  "7497")),
            client_id = int(os.environ.get("IB_CLIENT_ID", "1")),
        )

    # ── Conexión ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._ib.connect(self._host, self._port, clientId=self._client_id)
        self._ib.qualifyContracts(self._contract)
        print(f"[IB] Conectado — {self._host}:{self._port} (clientId={self._client_id})")

    def disconnect(self) -> None:
        self._ib.disconnect()

    # ── Cuenta ────────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        vals = self._ib.accountSummary()
        return {v.tag: v.value for v in vals}

    def get_balance(self) -> float:
        summary = self.get_account_summary()
        return float(summary.get("NetLiquidation", 0))

    # ── Precio ────────────────────────────────────────────────────────────────

    def get_price(self) -> dict:
        ticker = self._ib.reqMktData(self._contract, "", False, False)
        self._ib.sleep(1)
        return {
            "bid":  ticker.bid,
            "ask":  ticker.ask,
            "last": ticker.last,
        }

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        side:     str,              # "BUY" o "SELL"
        quantity: float,            # en oz (XAUUSD IDEALPRO mín = 100 oz)
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> dict:
        """
        Abre posición de mercado con SL/TP opcionales como órdenes bracket.

        quantity : oz de oro (mínimo 100 oz en IDEALPRO)
        """
        entry = MarketOrder(side.upper(), quantity)
        trade = self._ib.placeOrder(self._contract, entry)
        self._ib.sleep(1)

        orders = []
        if sl_price is not None:
            sl_side  = "SELL" if side.upper() == "BUY" else "BUY"
            sl_order = StopOrder(sl_side, quantity, sl_price,
                                 parentId=trade.order.orderId,
                                 transmit=(tp_price is None))
            self._ib.placeOrder(self._contract, sl_order)
            orders.append("SL")

        if tp_price is not None:
            tp_side  = "SELL" if side.upper() == "BUY" else "BUY"
            tp_order = LimitOrder(tp_side, quantity, tp_price,
                                  parentId=trade.order.orderId,
                                  transmit=True)
            self._ib.placeOrder(self._contract, tp_order)
            orders.append("TP")

        print(f"[IB] Orden {side} {quantity}oz XAUUSD — orderId={trade.order.orderId}"
              + (f" [{', '.join(orders)}]" if orders else ""))
        return {"order_id": trade.order.orderId, "status": trade.orderStatus.status}

    def close_position(self, side: str, quantity: float) -> dict:
        """Cierra posición con orden de mercado en dirección contraria."""
        close_side = "SELL" if side.upper() == "BUY" else "BUY"
        order = MarketOrder(close_side, quantity)
        trade = self._ib.placeOrder(self._contract, order)
        self._ib.sleep(1)
        print(f"[IB] Cierre {close_side} {quantity}oz")
        return {"order_id": trade.order.orderId}

    def get_positions(self) -> list:
        return [
            {"symbol": p.contract.symbol, "qty": p.position,
             "avg_cost": p.avgCost, "pnl": p.unrealizedPNL}
            for p in self._ib.positions()
            if p.contract.symbol == GOLD_SYMBOL
        ]

    def cancel_all_orders(self) -> None:
        for order in self._ib.openOrders():
            self._ib.cancelOrder(order)
        print("[IB] Todas las órdenes canceladas.")

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()
