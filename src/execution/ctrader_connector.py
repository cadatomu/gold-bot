"""Conector para IC Markets cTrader Open API.

Credenciales requeridas en .env:
  CTRADER_CLIENT_ID      — App Client ID del portal Open API
  CTRADER_CLIENT_SECRET  — App Client Secret
  CTRADER_ACCESS_TOKEN   — OAuth2 token para la cuenta
  CTRADER_ACCOUNT_ID     — ID numérico de la cuenta (ej. 10027534)
  CTRADER_DEMO           — "true" para demo, "false" para live
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOANewOrderReq,
    ProtoOAClosePositionReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAReconcileReq,
    ProtoOATraderReq,
    ProtoOASpotEvent,
)
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
)


DEMO_HOST = EndPoints.PROTOBUF_DEMO_HOST
LIVE_HOST = EndPoints.PROTOBUF_LIVE_HOST
PORT      = EndPoints.PROTOBUF_PORT

# XAUUSD symbol ID en IC Markets cTrader
# Se obtiene dinámicamente via SymbolsList — este es el valor estándar
XAUUSD_SYMBOL_NAME = "XAUUSD"


@dataclass
class CTraderConfig:
    client_id:     str
    client_secret: str
    access_token:  str
    account_id:    int
    is_demo:       bool = True


@dataclass
class TradeResult:
    success:     bool
    order_id:    Optional[int]  = None
    position_id: Optional[int]  = None
    error_msg:   str            = ""


class CTraderConnector:
    """
    Conector sincrónico simplificado sobre la cTrader Open API (Twisted).

    Uso:
        cfg = CTraderConfig.from_env()
        conn = CTraderConnector(cfg)
        conn.connect()          # autentica y suscribe
        conn.place_market_order("XAUUSD", "BUY", volume_lots=0.01,
                                sl_price=3250.0, tp_price=3400.0)
        conn.disconnect()
    """

    def __init__(self, cfg: CTraderConfig) -> None:
        self._cfg         = cfg
        self._client      = None
        self._symbol_id   = None
        self._connected   = False
        self._ready       = threading.Event()
        self._positions: dict[int, dict] = {}

    # ── Construcción desde variables de entorno ───────────────────────────

    @classmethod
    def from_env(cls) -> "CTraderConnector":
        cfg = CTraderConfig(
            client_id     = os.environ["CTRADER_CLIENT_ID"],
            client_secret = os.environ["CTRADER_CLIENT_SECRET"],
            access_token  = os.environ["CTRADER_ACCESS_TOKEN"],
            account_id    = int(os.environ["CTRADER_ACCOUNT_ID"]),
            is_demo       = os.environ.get("CTRADER_DEMO", "true").lower() == "true",
        )
        return cls(cfg)

    # ── Conexión ──────────────────────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> None:
        """Conecta, autentica la app y la cuenta. Bloquea hasta estar listo."""
        host = DEMO_HOST if self._cfg.is_demo else LIVE_HOST

        self._client = Client(host, PORT, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        # Twisted corre en hilo separado
        self._thread = threading.Thread(target=self._client.startService, daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout):
            raise TimeoutError("cTrader: timeout al conectar / autenticar")

    def disconnect(self) -> None:
        if self._client:
            self._client.stopService()
        self._connected = False

    # ── Callbacks internos ────────────────────────────────────────────────

    def _on_connected(self, client, _):
        req = ProtoOAApplicationAuthReq()
        req.clientId     = self._cfg.client_id
        req.clientSecret = self._cfg.client_secret
        deferred = client.send(req)
        deferred.addErrback(self._on_error)

    def _on_disconnected(self, client, reason):
        self._connected = False

    def _on_message(self, client, message):
        msg_type = message.payloadType

        # App autenticada → autenticar cuenta
        if msg_type == Protobuf.APP_AUTH_RES:
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self._cfg.account_id
            req.accessToken         = self._cfg.access_token
            client.send(req).addErrback(self._on_error)

        # Cuenta autenticada → obtener lista de símbolos
        elif msg_type == Protobuf.ACCOUNT_AUTH_RES:
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = self._cfg.account_id
            req.includeArchivedSymbols = False
            client.send(req).addErrback(self._on_error)

        # Símbolos recibidos → buscar XAUUSD
        elif msg_type == Protobuf.SYMBOLS_LIST_RES:
            res = Protobuf.extract(message)
            for sym in res.symbol:
                if sym.symbolName == XAUUSD_SYMBOL_NAME:
                    self._symbol_id = sym.symbolId
                    break
            self._connected = True
            self._ready.set()

        # Errores
        elif msg_type == Protobuf.ERROR_RES:
            res = Protobuf.extract(message)
            self._on_error(Exception(f"cTrader error {res.errorCode}: {res.description}"))

    def _on_error(self, error):
        print(f"[CTraderConnector] Error: {error}")

    # ── Operaciones de trading ────────────────────────────────────────────

    def place_market_order(
        self,
        symbol:       str,
        side:         str,           # "BUY" o "SELL"
        volume_lots:  float,
        sl_price:     Optional[float] = None,
        tp_price:     Optional[float] = None,
        label:        str = "gold_bot",
    ) -> TradeResult:
        """
        Envía una orden de mercado.

        volume_lots : lotes estándar (0.01 = 1 micro lot = 1 oz)
        sl_price    : Stop Loss en precio absoluto
        tp_price    : Take Profit en precio absoluto
        """
        if not self._connected or self._symbol_id is None:
            return TradeResult(success=False, error_msg="No conectado")

        volume_units = int(volume_lots * 100_000)  # cTrader usa unidades (100k = 1 lot)

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self._cfg.account_id
        req.symbolId            = self._symbol_id
        req.orderType           = ProtoOAOrderType.Value("MARKET")
        req.tradeSide           = ProtoOATradeSide.Value(side.upper())
        req.volume              = volume_units
        req.label               = label

        if sl_price is not None:
            req.relativeStopLoss  = 0       # usamos precio absoluto
            req.stopLoss          = int(sl_price * 100_000)

        if tp_price is not None:
            req.relativeStopLoss  = 0
            req.takeProfit        = int(tp_price * 100_000)

        deferred = self._client.send(req)
        deferred.addErrback(self._on_error)

        return TradeResult(success=True)

    def close_position(self, position_id: int, volume_lots: float) -> TradeResult:
        """Cierra parcial o totalmente una posición por su ID."""
        if not self._connected:
            return TradeResult(success=False, error_msg="No conectado")

        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self._cfg.account_id
        req.positionId          = position_id
        req.volume              = int(volume_lots * 100_000)

        self._client.send(req).addErrback(self._on_error)
        return TradeResult(success=True)

    def modify_sl_tp(
        self,
        position_id: int,
        sl_price:    Optional[float] = None,
        tp_price:    Optional[float] = None,
    ) -> TradeResult:
        """Modifica el SL/TP de una posición abierta."""
        if not self._connected:
            return TradeResult(success=False, error_msg="No conectado")

        req = ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = self._cfg.account_id
        req.positionId          = position_id
        if sl_price is not None:
            req.stopLoss   = int(sl_price * 100_000)
        if tp_price is not None:
            req.takeProfit = int(tp_price * 100_000)

        self._client.send(req).addErrback(self._on_error)
        return TradeResult(success=True)

    def get_open_positions(self) -> dict:
        """Devuelve las posiciones abiertas (requiere ReconcileReq)."""
        if not self._connected:
            return {}
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self._cfg.account_id
        self._client.send(req).addErrback(self._on_error)
        return self._positions

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def symbol_id(self) -> Optional[int]:
        return self._symbol_id
