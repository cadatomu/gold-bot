"""Conexión a IB Gateway y ejecución de órdenes paper (MGC — Micro Gold 10oz)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

# MGC: Micro Gold Futures, 10 oz/contrato, COMEX
# tick = $0.10/oz → pip_value = $1.00/tick/contrato
_MGC_TICK   = 0.10    # $/oz
_MGC_OZ     = 10      # oz por contrato
_MGC_PIPVAL = 1.00    # $ por tick por contrato

# Meses activos GC/MGC: Feb(G) Abr(J) Jun(M) Ago(Q) Oct(V) Dic(Z)
_ROLL_MONTHS = [2, 4, 6, 8, 10, 12]


def _front_month() -> str:
    """YYYYMM del contrato MGC front-month activo."""
    now = datetime.now(timezone.utc)
    y, m, d = now.year, now.month, now.day
    for rm in _ROLL_MONTHS:
        # Rola ~5 días antes del vencimiento (último día hábil del mes)
        if rm > m or (rm == m and d < 20):
            return f"{y}{rm:02d}"
    return f"{y+1}{_ROLL_MONTHS[0]:02d}"


def _bars_to_df(bars) -> pd.DataFrame:
    rows = [{"time": b.date, "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars if b.open > 0]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _to_utc(x):
        ts = pd.Timestamp(x)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    df["time"] = df["time"].apply(_to_utc)
    return df.set_index("time").sort_index().pipe(lambda d: d[~d.index.duplicated(keep="last")])


class IBTrader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ib = None
        self._gc  = None   # GC CONTFUT — para datos históricos
        self._mgc = None   # MGC front-month — para órdenes

    # ── Conexión ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        from ib_insync import IB, Contract, util
        util.logToConsole(level=40)
        self._ib = IB()
        self._ib.connect(
            self.cfg.ib_host, self.cfg.ib_port,
            clientId=self.cfg.ib_client_id, readonly=False,
        )

        # Contrato de datos: GC CONTFUT (continuo)
        gc = Contract(symbol="GC", secType="CONTFUT", exchange="COMEX", currency="USD")
        self._ib.qualifyContracts(gc)
        self._gc = gc

        # Contrato de ejecución: MGC front-month
        fm = _front_month()
        mgc = Contract(symbol="MGC", secType="FUT", exchange="COMEX",
                       currency="USD", lastTradeDateOrContractMonth=fm)
        self._ib.qualifyContracts(mgc)
        self._mgc = mgc

        log.info(f"IB conectado {self.cfg.ib_host}:{self.cfg.ib_port} "
                 f"| datos=GC CONTFUT | ejecución=MGC {fm}")

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    # ── Cuenta ────────────────────────────────────────────────────────────────

    def get_equity(self) -> float:
        try:
            vals = self._ib.accountValues(self.cfg.ib_account)
            for v in vals:
                if v.tag == "NetLiquidation" and v.currency == "USD":
                    return float(v.value)
        except Exception as e:
            log.warning(f"No pudo leer equity de IB: {e}")
        return self.cfg.initial_equity

    # ── Datos históricos ─────────────────────────────────────────────────────

    def download_h4(self) -> pd.DataFrame:
        bars = self._ib.reqHistoricalData(
            self._gc, endDateTime="", durationStr="2 Y",
            barSizeSetting="4 hours", whatToShow="TRADES",
            useRTH=False, formatDate=2, keepUpToDate=False,
        )
        time.sleep(3)
        return _bars_to_df(bars) if bars else pd.DataFrame()

    def download_daily(self) -> pd.DataFrame:
        bars = self._ib.reqHistoricalData(
            self._gc, endDateTime="", durationStr="2 Y",
            barSizeSetting="1 day", whatToShow="TRADES",
            useRTH=False, formatDate=2, keepUpToDate=False,
        )
        time.sleep(3)
        return _bars_to_df(bars) if bars else pd.DataFrame()

    def get_current_price(self) -> float:
        ticker = self._ib.reqMktData(self._mgc, "", False, False)
        self._ib.sleep(3)
        price = ticker.last or ticker.close or 0.0
        self._ib.cancelMktData(self._mgc)
        return float(price)

    # ── Sizing ────────────────────────────────────────────────────────────────

    def calc_lots(self, equity: float, sl_dist_oz: float) -> int:
        """Contratos MGC para arriesgar risk_pct del equity."""
        risk_usd      = equity * self.cfg.risk_pct
        sl_ticks      = sl_dist_oz / _MGC_TICK
        loss_per_mgc  = sl_ticks * _MGC_PIPVAL
        if loss_per_mgc <= 0:
            return 1
        return max(1, round(risk_usd / loss_per_mgc))

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def place_bracket(self, direction: str, lots: int,
                      sl_price: float, tp_price: float) -> tuple[int, int]:
        """
        Market entry + Stop SL + Limit TP.
        Retorna (sl_order_id, tp_order_id).
        """
        from ib_insync import MarketOrder, StopOrder, LimitOrder

        action    = "BUY"  if direction == "LONG"  else "SELL"
        sl_action = "SELL" if direction == "LONG"  else "BUY"
        tp_action = "SELL" if direction == "LONG"  else "BUY"

        entry_trade = self._ib.placeOrder(self._mgc, MarketOrder(action, lots, outsideRth=True))
        self._ib.sleep(1)

        sl_trade = self._ib.placeOrder(
            self._mgc, StopOrder(sl_action, lots, sl_price, outsideRth=True))
        tp_trade = self._ib.placeOrder(
            self._mgc, LimitOrder(tp_action, lots, tp_price, outsideRth=True))
        self._ib.sleep(1)

        log.info(f"Bracket {direction} {lots}×MGC | SL={sl_price:.2f} TP={tp_price:.2f}")
        return sl_trade.order.orderId, tp_trade.order.orderId

    def close_market(self, direction: str, lots: int) -> None:
        from ib_insync import MarketOrder
        action = "SELL" if direction == "LONG" else "BUY"
        self._ib.placeOrder(self._mgc, MarketOrder(action, lots, outsideRth=True))
        self._ib.sleep(1)
        log.info(f"Cierre market {direction} {lots}×MGC")

    def close_partial(self, direction: str, lots: int) -> None:
        """Cierra la mitad de la posición."""
        close_lots = max(1, lots // 2)
        self.close_market(direction, close_lots)

    def cancel_order(self, order_id: int) -> None:
        for o in self._ib.orders():
            if o.orderId == order_id:
                self._ib.cancelOrder(o)
                self._ib.sleep(0.5)
                return

    def move_stop(self, order_id: int, new_sl: float) -> None:
        """Modifica el precio del stop existente."""
        for o in self._ib.orders():
            if o.orderId == order_id:
                o.auxPrice = new_sl
                self._ib.placeOrder(self._mgc, o)
                self._ib.sleep(0.5)
                log.info(f"SL movido a {new_sl:.2f}")
                return

    def order_is_open(self, order_id: int) -> bool:
        """True si la orden sigue activa (no llenada ni cancelada)."""
        return any(o.orderId == order_id for o in self._ib.orders())

    def has_position(self) -> bool:
        """True si IB reporta posición abierta en MGC."""
        for p in self._ib.positions(self.cfg.ib_account):
            if p.contract.symbol == "MGC" and p.position != 0:
                return True
        return False
