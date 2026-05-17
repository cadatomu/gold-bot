"""Notificaciones y comandos Telegram via HTTP polling."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 10


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self._token   = token
        self._chat_id = str(chat_id)
        self._base    = f"https://api.telegram.org/bot{token}"
        self._offset  = 0  # para getUpdates long-polling

    def _enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    # ── Envío ──────────────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        if not self._enabled():
            log.debug(f"[TG sin configurar] {text}")
            return
        try:
            r = requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
        except Exception as e:
            log.error(f"Telegram send error: {e}")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def send_start(self, paper: bool) -> None:
        modo = "PAPER" if paper else "⚠️ LIVE"
        self.send(
            f"🚀 <b>GoldBot arrancado [{modo}]</b>\n"
            f"Instrumento : MGC (Micro Gold 10oz)\n"
            f"Estrategia  : AdaptiveScalp H4\n"
            f"Risk/trade  : 8%\n"
            f"🕐 {self._ts()}"
        )

    def send_entry(self, direction: str, price: float, sl: float, tp: float,
                   lots: int, equity: float, atr: float) -> None:
        emoji    = "🟢" if direction == "LONG" else "🔴"
        sl_dist  = abs(price - sl)
        risk_usd = sl_dist / 0.10 * 1.0 * lots
        self.send(
            f"{emoji} <b>ENTRADA {direction} — MGC</b>\n"
            f"Precio  : <b>${price:,.2f}</b>\n"
            f"SL      : ${sl:,.2f}  ({sl_dist:.1f}/oz)\n"
            f"TP      : ${tp:,.2f}\n"
            f"Lotes   : {lots} MGC ({lots*10} oz)\n"
            f"Riesgo  : ${risk_usd:.0f} ({risk_usd/equity*100:.1f}%)\n"
            f"ATR     : {atr:.2f}\n"
            f"Capital : ${equity:,.0f}\n"
            f"🕐 {self._ts()}"
        )

    def send_exit(self, reason: str, price: float, pnl: float,
                  equity: float, lots: int) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(
            f"{emoji} <b>SALIDA — {reason}</b>\n"
            f"Precio  : ${price:,.2f}\n"
            f"PnL     : <b>${pnl:+,.2f}</b>\n"
            f"Lotes   : {lots} MGC\n"
            f"Capital : ${equity:,.0f}\n"
            f"🕐 {self._ts()}"
        )

    def send_partial(self, price: float, pnl: float, remaining: int) -> None:
        self.send(
            f"⚡ <b>CIERRE PARCIAL 50%</b>\n"
            f"Precio     : ${price:,.2f}\n"
            f"PnL parcial: ${pnl:+,.2f}\n"
            f"Quedan     : {remaining} MGC\n"
            f"🕐 {self._ts()}"
        )

    def send_trail(self, new_sl: float, current_price: float) -> None:
        self.send(
            f"📍 <b>Trail SL actualizado</b>\n"
            f"Precio actual: ${current_price:,.2f}\n"
            f"Nuevo SL     : ${new_sl:,.2f}\n"
            f"🕐 {self._ts()}"
        )

    def send_status(self, pos, equity: float, last_signal: str) -> None:
        if not pos.is_open:
            pos_str = "Sin posición (FLAT)"
        else:
            emoji   = "🟢" if pos.direction == "LONG" else "🔴"
            pos_str = (
                f"{emoji} {pos.direction} | {pos.lots} MGC\n"
                f"  Entrada : ${pos.entry_price:,.2f}\n"
                f"  SL      : ${pos.sl_price:,.2f}\n"
                f"  TP      : ${pos.tp_price:,.2f}\n"
                f"  Parcial : {'✅' if pos.partial_done else 'pendiente'}"
            )
        self.send(
            f"📊 <b>Status GoldBot</b>\n"
            f"Capital       : ${equity:,.0f}\n"
            f"PnL hoy       : ${pos.day_pnl:+,.2f} ({pos.day_trades} trades)\n"
            f"PnL total     : ${pos.total_pnl:+,.2f} ({pos.total_trades} trades)\n"
            f"Posición      : {pos_str}\n"
            f"Última señal  : {last_signal}\n"
            f"🕐 {self._ts()}"
        )

    def send_daily(self, equity: float, day_pnl: float, day_trades: int) -> None:
        emoji = "📈" if day_pnl >= 0 else "📉"
        self.send(
            f"{emoji} <b>Resumen diario — GoldBot</b>\n"
            f"Capital : ${equity:,.0f}\n"
            f"PnL hoy : ${day_pnl:+,.2f}\n"
            f"Trades  : {day_trades}\n"
            f"🕐 {self._ts()}"
        )

    def send_error(self, msg: str) -> None:
        self.send(f"⚠️ <b>ERROR</b>\n{msg[:400]}")

    # ── Comandos (polling) ────────────────────────────────────────────────────

    def poll_commands(self, handlers: dict[str, Callable]) -> None:
        """
        Procesa comandos pendientes de Telegram.
        handlers: {'/status': fn, '/stop': fn, ...}
        Llamar periódicamente en el loop principal.
        """
        if not self._enabled():
            return
        try:
            r = requests.get(
                f"{self._base}/getUpdates",
                params={"offset": self._offset, "timeout": 1},
                timeout=5,
            )
            data = r.json()
            for update in data.get("result", []):
                self._offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat = str(msg.get("chat", {}).get("id", ""))
                if chat != self._chat_id:
                    continue
                for cmd, fn in handlers.items():
                    if text.startswith(cmd.lower()):
                        fn()
                        break
        except Exception as e:
            log.debug(f"Telegram poll error: {e}")
