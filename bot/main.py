"""GoldBot — paper trading H4 AdaptiveScalp en MGC (Micro Gold IB).

Loop principal:
  - Cada CHECK_INTERVAL_MIN minutos: revisa señal H4 y gestiona posición abierta
  - Si hay nueva barra H4 cerrada: evalúa entrada
  - Si posición abierta: revisa trail, cierre parcial, SL/TP llenados
  - Comandos Telegram: /status /pnl /pause /resume /stop

Uso:
  python -m bot.main
  python -m bot.main --once      # un solo ciclo y sale (test/debug)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config import load_config
from bot.ib_trader import IBTrader
from bot.signal_gen import get_signal
from bot.state import Position, StateManager
from bot.telegram_notif import Telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data" / "goldbot.log"),
    ],
)
log = logging.getLogger("goldbot.main")

_RUNNING = True
_PAUSED  = False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _manage_open_position(trader: IBTrader, pos: Position,
                           state: StateManager, tg: Telegram) -> Position:
    """
    Gestiona trail, cierre parcial y detección de SL/TP llenado
    para una posición abierta. Actualiza y retorna el estado.
    """
    current_price = trader.get_current_price()
    if current_price <= 0:
        log.warning("No se pudo obtener precio actual")
        return pos

    direction = pos.direction

    # ── Detectar si SL/TP fue ejecutado por IB ─────────────────────────────
    sl_filled = pos.ib_sl_order_id and not trader.order_is_open(pos.ib_sl_order_id)
    tp_filled = pos.ib_tp_order_id and not trader.order_is_open(pos.ib_tp_order_id)

    # También verificar si IB ya no reporta posición
    if not trader.has_position() and pos.is_open:
        # Estimar qué ocurrió por precio
        if direction == "LONG":
            reason = "TP" if current_price >= pos.tp_price * 0.999 else "SL"
        else:
            reason = "TP" if current_price <= pos.tp_price * 1.001 else "SL"
        exit_price = pos.tp_price if reason == "TP" else pos.sl_price
        pnl = _calc_pnl(direction, pos.entry_price, exit_price, pos.lots)
        _record_exit(pos, state, tg, reason, exit_price, pnl, trader.get_equity())
        return pos

    if tp_filled:
        pnl = _calc_pnl(direction, pos.entry_price, pos.tp_price, pos.lots)
        trader.cancel_order(pos.ib_sl_order_id)
        _record_exit(pos, state, tg, "TP", pos.tp_price, pnl, trader.get_equity())
        return pos

    if sl_filled:
        pnl = _calc_pnl(direction, pos.entry_price, pos.sl_price, pos.lots)
        trader.cancel_order(pos.ib_tp_order_id)
        _record_exit(pos, state, tg, "SL", pos.sl_price, pnl, trader.get_equity())
        return pos

    # ── Cierre parcial ──────────────────────────────────────────────────────
    if not pos.partial_done and pos.lots >= 2:
        hit = (direction == "LONG"  and current_price >= pos.partial_trigger) or \
              (direction == "SHORT" and current_price <= pos.partial_trigger)
        if hit:
            close_lots = max(1, pos.lots // 2)
            pnl = _calc_pnl(direction, pos.entry_price, current_price, close_lots)
            trader.close_partial(direction, pos.lots)
            pos.lots -= close_lots
            pos.partial_done = True
            tg.send_partial(current_price, pnl, pos.lots)
            log.info(f"Cierre parcial: {close_lots} MGC @ {current_price:.2f}  PnL=${pnl:.2f}")
            state.save(pos)

    # ── Trailing stop ────────────────────────────────────────────────────────
    trail_hit = (direction == "LONG"  and current_price >= pos.trail_trigger) or \
                (direction == "SHORT" and current_price <= pos.trail_trigger)
    if trail_hit:
        if direction == "LONG":
            new_sl = current_price - pos.trail_distance
            if new_sl > pos.sl_price:
                trader.move_stop(pos.ib_sl_order_id, new_sl)
                tg.send_trail(new_sl, current_price)
                pos.sl_price     = new_sl
                pos.trail_trigger = current_price + pos.trail_distance * 0.5
                state.save(pos)
        else:
            new_sl = current_price + pos.trail_distance
            if new_sl < pos.sl_price:
                trader.move_stop(pos.ib_sl_order_id, new_sl)
                tg.send_trail(new_sl, current_price)
                pos.sl_price      = new_sl
                pos.trail_trigger = current_price - pos.trail_distance * 0.5
                state.save(pos)

    return pos


def _calc_pnl(direction: str, entry: float, exit_p: float, lots: int) -> float:
    """PnL en USD para MGC (10oz, $1/tick, tick=$0.10)."""
    diff = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
    ticks = diff / 0.10
    return ticks * 1.0 * lots


def _record_exit(pos: Position, state: StateManager, tg: Telegram,
                 reason: str, price: float, pnl: float, equity: float) -> None:
    log.info(f"SALIDA {reason} @ {price:.2f}  PnL=${pnl:.2f}")
    pos.day_pnl    += pnl
    pos.day_trades += 1
    pos.total_pnl  += pnl
    pos.total_trades += 1
    tg.send_exit(reason, price, pnl, equity, pos.lots)
    state.reset_position(pos)


def _process_signal(sig: dict, trader: IBTrader, pos: Position,
                    state: StateManager, tg: Telegram) -> Position:
    """Abre nueva posición si hay señal y no hay posición abierta."""
    if sig["direction"] == "FLAT" or pos.is_open:
        return pos

    equity   = trader.get_equity()
    sl_dist  = abs(sig["close"] - sig["sl_price"])
    lots     = trader.calc_lots(equity, sl_dist)

    sl_id, tp_id = trader.place_bracket(
        direction = sig["direction"],
        lots      = lots,
        sl_price  = sig["sl_price"],
        tp_price  = sig["tp_price"],
    )

    pos.direction       = sig["direction"]
    pos.lots            = lots
    pos.entry_price     = sig["close"]
    pos.sl_price        = sig["sl_price"]
    pos.tp_price        = sig["tp_price"]
    pos.trail_trigger   = sig["trail_trigger"]
    pos.trail_distance  = sig["trail_distance"]
    pos.partial_trigger = sig["partial_trigger"]
    pos.partial_done    = False
    pos.entry_time      = str(_now_utc())
    pos.ib_sl_order_id  = sl_id
    pos.ib_tp_order_id  = tp_id
    state.save(pos)

    log.info(f"ENTRADA {pos.direction} {lots}×MGC @ {pos.entry_price:.2f} "
             f"SL={pos.sl_price:.2f} TP={pos.tp_price:.2f}")
    tg.send_entry(pos.direction, pos.entry_price, pos.sl_price,
                  pos.tp_price, lots, equity, sig["atr"])
    return pos


def run_cycle(cfg, state: StateManager, tg: Telegram, last_signal: list) -> None:
    """Un ciclo completo: descarga datos, gestiona posición, evalúa señal."""
    trader = IBTrader(cfg)
    try:
        trader.connect()
        pos = state.load()

        # ── Descargar datos ────────────────────────────────────────────────
        h4_df    = trader.download_h4()
        daily_df = trader.download_daily()

        if h4_df.empty or daily_df.empty:
            tg.send_error("Sin datos de IB — verifica que Gateway esté activo")
            return

        # ── Gestionar posición abierta ─────────────────────────────────────
        if pos.is_open:
            pos = _manage_open_position(trader, pos, state, tg)

        # ── Nueva señal (solo en barra H4 nueva) ───────────────────────────
        bar_time = str(h4_df.index[-1])
        if bar_time != pos.last_h4_bar:
            sig = get_signal(h4_df, daily_df)
            last_signal[0] = f"{sig['direction']} @ {sig.get('close', 0):.2f} [{bar_time[:16]}]"
            pos.last_h4_bar = bar_time
            state.save(pos)

            if not pos.is_open:
                pos = _process_signal(sig, trader, pos, state, tg)
        else:
            log.info(f"Misma barra H4 ({bar_time[:16]}) — sin nueva señal")

    except Exception as e:
        log.error(f"Error en ciclo: {e}", exc_info=True)
        tg.send_error(str(e))
    finally:
        trader.disconnect()


def main() -> None:
    global _RUNNING, _PAUSED

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Un ciclo y sale")
    args = parser.parse_args()

    cfg       = load_config()
    state_mgr = StateManager(cfg.state_file)
    tg        = Telegram(cfg.telegram_token, cfg.telegram_chat_id)

    last_signal   = ["Sin señal aún"]
    daily_summary = [_now_utc().date()]

    def _handle_stop(sig, frame):
        global _RUNNING
        log.info("Señal SIGTERM recibida — deteniendo bot")
        tg.send("🛑 GoldBot detenido (SIGTERM)")
        _RUNNING = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT,  _handle_stop)

    # Comandos Telegram
    def cmd_status():
        pos    = state_mgr.load()
        equity = cfg.initial_equity  # aproximación sin IB abierto
        tg.send_status(pos, equity, last_signal[0])

    def cmd_pause():
        global _PAUSED
        _PAUSED = True
        tg.send("⏸ Bot pausado. /resume para continuar.")

    def cmd_resume():
        global _PAUSED
        _PAUSED = False
        tg.send("▶️ Bot reanudado.")

    def cmd_stop():
        global _RUNNING
        tg.send("🛑 Deteniendo bot...")
        _RUNNING = False

    handlers = {
        "/status": cmd_status,
        "/pause":  cmd_pause,
        "/resume": cmd_resume,
        "/stop":   cmd_stop,
    }

    tg.send_start(cfg.paper_mode)
    log.info(f"GoldBot iniciado | paper={cfg.paper_mode} | "
             f"IB {cfg.ib_host}:{cfg.ib_port} | interval={cfg.check_interval_min}min")

    if args.once:
        run_cycle(cfg, state_mgr, tg, last_signal)
        return

    interval_s = cfg.check_interval_min * 60
    next_run   = time.time()

    while _RUNNING:
        tg.poll_commands(handlers)

        now = time.time()
        if now >= next_run:
            next_run = now + interval_s
            if not _PAUSED:
                log.info(f"--- Ciclo [{_now_utc().strftime('%H:%M UTC')}] ---")
                run_cycle(cfg, state_mgr, tg, last_signal)
            else:
                log.info("Bot pausado — saltando ciclo")

        # Resumen diario a las 22:00 UTC
        today = _now_utc().date()
        if today != daily_summary[0] and _now_utc().hour == 22:
            pos    = state_mgr.load()
            equity = cfg.initial_equity
            tg.send_daily(equity, pos.day_pnl, pos.day_trades)
            pos.day_pnl    = 0.0
            pos.day_trades = 0
            state_mgr.save(pos)
            daily_summary[0] = today

        time.sleep(30)  # poll Telegram cada 30s, ciclo cada interval_s


if __name__ == "__main__":
    main()
