"""Persistencia del estado de la posición abierta en JSON."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Position:
    direction: str = "FLAT"       # LONG | SHORT | FLAT
    lots: int = 0                  # contratos MGC
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    trail_trigger: float = 0.0     # precio en que activa el trail
    trail_distance: float = 0.0    # distancia del trail en $/oz
    partial_trigger: float = 0.0   # precio para cierre parcial
    partial_done: bool = False
    entry_time: str = ""
    last_h4_bar: str = ""          # timestamp de la última barra H4 procesada
    ib_sl_order_id: int = 0
    ib_tp_order_id: int = 0
    day_pnl: float = 0.0
    day_trades: int = 0
    total_pnl: float = 0.0
    total_trades: int = 0

    @property
    def is_open(self) -> bool:
        return self.direction != "FLAT"


class StateManager:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Position:
        if not self._path.exists():
            return Position()
        try:
            data = json.loads(self._path.read_text())
            return Position(**{k: v for k, v in data.items() if k in Position.__dataclass_fields__})
        except Exception as e:
            log.warning(f"Error cargando estado: {e} — usando estado limpio")
            return Position()

    def save(self, pos: Position) -> None:
        self._path.write_text(json.dumps(asdict(pos), indent=2))

    def reset_position(self, pos: Position) -> Position:
        """Cierra la posición manteniendo contadores de P&L."""
        pos.direction = "FLAT"
        pos.lots = 0
        pos.entry_price = 0.0
        pos.sl_price = 0.0
        pos.tp_price = 0.0
        pos.trail_trigger = 0.0
        pos.trail_distance = 0.0
        pos.partial_trigger = 0.0
        pos.partial_done = False
        pos.entry_time = ""
        pos.ib_sl_order_id = 0
        pos.ib_tp_order_id = 0
        self.save(pos)
        return pos
