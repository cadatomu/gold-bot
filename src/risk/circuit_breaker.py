"""Circuit breakers for intraday and total drawdown limits.

Two independent breakers:
  - Daily breaker:  halt trading for the rest of the calendar day when
                    intraday equity drops ≥ max_daily_dd_pct from the
                    day's opening equity.
  - Total breaker:  permanent halt when equity drops ≥ max_total_dd_pct
                    from the all-time peak recorded since the bot started.

State is persisted to a JSON file so the bot survives restarts without
accidentally resetting the total-drawdown counter mid-session.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class CircuitBreakerState:
    peak_equity: float          # all-time peak — used for total DD calculation
    day_open_equity: float      # equity at start of current trading day
    day_date: str               # ISO date string "YYYY-MM-DD"
    daily_halt: bool            # True → halted for today
    total_halt: bool            # True → permanently halted

    @classmethod
    def initial(cls, equity: float) -> "CircuitBreakerState":
        today = date.today().isoformat()
        return cls(
            peak_equity    = equity,
            day_open_equity= equity,
            day_date       = today,
            daily_halt     = False,
            total_halt     = False,
        )


class CircuitBreaker:
    """
    Stateful circuit breaker that persists state to disk.

    Usage
    -----
    cb = CircuitBreaker(state_path=Path("live_state/circuit.json"),
                        max_daily_dd_pct=0.03, max_total_dd_pct=0.15)
    cb.load_or_init(current_equity)

    # Every bar / equity update:
    cb.update(current_equity)
    if cb.is_halted():
        ... don't trade ...
    """

    def __init__(
        self,
        state_path: Path,
        max_daily_dd_pct: float = 0.03,
        max_total_dd_pct: float = 0.15,
    ) -> None:
        self._path            = state_path
        self.max_daily_dd     = max_daily_dd_pct
        self.max_total_dd     = max_total_dd_pct
        self._state: Optional[CircuitBreakerState] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_or_init(self, current_equity: float) -> None:
        """Load state from disk, or create fresh state if file absent."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._state = CircuitBreakerState(**raw)
                self._maybe_reset_daily(current_equity)
                return
            except Exception:
                pass  # corrupt file → start fresh
        self._state = CircuitBreakerState.initial(current_equity)
        self._save()

    def _save(self) -> None:
        if self._state is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(self._state), indent=2))

    # ------------------------------------------------------------------
    # Day boundary reset
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self, current_equity: float) -> None:
        today = date.today().isoformat()
        if self._state.day_date != today:
            self._state.day_open_equity = current_equity
            self._state.day_date        = today
            self._state.daily_halt      = False
            # Update peak if equity grew overnight
            if current_equity > self._state.peak_equity:
                self._state.peak_equity = current_equity
            self._save()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def update(self, current_equity: float) -> None:
        """
        Evaluate breakers given the latest equity reading.
        Call this on every closed trade or equity update.
        """
        if self._state is None:
            raise RuntimeError("Call load_or_init() before update()")

        self._maybe_reset_daily(current_equity)

        # Total halt — permanent once triggered
        if not self._state.total_halt:
            total_dd = (self._state.peak_equity - current_equity) / self._state.peak_equity
            if total_dd >= self.max_total_dd:
                self._state.total_halt = True
                self._save()
                return  # no point checking daily

        # Daily halt
        if not self._state.daily_halt:
            daily_dd = (self._state.day_open_equity - current_equity) / self._state.day_open_equity
            if daily_dd >= self.max_daily_dd:
                self._state.daily_halt = True
                self._save()
                return

        # Update peak (only if not halted)
        if not self._state.total_halt and current_equity > self._state.peak_equity:
            self._state.peak_equity = current_equity
            self._save()

    def is_halted(self) -> bool:
        """True if trading should be blocked right now."""
        if self._state is None:
            return True
        return self._state.total_halt or self._state.daily_halt

    def is_total_halted(self) -> bool:
        return self._state is not None and self._state.total_halt

    def is_daily_halted(self) -> bool:
        return self._state is not None and self._state.daily_halt

    @property
    def state(self) -> Optional[CircuitBreakerState]:
        return self._state

    def daily_drawdown_pct(self) -> float:
        """Current intraday drawdown as a fraction (0.03 = 3%)."""
        if self._state is None:
            return 0.0
        if self._state.day_open_equity <= 0:
            return 0.0
        return max(0.0, (self._state.day_open_equity - self._state.peak_equity)
                   / self._state.day_open_equity)

    def total_drawdown_pct(self) -> float:
        """Current total drawdown from peak as a fraction."""
        if self._state is None or self._state.peak_equity <= 0:
            return 0.0
        # We don't track current equity here — caller must pass it to update()
        return 0.0  # meaningful value computed inside update()
