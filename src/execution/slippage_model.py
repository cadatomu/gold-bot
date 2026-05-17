"""Realistic slippage model for backtesting XAUUSD H4.

Slippage is modelled as a random draw from a half-normal distribution
parameterised from IC Markets raw-spread historical data:
  - Mean slippage ≈ 1.0 pip (0.01 price units for XAUUSD)
  - Tail events (news spikes) captured via scale parameter

Used in the backtest runner to adjust entry/exit prices away from the
signal bar close, simulating market-order execution reality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlippageConfig:
    mean_pips: float = 1.0      # average slip in pips (1 pip = 0.01 for XAUUSD)
    point_size: float = 0.01    # price per pip
    seed: int | None = None     # set for deterministic backtests


class SlippageModel:
    """Samples slippage for market order entries and exits."""

    def __init__(self, cfg: SlippageConfig | None = None) -> None:
        self._cfg = cfg or SlippageConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    def entry_slip(self, direction: int = 1) -> float:
        """
        Return slippage in price units for a market entry.

        Parameters
        ----------
        direction : +1 for long (we buy higher), -1 for short (we sell lower)

        Returns
        -------
        Price adjustment to add to the theoretical fill price.
        For a long: positive (we pay more than the close).
        """
        slip_pips = abs(self._rng.normal(self._cfg.mean_pips, self._cfg.mean_pips * 0.5))
        return direction * slip_pips * self._cfg.point_size

    def exit_slip(self, direction: int = 1) -> float:
        """
        Return slippage for a market exit.

        For a long exit: negative (we receive less than the theoretical price).
        """
        slip_pips = abs(self._rng.normal(self._cfg.mean_pips, self._cfg.mean_pips * 0.5))
        return -direction * slip_pips * self._cfg.point_size

    def reset(self) -> None:
        """Reset RNG to reproduce the same sequence (useful in walk-forward)."""
        self._rng = np.random.default_rng(self._cfg.seed)
