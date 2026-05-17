"""Position sizing for XAUUSD trades.

Calculates lot size so that a fixed percentage of account equity is at risk
if the stop-loss is hit. Enforces ESMA leverage limits and per-trade caps.

XAUUSD contract spec (IC Markets Raw):
  - 1 standard lot = 100 troy oz
  - pip = $0.01, pip value per lot = $1.00
  - With ESMA max 1:20 leverage, margin per lot ≈ price × 100 / 20
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingConfig:
    risk_per_trade_pct: float = 0.005   # 0.5% equity risk per trade
    max_risk_per_trade_pct: float = 0.01  # hard cap: never risk > 1%
    max_leverage: float = 20.0            # ESMA cap for gold
    min_lot: float = 0.01
    max_lot: float = 50.0
    lot_step: float = 0.01
    point_size: float = 0.01             # XAUUSD 1 point = $0.01
    pip_value_per_lot: float = 1.0       # $1 per pip per lot at 1 point/pip
    contract_size: float = 100.0         # oz per standard lot


def calc_lot_size(
    equity: float,
    sl_distance: float,      # distance in price points (entry - SL for long)
    entry_price: float,
    cfg: SizingConfig | None = None,
) -> float:
    """
    Calculate lot size respecting risk % and ESMA leverage.

    Parameters
    ----------
    equity       : Current account equity in USD
    sl_distance  : |entry_price - sl_price| in price units (e.g. 2.50 for $2.50 SL)
    entry_price  : Trade entry price (used for margin leverage check)
    cfg          : SizingConfig (defaults if None)

    Returns
    -------
    Lot size rounded to nearest lot_step, clipped to [min_lot, max_lot].
    Returns min_lot if inputs are degenerate.
    """
    if cfg is None:
        cfg = SizingConfig()

    if equity <= 0 or sl_distance <= 0 or entry_price <= 0:
        return cfg.min_lot

    # Risk in USD — use the smaller of configured and hard-cap risk
    risk_pct = min(cfg.risk_per_trade_pct, cfg.max_risk_per_trade_pct)
    risk_usd = equity * risk_pct

    # Loss per lot if SL is hit
    sl_pips        = sl_distance / cfg.point_size
    loss_per_lot   = sl_pips * cfg.pip_value_per_lot
    if loss_per_lot <= 0:
        return cfg.min_lot

    lots_by_risk = risk_usd / loss_per_lot

    # ESMA margin check: notional = lots × contract_size × entry_price
    # max_notional = equity × max_leverage
    max_notional = equity * cfg.max_leverage
    max_lots_esma = max_notional / (cfg.contract_size * entry_price)

    lots = min(lots_by_risk, max_lots_esma, cfg.max_lot)
    lots = max(lots, cfg.min_lot)

    # Round to nearest lot_step
    lots = round(round(lots / cfg.lot_step) * cfg.lot_step, 2)
    return lots


def effective_leverage(
    lots: float,
    entry_price: float,
    equity: float,
    cfg: SizingConfig | None = None,
) -> float:
    """Return actual leverage used by a given position."""
    if cfg is None:
        cfg = SizingConfig()
    if equity <= 0:
        return 0.0
    notional = lots * cfg.contract_size * entry_price
    return notional / equity
