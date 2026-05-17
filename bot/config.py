"""Configuración del bot leída desde variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # IB Gateway
    ib_host: str
    ib_port: int
    ib_client_id: int
    ib_account: str

    # Telegram
    telegram_token: str
    telegram_chat_id: str

    # Risk
    risk_pct: float
    initial_equity: float

    # Bot
    paper_mode: bool
    state_file: str
    check_interval_min: int  # frecuencia del loop principal


def load_config() -> Config:
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    return Config(
        ib_host           = os.getenv("IB_HOST", "127.0.0.1"),
        ib_port           = int(os.getenv("IB_PORT", "7497")),
        ib_client_id      = int(os.getenv("IB_CLIENT_ID_BOT", "5")),
        ib_account        = os.getenv("IB_ACCOUNT", ""),
        telegram_token    = os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id  = os.getenv("TELEGRAM_CHAT_ID", ""),
        risk_pct          = float(os.getenv("RISK_PCT", "0.08")),
        initial_equity    = float(os.getenv("INITIAL_EQUITY", "10000")),
        paper_mode        = os.getenv("PAPER_MODE", "true").lower() == "true",
        state_file        = os.getenv("STATE_FILE", str(root / "data" / "bot_state.json")),
        check_interval_min = int(os.getenv("CHECK_INTERVAL_MIN", "15")),
    )
