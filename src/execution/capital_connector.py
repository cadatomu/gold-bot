"""Conector para Capital.com REST API.

Credenciales requeridas en .env:
  CAPITAL_API_KEY   — Clave API generada en la plataforma
  CAPITAL_PASSWORD  — Password de la API (no el de la cuenta)
  CAPITAL_EMAIL     — Email de tu cuenta Capital.com
  CAPITAL_DEMO      — "true" para demo, "false" para live
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests


DEMO_URL = "https://demo-api-capital.backend-capital.com/api/v1"
LIVE_URL = "https://api-capital.backend-capital.com/api/v1"

GOLD_EPIC = "GOLD"


class CapitalConnector:
    """
    Cliente REST para Capital.com.

    Uso:
        conn = CapitalConnector.from_env()
        conn.connect()
        epic = conn.find_epic("GOLD")
        conn.place_order(epic, "BUY", size=1, stop_level=3200.0, profit_level=3400.0)
        conn.disconnect()
    """

    def __init__(self, api_key: str, email: str, password: str, is_demo: bool = True) -> None:
        self._api_key  = api_key
        self._email    = email
        self._password = password
        self._base_url = DEMO_URL if is_demo else LIVE_URL
        self._cst:            Optional[str] = None
        self._security_token: Optional[str] = None
        self._account_id:     Optional[str] = None

    @classmethod
    def from_env(cls) -> "CapitalConnector":
        return cls(
            api_key  = os.environ["CAPITAL_API_KEY"],
            email    = os.environ["CAPITAL_EMAIL"],
            password = os.environ["CAPITAL_PASSWORD"],
            is_demo  = os.environ.get("CAPITAL_DEMO", "true").lower() == "true",
        )

    # ── Autenticación ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Crea sesión y obtiene tokens CST + X-SECURITY-TOKEN."""
        resp = requests.post(
            f"{self._base_url}/session",
            json={
                "identifier":        self._email,
                "password":          self._password,
                "encryptedPassword": False,
            },
            headers={
                "X-CAP-API-KEY": self._api_key,
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        self._cst            = resp.headers["CST"]
        self._security_token = resp.headers["X-SECURITY-TOKEN"]
        data = resp.json()
        self._account_id = data.get("currentAccountId")
        print(f"[Capital] Conectado — cuenta: {self._account_id}")

    def disconnect(self) -> None:
        if self._cst:
            try:
                requests.delete(f"{self._base_url}/session", headers=self._headers(), timeout=5)
            except Exception:
                pass
        self._cst = self._security_token = None

    def _headers(self) -> dict:
        return {
            "X-CAP-API-KEY":    self._api_key,
            "CST":              self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
            "Content-Type":     "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(f"{self._base_url}{path}", headers=self._headers(),
                            params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(f"{self._base_url}{path}", json=body,
                             headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = requests.delete(f"{self._base_url}{path}",
                               headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Cuenta ────────────────────────────────────────────────────────────────

    def get_accounts(self) -> list:
        return self._get("/accounts").get("accounts", [])

    def get_account_balance(self) -> dict:
        for acc in self.get_accounts():
            if acc["accountId"] == self._account_id:
                return acc.get("balance", {})
        return {}

    # ── Mercados ──────────────────────────────────────────────────────────────

    def find_epic(self, search_term: str) -> Optional[str]:
        """Busca el epic de un instrumento (ej. 'GOLD' → devuelve el epic)."""
        data = self._get("/markets", params={"searchTerm": search_term})
        markets = data.get("markets", [])
        if not markets:
            return None
        for m in markets:
            name = m.get("instrumentName", "").upper()
            if search_term.upper() in name:
                return m["epic"]
        return markets[0]["epic"]

    def get_price(self, epic: str) -> dict:
        """Precio bid/ask actual de un instrumento."""
        return self._get(f"/markets/{epic}")

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        epic:         str,
        direction:    str,           # "BUY" o "SELL"
        size:         float,
        stop_level:   Optional[float] = None,
        profit_level: Optional[float] = None,
    ) -> dict:
        """
        Abre una posición de mercado.

        stop_level   : precio absoluto del Stop Loss
        profit_level : precio absoluto del Take Profit
        size         : tamaño en la unidad del instrumento (para GOLD: oz)
        """
        body: dict = {
            "epic":            epic,
            "direction":       direction.upper(),
            "size":            size,
            "guaranteedStop":  False,
            "trailingStop":    False,
        }
        if stop_level is not None:
            body["stopLevel"] = stop_level
        if profit_level is not None:
            body["profitLevel"] = profit_level

        result = self._post("/positions", body)
        deal_ref = result.get("dealReference", "")
        print(f"[Capital] Orden enviada — dealRef: {deal_ref}")
        return result

    def close_position(self, deal_id: str) -> dict:
        """Cierra una posición por su dealId."""
        result = self._delete(f"/positions/{deal_id}")
        print(f"[Capital] Posición cerrada — dealId: {deal_id}")
        return result

    def get_positions(self) -> list:
        """Lista de posiciones abiertas."""
        return self._get("/positions").get("positions", [])

    def modify_position(
        self,
        deal_id:      str,
        stop_level:   Optional[float] = None,
        profit_level: Optional[float] = None,
    ) -> dict:
        """Modifica SL/TP de una posición abierta."""
        body: dict = {}
        if stop_level is not None:
            body["stopLevel"] = stop_level
        if profit_level is not None:
            body["profitLevel"] = profit_level
        resp = requests.put(
            f"{self._base_url}/positions/{deal_id}",
            json=body, headers=self._headers(), timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._cst is not None

    @property
    def account_id(self) -> Optional[str]:
        return self._account_id
