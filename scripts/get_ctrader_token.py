"""Obtiene el Access Token de cTrader via OAuth2.

Pasos:
  1. Ejecuta este script: python scripts/get_ctrader_token.py
  2. Abre la URL que imprime en tu navegador
  3. Inicia sesión con tu cuenta IC Markets (cadatomu)
  4. Copia el 'code' de la URL de redirección
  5. Pégalo aquí → el script guarda el token en .env

Requiere en .env (o como variables de entorno):
  CTRADER_CLIENT_ID
  CTRADER_CLIENT_SECRET
"""

import os
import sys
import urllib.parse
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

AUTH_URL  = "https://connect.spotware.com/apps/auth"
TOKEN_URL = "https://connect.spotware.com/apps/token"
REDIRECT  = "https://localhost"


def load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def save_to_env(key: str, value: str):
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    print(f"  Guardado {key} en .env")


def main():
    load_env()

    client_id     = os.environ.get("CTRADER_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        print("Necesitas CTRADER_CLIENT_ID y CTRADER_CLIENT_SECRET en .env")
        print()
        print("Pasos para obtenerlos:")
        print("  1. Inicia sesión en el portal cTrader Open API")
        print("  2. Crea una nueva aplicación")
        print("  3. Copia Client ID y Client Secret")
        print("  4. Agrega al .env:")
        print("     CTRADER_CLIENT_ID=tu_client_id")
        print("     CTRADER_CLIENT_SECRET=tu_client_secret")
        sys.exit(1)

    # Paso 1: URL de autorización
    params = urllib.parse.urlencode({
        "client_id":     client_id,
        "redirect_uri":  REDIRECT,
        "response_type": "code",
        "scope":         "trading",
    })
    auth_url = f"{AUTH_URL}?{params}"

    print("=" * 60)
    print("  PASO 1: Abre esta URL en tu navegador:")
    print("=" * 60)
    print(f"\n  {auth_url}\n")
    print("Inicia sesión con tu cuenta IC Markets (cadatomu@gmail.com)")
    print("Después de autorizar, serás redirigido a una URL tipo:")
    print("  https://localhost/?code=XXXXXXXX")
    print()

    # Paso 2: Pegar el code
    code = input("Pega el 'code' de la URL de redirección: ").strip()
    if not code:
        print("No se proporcionó código.")
        sys.exit(1)

    # Paso 3: Intercambiar code por token
    data = urllib.parse.urlencode({
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT,
        "client_id":     client_id,
        "client_secret": client_secret,
    }).encode()

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"Error HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

    access_token  = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")

    if not access_token:
        print(f"Error: {result}")
        sys.exit(1)

    print("\nToken obtenido exitosamente.")
    save_to_env("CTRADER_ACCESS_TOKEN",  access_token)
    save_to_env("CTRADER_REFRESH_TOKEN", refresh_token)
    save_to_env("CTRADER_ACCOUNT_ID",    "10027534")
    save_to_env("CTRADER_DEMO",          "true")

    print()
    print("Listo. Ahora puedes correr el bot con:")
    print("  python scripts/run_live.py")


if __name__ == "__main__":
    main()
