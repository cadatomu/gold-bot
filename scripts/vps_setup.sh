#!/usr/bin/env bash
# Setup completo de GoldBot en VPS limpio (Ubuntu 24.04)
# Pegar completo en el terminal web de Hostinger
set -e

echo "==> [1/5] Instalando dependencias..."
apt-get update -qq
apt-get install -y -qq git curl netcat-openbsd

echo "==> [2/5] Instalando Docker si no está..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

echo "==> [3/5] Clonando repositorio..."
mkdir -p /opt/gold-bot && cd /opt/gold-bot
if [ -d ".git" ]; then
    git pull
else
    git clone https://github.com/cadatomu/gold-bot.git .
fi

echo "==> [4/5] Creando .env..."
cat > /opt/gold-bot/.env << 'ENVEOF'
# ── Interactive Brokers ───────────────────────────────────────────────────────
IB_HOST=ib-gateway
IB_PORT=4002
IB_CLIENT_ID=1
IB_CLIENT_ID_BOT=5
IB_ACCOUNT=DUQ057139
IB_USERNAME=cadatomu28
IB_PASSWORD=Minerva_2892

# ── IB Gateway Docker ─────────────────────────────────────────────────────────
TRADING_MODE=paper
VNC_PASSWORD=GoldVNC2024

# ── GoldBot ──────────────────────────────────────────────────────────────────
PAPER_MODE=true
RISK_PCT=0.08
INITIAL_EQUITY=10000
CHECK_INTERVAL_MIN=15
STATE_FILE=/app/data/bot_state.json

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN=8959952042:AAEmB39mHfeRRFh_34SLpdp4ILJ1Yf_2pcs
TELEGRAM_CHAT_ID=8325545615
ENVEOF

chmod 600 /opt/gold-bot/.env
echo "  .env creado"

echo "==> [5/5] Levantando servicios..."
cd /opt/gold-bot
docker compose pull ib-gateway
docker compose build goldbot
docker compose up -d

echo ""
echo "========================================"
echo "  Deploy completado!"
echo "  IB Gateway arranca en ~90 segundos"
echo ""
echo "  Ver logs:"
echo "  docker logs -f ib-gateway"
echo "  docker logs -f goldbot"
echo ""
echo "  Estado:"
docker compose ps
echo "========================================"
