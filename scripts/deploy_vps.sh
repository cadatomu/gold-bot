#!/usr/bin/env bash
# Despliega GoldBot + IB Gateway al VPS Hostinger.
# Uso:
#   ./scripts/deploy_vps.sh            # deploy normal
#   ./scripts/deploy_vps.sh --rebuild  # fuerza rebuild de imágenes
set -euo pipefail

VPS_HOST="187.124.133.92"
VPS_USER="root"
REMOTE_DIR="/opt/gold-bot"
REBUILD="${1:-}"

echo "==> [1/4] Sincronizando código al VPS..."
rsync -avz --exclude='.git' --exclude='data/' --exclude='__pycache__' \
      --exclude='*.pkl' --exclude='*.log' --exclude='.env' \
      ./ "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/"

echo "==> [2/4] Copiando .env al VPS..."
scp .env "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/.env"

echo "==> [3/4] Desplegando en VPS..."
ssh "${VPS_USER}@${VPS_HOST}" bash << ENDSSH
  set -e
  cd ${REMOTE_DIR}
  mkdir -p data

  # Instalar netcat si no está (para healthcheck de ib-gateway)
  apt-get install -y netcat-openbsd 2>/dev/null || true

  if [ "${REBUILD}" = "--rebuild" ]; then
    echo "  Rebuild completo..."
    docker compose down --remove-orphans
    docker compose pull ib-gateway
    docker compose build --no-cache goldbot
  fi

  docker compose up -d
  echo ""
  echo "  Esperando que IB Gateway arranque (~90s)..."
  sleep 10
  docker compose ps
ENDSSH

echo ""
echo "==> [4/4] Deploy completado."
echo ""
echo "  Logs IB Gateway : ssh ${VPS_USER}@${VPS_HOST} 'docker logs -f ib-gateway'"
echo "  Logs GoldBot    : ssh ${VPS_USER}@${VPS_HOST} 'docker logs -f goldbot'"
echo "  VNC debug       : vnc://${VPS_HOST}:5900  (pass: GoldVNC2024)"
echo ""
echo "  Estado: ssh ${VPS_USER}@${VPS_HOST} 'docker compose -f ${REMOTE_DIR}/docker-compose.yml ps'"
