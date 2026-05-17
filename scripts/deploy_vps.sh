#!/usr/bin/env bash
# Despliega GoldBot al VPS Hostinger.
# Uso: ./scripts/deploy_vps.sh [--rebuild]
set -euo pipefail

VPS_HOST="187.124.133.92"
VPS_USER="root"
REMOTE_DIR="/opt/gold-bot"
REBUILD="${1:-}"

echo "==> Sincronizando código al VPS..."
rsync -avz --exclude='.git' --exclude='data/' --exclude='__pycache__' \
      --exclude='*.pkl' --exclude='*.log' --exclude='.env' \
      ./ "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/"

echo "==> Copiando .env al VPS (si existe localmente)..."
if [ -f .env ]; then
    scp .env "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/.env"
fi

echo "==> Desplegando en VPS..."
ssh "${VPS_USER}@${VPS_HOST}" bash <<EOF
  cd ${REMOTE_DIR}
  mkdir -p data

  if [ "${REBUILD}" = "--rebuild" ]; then
    docker-compose down
    docker-compose build --no-cache
  fi

  docker-compose up -d
  echo "==> Estado del contenedor:"
  docker ps | grep goldbot || echo "(no corriendo)"
EOF

echo ""
echo "==> Deploy completado."
echo "    Logs: ssh ${VPS_USER}@${VPS_HOST} 'docker logs -f goldbot'"
