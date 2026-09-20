#!/usr/bin/env bash

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME=$(grep -oP '^SERVICE_NAME=\K.*' "${APP_DIR}/.env")

echo "=================================================="
echo " Deinstalling Weather Station Bot Service "
echo " Project Directory: ${APP_DIR}"
echo " Service Name:     ${SERVICE_NAME}"
echo "=================================================="

sudo systemctl stop ${SERVICE_NAME}
sudo systemctl disable ${SERVICE_NAME}

sudo rm /etc/systemd/system/${SERVICE_NAME}.service

sudo systemctl daemon-reload
sudo systemctl reset-failed