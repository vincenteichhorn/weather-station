#!/usr/bin/env bash

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME=$(grep -oP '^SERVICE_NAME=\K.*' "${APP_DIR}/.env")
APP_PORT=$(grep -oP '^APP_PORT=\K.*' "${APP_DIR}/.env")

echo "=================================================="
echo " Deinstalling Weather Station Dashboard Service "
echo " Project Directory: ${APP_DIR}"
echo " Service Name:     ${SERVICE_NAME}"
echo " App Port:        ${APP_PORT}"
echo "=================================================="

sudo systemctl stop ${SERVICE_NAME}
sudo systemctl disable ${SERVICE_NAME}

sudo rm /etc/systemd/system/${SERVICE_NAME}.service

sudo systemctl daemon-reload
sudo systemctl reset-failed