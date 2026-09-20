#!/usr/bin/env bash

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/backend/weather-backend
SERVICE_NAME=$(grep -oP '^SERVICE_NAME=\K.*' "${APP_DIR}/.env")

sudo systemctl stop ${SERVICE_NAME}
sudo systemctl disable ${SERVICE_NAME}

sudo rm /etc/systemd/system/${SERVICE_NAME}.service

sudo systemctl daemon-reload
sudo systemctl reset-failed