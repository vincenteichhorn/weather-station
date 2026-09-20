#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration Variables
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/backend
SERVICE_NAME=$(grep -oP '^SERVICE_NAME=\K.*' "${APP_DIR}/.env")
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
APP_USER="${USER}" # Defaults to current user

echo "=========================================="
echo " Starting Weather Station Backend w/ Poetry Setup "
echo " Project Directory: ${APP_DIR}"
echo " Service User:      ${APP_USER}"
echo "=========================================="

# 1. Ensure Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# 2. Check for Poetry installation
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Instal Poetry."
    exit 1
else
    echo "Poetry is already installed."
fi

# 3. Configure Poetry to create virtual environment inside the project directory (.venv)
poetry config virtualenvs.in-project true --local

# 4. Check for pyproject.toml 
if [ ! -f "${APP_DIR}/pyproject.toml" ]; then
    echo "No pyproject.toml found"
    exit 1
fi

# 5. Install project dependencies from lockfile
echo "Installing project dependencies with Poetry..."
poetry install --no-root

# Get absolute path to the virtual environment created by Poetry
POETRY_VENV="$(poetry env info --path)"
UVICORN_EXEC="${POETRY_VENV}/bin/uvicorn"

# 6. Generate systemd service file
echo "Configuring systemd service..."

SERVICE_CONTENT="[Unit]
Description=Weather Station Backend Service
After=network.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}/weather-backend
ExecStart=${UVICORN_EXEC} main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3
Environment=\"PATH=${POETRY_VENV}/bin\"

[Install]
WantedBy=multi-user.target"

# Write service file using sudo
echo "${SERVICE_CONTENT}" | sudo tee "${SYSTEMD_PATH}" > /dev/null

# 7. Enable and start systemd service
echo "Reloading systemd daemon and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "=========================================="
echo " Setup Complete!"
echo " Service Status:"
echo "=========================================="
sudo systemctl status "${SERVICE_NAME}" --no-pager