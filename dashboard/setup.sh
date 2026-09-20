#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration Variables
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME=$(grep -oP '^SERVICE_NAME=\K.*' "${APP_DIR}/.env")
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
APP_USER="${USER}" # Defaults to current user
APP_PORT=$(grep -oP '^APP_PORT=\K.*' "${APP_DIR}/.env")

echo "=================================================="
echo " Starting Weather Dashboard w/ Poetry Setup "
echo " Project Directory: ${APP_DIR}"
echo " Service User:      ${APP_USER}"
echo " Service Name:     ${SERVICE_NAME}"
echo " App Port:        ${APP_PORT}"
echo "=================================================="

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
cd "${APP_DIR}"
poetry install --no-root

# Get absolute path to the virtual environment created by Poetry
POETRY_VENV="$(poetry env info --path)"
STREAMLIT_EXEC="${POETRY_VENV}/bin/streamlit"

# 6. Generate systemd service file
echo "Configuring systemd service..."

SERVICE_CONTENT="[Unit]
Description=Weather Station Bot Service
After=network.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}/weather-bot
ExecStart=${STREAMLIT_EXEC} run main.py --server.port ${APP_PORT}
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