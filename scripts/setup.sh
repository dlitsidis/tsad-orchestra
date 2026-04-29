#! /bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============ SETTING UP PYTHON PACKAGES ============"
bash $SCRIPT_DIR/setup-uv.sh

echo "============ SETTING UP UV ENVIRONMENT ============"
uv sync

echo "============ STARTING DOCKER CONTAINERS ============"
docker compose up -d

echo "Waiting for database to be ready..."
sleep 10

echo "============ RUNNING DB MIGRATION ============"
uv run scripts/db_migration/tsb-uad.py

echo "Setup complete!"
