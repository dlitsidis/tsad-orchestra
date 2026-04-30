#! /bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============ SETTING UP PYTHON PACKAGES ============"
bash $SCRIPT_DIR/setup-uv.sh

echo "============ SETTING UP UV ENVIRONMENT ============"
uv sync

echo "============ STARTING DOCKER CONTAINERS ============"
docker compose up -d

echo "============ RUNNING DB MIGRATION ============"
MAX_MIGRATION_ATTEMPTS=12
MIGRATION_RETRY_DELAY=5
ATTEMPT=1

while ! uv run scripts/db_migration/tsb-uad.py; do
  if [ "$ATTEMPT" -ge "$MAX_MIGRATION_ATTEMPTS" ]; then
    echo "Database migration failed after $MAX_MIGRATION_ATTEMPTS attempts."
    exit 1
  fi

  echo "Database not ready yet for migration. Retrying in $MIGRATION_RETRY_DELAY seconds... (attempt $ATTEMPT/$MAX_MIGRATION_ATTEMPTS)"
  sleep "$MIGRATION_RETRY_DELAY"
  ATTEMPT=$((ATTEMPT + 1))
done
echo "Setup complete!"
