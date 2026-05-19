#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/viktorwalter/Work/forecasting_framework"
CONFIG_PATH="$REPO_DIR/configs/config_wind_interpol.yaml"
LOCK_FILE="/tmp/run_spatial_interpolation_weekly.lock"
LOG_DIR="$REPO_DIR/logs/cron"

PYTHON_BIN="$REPO_DIR/frcst/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_spatial_interpolation_weekly.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another interpolation run is still active, skipping this schedule." >> "$LOG_FILE"
  exit 0
fi

cd "$REPO_DIR"
"$PYTHON_BIN" geostatistics/run_spatial_interpolation.py \
  --config "$CONFIG_PATH" \
  --log-level INFO >> "$LOG_FILE" 2>&1
