#!/usr/bin/env bash
set -euo pipefail

# Run from Detectors/rect_detector by default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${1:-127.0.0.1}"
PORT="${2:-7868}"
RESULT_JSON="${3:-$SCRIPT_DIR/outputs/json/main_inferer_dataloader_results.json}"
PRJ_ROOT="${4:-$SCRIPT_DIR}"

if [[ -x "../../.venv/bin/python" ]]; then
  PYTHON="../../.venv/bin/python"
elif [[ -x "../.venv/bin/python" ]]; then
  PYTHON="../.venv/bin/python"
else
  PYTHON="python"
fi

"$PYTHON" inspection_web_app/app.py \
  --host "$HOST" \
  --port "$PORT" \
  --result-json "$RESULT_JSON" \
  --project-dir "$PRJ_ROOT"