#!/usr/bin/env bash
set -euo pipefail

# Process all completed 48AMA products in chronological directory-mtime order.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../../.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

IMGS_ROOT="${IMGS_ROOT:-../../48AMA/imgs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/line_run_$(date +%Y%m%d_%H%M%S)}"

"$PYTHON_BIN" run_pipeline_line.py "$IMGS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --persistent-workers \
  --prefetch-factor "${PREFETCH_FACTOR:-2}" \
  --light-read-workers "${LIGHT_READ_WORKERS:-4}" \
  "$@"


#需要保存时显式传入：./run_pipeline_line.sh --save-predict-input
#通常产线：./run_pipeline_line.sh --output-root outputs/48AMA_line --watch --rescan-interval 30