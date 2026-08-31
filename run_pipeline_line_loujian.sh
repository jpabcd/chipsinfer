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

PRODUCT_CONFIG="${PRODUCT_CONFIG:-configs/products/48AMA_loujian.json}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/production_loujian.json}"

"$PYTHON_BIN" run_pipeline_line.py \
  --product-config "$PRODUCT_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  "$@"


#需要保存时显式传入：./run_pipeline_line_loujian.sh --save-predict-input --save-predict-input-on-any-light-ng
#通常产线：./run_pipeline_line_loujian.sh --output-root outputs/48AMA_line --watch --rescan-interval 30