#!/usr/bin/env bash
set -euo pipefail

# Run from Detectors/rect_detector directory by default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../../.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

EXTERNAL_XY_CSV_PATH="${EXTERNAL_XY_CSV_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
WAFER_MAP_WIDTH="${WAFER_MAP_WIDTH:-20}"
WAFER_MAP_HEIGHT="${WAFER_MAP_HEIGHT:-16}"
WAFER_MAP_CHIP_ASPECT="${WAFER_MAP_CHIP_ASPECT:-5.0}"

CMD=(
  "$PYTHON_BIN" run_pipeline.py ../../48AMA/imgs/S26G03090-06
  --batch-size 2
  --num-workers 4
  --persistent-workers
  --prefetch-factor 2
  --light-read-workers 4
  --no-save-predict-input
  #--save-predict-input
  --output-dir "$OUTPUT_DIR"
  --output-json "$OUTPUT_DIR/json/main_inferer_dataloader_results.json"
  --predict-input-root "$OUTPUT_DIR/predict_input"
  --defect-report
  --defect-report-path "$OUTPUT_DIR/csv/S26G03090-06.csv"
  --external-xy-csv-path "../../48AMA/imgs/S26G03090-06/S26G03090-06.csv"
  --wafer-map-path "$OUTPUT_DIR/plots/wafer_map_overall.png"
  --wafer-map-figsize "$WAFER_MAP_WIDTH" "$WAFER_MAP_HEIGHT"
  --wafer-map-chip-aspect "$WAFER_MAP_CHIP_ASPECT"
)

if [[ -n "$EXTERNAL_XY_CSV_PATH" ]]; then
  CMD+=(--external-xy-csv-path "$EXTERNAL_XY_CSV_PATH")
fi

"${CMD[@]}"
