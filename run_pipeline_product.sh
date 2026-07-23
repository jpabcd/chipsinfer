#!/usr/bin/env bash
set -euo pipefail

# Run from Detectors/rect_detector directory by default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXTERNAL_XY_CSV_PATH="${EXTERNAL_XY_CSV_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
WAFER_MAP_WIDTH="${WAFER_MAP_WIDTH:-20}"
WAFER_MAP_HEIGHT="${WAFER_MAP_HEIGHT:-16}"
WAFER_MAP_CHIP_ASPECT="${WAFER_MAP_CHIP_ASPECT:-5.0}"

CMD=(
  python run_pipeline.py ../../48AMA/imgs/S26F30091-09
  --batch-size 2
  --num-workers 8
  --persistent-workers
  --prefetch-factor 2
  --light-read-workers 4
  #--no-save-predict-input
  --save-predict-input
  --output-dir "$OUTPUT_DIR"
  --output-json "$OUTPUT_DIR/json/main_inferer_dataloader_results.json"
  --predict-input-root "$OUTPUT_DIR/predict_input"
  --defect-report
  --defect-report-path "$OUTPUT_DIR/csv/S26F30091-09.csv"
  --external-xy-csv-path "../../48AMA/imgs/S26F30091-09/S26F30091-09.csv"
  --wafer-map-path "$OUTPUT_DIR/plots/wafer_map_overall.png"
  --wafer-map-figsize "$WAFER_MAP_WIDTH" "$WAFER_MAP_HEIGHT"
  --wafer-map-chip-aspect "$WAFER_MAP_CHIP_ASPECT"
)

if [[ -n "$EXTERNAL_XY_CSV_PATH" ]]; then
  CMD+=(--external-xy-csv-path "$EXTERNAL_XY_CSV_PATH")
fi

"${CMD[@]}"
