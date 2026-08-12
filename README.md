# rect_detector

This directory is the project root for the rect detector pipeline.

## Keep Directory

`inspection_web_app/` is a required component and must be kept.
It visualizes the final pipeline JSON output and supports manual review.

## Main Run Entrypoints

- `run_pipeline.py`
  - unified product-level pipeline entrypoint
- `run_pipeline_product.sh`
  - one-command wrapper for `../../48AMA/imgs/S26F20082-02`
- `run_pipeline_line.py`
  - chronological production-line runner for all completed product folders in `../../48AMA/imgs`
- `run_pipeline_line.sh`
  - one-command 48AMA production-line wrapper
- `run_main_inferer_dataloader.py`
  - backward-compatible dataloader entrypoint
- `run_main_inferer_from_raw_files.py`
  - backward-compatible raw-files entrypoint

## Output Folders

- `outputs/json/`
  - final merged inference JSON
- `outputs/predict_input/`
  - optional saved model input crops

## 48AMA Production-Line Inference

The line runner separates configuration into two files:

- `configs/products/48AMA.json`: product geometry, rectangle detection, alignment, YOLO model config, and wafer-map geometry.
- `configs/runtime/production.json`: input/output roots, line scanning behavior, DataLoader performance settings, and output switches.

For production line B, copy both examples, change the product-related values in the product profile and the machine/path values in the runtime profile, then run:

```bash
./run_pipeline_line.sh --product-config configs/products/LINE_B_PRODUCT.json --runtime-config configs/runtime/line_b.json
```

Relative paths in both profiles are resolved from the project root. Explicit command-line arguments override JSON values.

Process every completed product directory under `../../48AMA/imgs` in directory modification-time order:

```bash
./run_pipeline_line.sh
```

Each product has isolated artifacts under `outputs/line_run_YYYYmmdd_HHMMSS/products/PRODUCT_NAME/`. Product artifacts use the product identifier in their names, for example `json/PRODUCT_NAME.json`, `csv/PRODUCT_NAME.csv`, and `plots/PRODUCT_NAME_wafer_map.png`. The `MainInferer` and YOLO model weights are initialized for the first completed product and then reused for the remaining products in the same line-run process.
The line-level `line_summary.json` records completed, failed, and incomplete products. Folders without all four `LightN-raw` directories or an `IMAGE3_*.raw` file are skipped by default, so a wafer still being written is not inferred prematurely.

For continuous operation, keep a stable output root and enable watch mode:

```bash
./run_pipeline_line.sh --output-root outputs/48AMA_line --watch --rescan-interval 30
```

The new runner forwards normal product-inference options, such as `--batch-size`, `--yolo-config`, `--strict-align`, and `--no-save-predict-input`. It manages output paths per product itself. A product-level CSV named `PRODUCT_NAME.csv` is detected automatically and used to generate the product wafer map.

## Web Visualization

Run the web app from `inspection_web_app/`:

```bash
python app.py --host 127.0.0.1 --port 7868
```

By default it reads:

`outputs/json/main_inferer_dataloader_results.json`
