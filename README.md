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
- `run_main_inferer_dataloader.py`
  - backward-compatible dataloader entrypoint
- `run_main_inferer_from_raw_files.py`
  - backward-compatible raw-files entrypoint

## Output Folders

- `outputs/json/`
  - final merged inference JSON
- `outputs/predict_input/`
  - optional saved model input crops

## Web Visualization

Run the web app from `inspection_web_app/`:

```bash
python app.py --host 127.0.0.1 --port 7868
```

By default it reads:

`outputs/json/main_inferer_dataloader_results.json`
