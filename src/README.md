# rect_detector Stage-1 Migration

This directory contains the first-stage package migration for rect detector.

## New package layout

- src/rect_detector/__init__.py
- src/rect_detector/main_inferer.py
- src/rect_detector/align_chip_rects.py
- src/rect_detector/extract_chip_rects.py
- src/rect_detector/rect_models.py
- src/rect_detector/raw_batch_datasetV2.py
- src/rect_detector/yolo_inferers.py
- src/rect_detector/cli/infer_dataloader.py
- src/rect_detector/cli/infer_from_raw_files.py

## Backward compatibility

The old launchers remain available and now delegate to package CLI modules:

- Detectors/rect_detector/run_main_inferer_dataloader.py
- Detectors/rect_detector/run_main_inferer_from_raw_files.py

You can keep using existing commands while gradually moving imports to the package path.

## Recommended next step (Stage-2)

- Split CLI/export logic from infer_dataloader.py into dedicated modules.
- Introduce separate modules for config parsing and JSON export.
- Replace print statements with structured logging.
- Add smoke tests for the dataloader entrypoint and JSON schema.

## Stage-2 split status

The dataloader flow has now been split into:

- `rect_detector.pipeline.dataloader_runner`
	- batch fallback execution
	- no-chip sample checks
	- light-batch normalization
- `rect_detector.export.dataloader_export`
	- JSON-safe serialization helpers
	- per-chip/per-sample export builders
	- final merged payload writer

`rect_detector.cli.infer_dataloader` now focuses on argument parsing and orchestration.

## Product pipeline entrypoint

From `Detectors/rect_detector`, run one pipeline command for a product directory:

```bash
python run_pipeline.py ../../48AMA/imgs/S26F20082-02
```

Or use the shell wrapper:

```bash
./run_pipeline_product.sh
```

Default outputs are stored under the project directory:

- `outputs/json/main_inferer_dataloader_results.json`
- `outputs/predict_input/`
