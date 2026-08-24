from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rect_detector.main_inferer import MainInferResult
from rect_detector.yolo_inferers import YoloPrediction


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def prediction_source_file_name(prediction: YoloPrediction) -> str:
    mech = prediction.aligned_rect.mech
    return (
        f"{prediction.light_name}_img{mech.nImageNum}_"
        f"idx{mech.nIndex}_rect{prediction.rect_id}.png"
    )


def structured_predict_input_path(
    prediction: YoloPrediction,
    sample_id: int,
    predict_input_root: Path,
) -> Path:
    del sample_id
    return predict_input_root / prediction.light_name / prediction_source_file_name(prediction)


def relocate_predict_input_if_exists(
    prediction: YoloPrediction,
    sample_id: int,
    predict_input_root: Path,
) -> Path | None:
    target_path = structured_predict_input_path(
        prediction=prediction,
        sample_id=sample_id,
        predict_input_root=predict_input_root,
    )
    if not target_path.exists():
        return None
    return target_path


def build_chip_record(
    chip: Any,
    sample_id: int,
    predict_input_root: Path,
    save_predict_input: bool,
    workspace_root: Path,
) -> dict[str, Any]:
    light_records: dict[str, Any] = {}
    path_map: dict[str, str | None] = {}

    for light_name in sorted(chip.light_predictions):
        prediction = chip.light_predictions[light_name]
        if prediction is None:
            light_records[light_name] = {
                "path_": None,
                "raw_trt_output": None,
                "analyzed_output": None,
            }
            path_map[light_name] = None
            continue

        saved_path: Path | None = None
        if save_predict_input:
            saved_path = relocate_predict_input_if_exists(
                prediction=prediction,
                sample_id=sample_id,
                predict_input_root=predict_input_root,
            )

        path_text = safe_relpath(saved_path, workspace_root) if saved_path is not None else None
        path_map[light_name] = path_text
        light_records[light_name] = {
            "path_": path_text,
            "raw_trt_output": prediction.to_dict(),
            "analyzed_output": {
                "pred_status": prediction.pred_status,
                "pred_class": sorted(prediction.pred_class),
                "decision_reason": sorted(prediction.decision_reason),
                "yolo_ms": prediction.yolo_ms,
            },
        }

    crop_box = chip.crop_box
    return {
        "chip_key": {
            "nImageNum": chip.chip_key[0],
            "nIndex": chip.chip_key[1],
        },
        "path_": path_map,
        "mechanical_columns": chip.mechanical_info.to_dict(),
        "alignment": {
            "residual_um": chip.alignment_residual_um,
            "aligned_rect": chip.aligned_rect.to_dict(),
            "crop_box": (
                {
                    "x1": crop_box.x1,
                    "y1": crop_box.y1,
                    "x2": crop_box.x2,
                    "y2": crop_box.y2,
                }
                if crop_box is not None
                else None
            ),
        },
        "yolo": light_records,
        "final": {
            "status": chip.final_status,
            "class": chip.final_class,
            "reason": chip.decision_reason,
            "trigger_light": chip.trigger_light,
        },
    }


def build_sample_record(
    result: MainInferResult,
    sample_id: int,
    num_str: str,
    predict_input_root: Path,
    save_predict_input: bool,
    workspace_root: Path,
) -> dict[str, Any]:
    chips = []
    for _, chip in sorted(result.chips.items()):
        chips.append(
            build_chip_record(
                chip=chip,
                sample_id=sample_id,
                predict_input_root=predict_input_root,
                save_predict_input=save_predict_input,
                workspace_root=workspace_root,
            )
        )

    return {
        "sample_id": sample_id,
        "num_str": num_str,
        "summary": asdict(result.summary),
        "timings": asdict(result.timings),
        "warnings": result.warnings,
        "align_result": {
            "rmse_um": result.align_result.rmse_um,
            "max_residual_um": result.align_result.max_residual_um,
            "x_reversed": result.align_result.x_reversed,
            "y_reversed": result.align_result.y_reversed,
            "is_partial": result.align_result.is_partial,
            "used_delta_prior": result.align_result.used_delta_prior,
            "mech_delta_x": result.align_result.mech_delta_x,
            "mech_delta_y": result.align_result.mech_delta_y,
            "delta_penalty_um": result.align_result.delta_penalty_um,
            "unmatched_rect_count": len(result.align_result.unmatched_rects),
            "unmatched_mech_count": len(result.align_result.unmatched_mechs),
        },
        "chips": chips,
    }


def build_export_payload(
    *,
    workspace_root: Path,
    root_dir: Path,
    yolo_config: Path,
    predict_input_root: Path,
    output_args: dict[str, Any],
    total_batches: int,
    total_samples_seen: int,
    total_samples_inferred: int,
    total_samples_skipped_no_chip: int,
    total_samples_skipped_error: int,
    elapsed_s: float,
    samples: list[dict[str, Any]],
    skipped_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    throughput = total_samples_inferred / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": safe_relpath(root_dir, workspace_root),
        "yolo_config": safe_relpath(yolo_config, workspace_root),
        "predict_input_root": safe_relpath(predict_input_root, workspace_root),
        "summary": {
            "batches_processed": total_batches,
            "samples_seen": total_samples_seen,
            "samples_inferred": total_samples_inferred,
            "samples_skipped_no_chip": total_samples_skipped_no_chip,
            "samples_skipped_error": total_samples_skipped_error,
            "elapsed_s": elapsed_s,
            "inferred_samples_per_s": throughput,
        },
        "args": output_args,
        "samples": samples,
        "skipped_errors": skipped_errors,
    }


def write_export_json(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)


def write_export_json_streamed(
    output_json: Path,
    payload_without_arrays: dict[str, Any],
    samples_jsonl: Path,
    skipped_errors_jsonl: Path,
) -> None:
    """Write final export JSON while streaming large arrays from jsonl files.

    payload_without_arrays must not contain keys: samples, skipped_errors.
    """

    def _write_jsonl_array(file_obj: Any, jsonl_path: Path, indent: str = "  ") -> None:
        child_indent = indent + "  "
        wrote_item = False
        if jsonl_path.is_file():
            with jsonl_path.open("r", encoding="utf-8") as source:
                for raw_line in source:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if wrote_item:
                        file_obj.write(",\n")
                    file_obj.write(f"{child_indent}{line}")
                    wrote_item = True
        if wrote_item:
            file_obj.write("\n")
        file_obj.write(f"{indent}]")

    payload = dict(payload_without_arrays)
    payload.pop("samples", None)
    payload.pop("skipped_errors", None)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        file.write("{\n")
        items = list(payload.items())
        for index, (key, value) in enumerate(items):
            key_text = json.dumps(key, ensure_ascii=False)
            value_text = json.dumps(value, ensure_ascii=False, default=json_default)
            file.write(f"  {key_text}: {value_text}")
            file.write(",\n" if index < len(items) - 1 else ",\n")

        file.write("  \"samples\": [\n")
        _write_jsonl_array(file, samples_jsonl, indent="  ")
        file.write(",\n")
        file.write("  \"skipped_errors\": [\n")
        _write_jsonl_array(file, skipped_errors_jsonl, indent="  ")
        file.write("\n}\n")
