from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from rect_detector.extract_chip_rects import RectInferer
from rect_detector.main_inferer import MainInferResult, MainInferer
from rect_detector.raw_batch_datasetV2 import build_raw_batch_dataloader
from rect_detector.yolo_inferers import CombinedYoloInferers, YoloPrediction


def _is_no_chip_sample(rect_input_img: np.ndarray, mechanical_infos: Sequence[Any]) -> bool:
    return rect_input_img.size == 0 or len(mechanical_infos) == 0


def _json_default(value: Any) -> Any:
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


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _prediction_source_file_name(prediction: YoloPrediction) -> str:
    mech = prediction.aligned_rect.mech
    mx_text = f"{float(mech.MX):.3f}".replace(".", "p")
    my_text = f"{float(mech.MY):.3f}".replace(".", "p")
    return (
        f"{prediction.light_name}_MX{mx_text}_MY{my_text}_"
        f"img{mech.nImageNum}_idx{mech.nIndex}_rect{prediction.rect_id}.png"
    )


def _structured_predict_input_path(
    prediction: YoloPrediction,
    sample_id: int,
    predict_input_root: Path,
) -> Path:
    mech = prediction.aligned_rect.mech
    mx_text = f"{float(mech.MX):.3f}"
    my_text = f"{float(mech.MY):.3f}"
    folder = (
        predict_input_root
        / prediction.light_name
        / f"sample_id_{sample_id}"
        / f"MX_{mx_text}_MY_{my_text}"
    )
    file_name = f"chip_img{mech.nImageNum}_idx{mech.nIndex}_rect{prediction.rect_id}.png"
    return folder / file_name


def _relocate_predict_input_if_exists(
    prediction: YoloPrediction,
    sample_id: int,
    predict_input_root: Path,
) -> Path | None:
    source_path = predict_input_root / _prediction_source_file_name(prediction)
    if not source_path.exists():
        return None

    target_path = _structured_predict_input_path(
        prediction=prediction,
        sample_id=sample_id,
        predict_input_root=predict_input_root,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path != source_path:
        shutil.move(str(source_path), str(target_path))
    return target_path


def _print_result(index: int, sample_id: int, num_str: str, result: MainInferResult) -> None:
    print(f"\n===== Sample #{index} (sample_id={sample_id}, num_str={num_str}) =====")
    print("Summary")
    print(result.summary)
    print("\nTimings")
    print(result.timings)
    if result.warnings:
        print("\nWarnings")
        for warning in result.warnings:
            print(f"- {warning}")

    print("\nChip results")
    for chip_key, chip in sorted(result.chips.items()):
        light_status = ", ".join(
            f"{light_name}={prediction.pred_status if prediction is not None else 'INVALID'}"
            for light_name, prediction in chip.light_predictions.items()
        )
        print(
            f"chip={chip_key} MX={chip.mechanical_info.MX:.0f} MY={chip.mechanical_info.MY:.0f} "
            f"final={chip.final_status}/{chip.final_class} trigger={chip.trigger_light} "
            f"residual_um={chip.alignment_residual_um:.3f} [{light_status}]"
        )


def _prepare_light_batches(light_image_batches: Sequence[Mapping[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    normalized: list[dict[str, np.ndarray]] = []
    for batch in light_image_batches:
        normalized.append(
            {
                "light_1": batch["light_1"],
                "light_2": batch["light_2"],
                "light_3": batch["light_3"],
                "light_4": batch["light_4"],
            }
        )
    return normalized


def _run_batch_with_fallback(
    inferer: MainInferer,
    rect_input_imgs: Sequence[np.ndarray],
    mechanical_infos: Sequence[Sequence[Any]],
    light_image_batches: Sequence[Mapping[str, np.ndarray]],
    sample_ids: Sequence[int],
    num_strs: Sequence[str],
) -> tuple[list[tuple[int, MainInferResult]], list[tuple[int, str, str]]]:
    try:
        batch_results = inferer.batch_infer(
            rect_input_imgs=rect_input_imgs,
            mechanical_infos=mechanical_infos,
            light_image_batches=light_image_batches,
        )
        return list(enumerate(batch_results)), []
    except Exception as batch_error:
        print(f"  batch_infer failed, fallback to per-sample mode: {batch_error}")

    results: list[tuple[int, MainInferResult]] = []
    skipped: list[tuple[int, str, str]] = []
    for i in range(len(rect_input_imgs)):
        try:
            result = inferer(
                rect_input_img=rect_input_imgs[i],
                mechanical_info=mechanical_infos[i],
                light_images=light_image_batches[i],
            )
            results.append((i, result))
        except Exception as sample_error:
            skipped.append((sample_ids[i], num_strs[i], str(sample_error)))
    return results, skipped


def _build_chip_record(
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
            saved_path = _relocate_predict_input_if_exists(
                prediction=prediction,
                sample_id=sample_id,
                predict_input_root=predict_input_root,
            )

        path_text = _safe_relpath(saved_path, workspace_root) if saved_path is not None else None
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


def _build_sample_record(
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
            _build_chip_record(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MainInferer in DataLoader mode from a raw dataset root directory."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Dataset root directory, e.g. 48AMA/imgs/S26F20082-02",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable persistent DataLoader workers when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetch factor per worker when num_workers > 0.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--unsafe-missing-ok", action="store_true", help="Do not fail if some files are missing.")
    parser.add_argument(
        "--light-read-workers",
        type=int,
        default=4,
        help="Per-sample thread count for parallel light image reads (1 disables).",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after N dataloader batches (0 means all).")
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after N inferred samples (0 means all).")
    parser.add_argument(
        "--yolo-config",
        type=Path,
        default=Path(__file__).with_name("combined_yolo_inferers_config.example.json"),
        help="CombinedYoloInferers JSON config path.",
    )
    parser.add_argument("--x-scale", type=float, default=0.30)
    parser.add_argument("--y-scale", type=float, default=0.50)
    parser.add_argument("--mech-delta-x", type=float, default=1421.0)
    parser.add_argument("--mech-delta-y", type=float, default=283.0)
    parser.add_argument(
        "--strict-align",
        action="store_true",
        help="Require rect and mechanical counts to match; default allows partial alignment.",
    )
    parser.add_argument(
        "--trace-batches",
        action="store_true",
        help="Print how multiple samples are merged into per-light YOLO batches.",
    )
    parser.add_argument(
        "--save-predict-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save original crops passed to model.predict and include path_ in exported JSON.",
    )
    parser.add_argument(
        "--predict-input-root",
        type=Path,
        default=Path("temp") / "predict_input",
        help="Root directory for saved model.predict input crops.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("temp") / "main_inferer_from_raw_files_results.json",
        help="Final merged JSON output path.",
    )
    parser.add_argument(
        "--print-chip-results",
        action="store_true",
        help="Print per-chip details for each inferred sample.",
    )
    args = parser.parse_args()

    workspace_root = Path.cwd()
    args.predict_input_root.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    dataloader = build_raw_batch_dataloader(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        strict=not args.unsafe_missing_ok,
        light_read_workers=args.light_read_workers,
    )

    rect_inferer = RectInferer(
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        mech_delta_x=args.mech_delta_x,
        mech_delta_y=args.mech_delta_y,
    )
    combined_yolo_inferers = CombinedYoloInferers.from_json(args.yolo_config)
    if args.trace_batches:
        combined_yolo_inferers.set_trace_batching(True)
    combined_yolo_inferers.set_save_predict_input(args.save_predict_input)
    combined_yolo_inferers.set_predict_input_dir(args.predict_input_root)

    main_inferer = MainInferer(
        rect_inferer=rect_inferer,
        yolo_inferers=combined_yolo_inferers,
        allow_partial=not args.strict_align,
    )

    total_samples_seen = 0
    total_inferred_samples = 0
    total_skipped_no_chip = 0
    total_skipped_error = 0
    total_batches = 0
    run_start = perf_counter()

    exported_samples: list[dict[str, Any]] = []
    exported_skipped_errors: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(dataloader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break

        sample_ids = list(batch["sample_ids"])
        num_strs = list(batch["num_strs"])
        rect_input_imgs = list(batch["rect_input_imgs"])
        mechanical_infos_batch = list(batch["mechanical_infos"])
        light_image_batches = list(batch["light_images"])
        total_samples_seen += len(sample_ids)
        total_batches += 1

        valid_indices: list[int] = []
        for idx, (rect_input_img, mechanical_infos) in enumerate(zip(rect_input_imgs, mechanical_infos_batch)):
            if _is_no_chip_sample(rect_input_img=rect_input_img, mechanical_infos=mechanical_infos):
                total_skipped_no_chip += 1
                continue
            valid_indices.append(idx)

        if not valid_indices:
            print(f"batch[{batch_index}] skipped: all {len(sample_ids)} samples have no chips")
            continue

        valid_rect_input_imgs = [rect_input_imgs[i] for i in valid_indices]
        valid_mechanical_infos = [mechanical_infos_batch[i] for i in valid_indices]
        valid_light_image_batches = _prepare_light_batches([light_image_batches[i] for i in valid_indices])
        valid_sample_ids = [sample_ids[i] for i in valid_indices]
        valid_num_strs = [num_strs[i] for i in valid_indices]

        batch_start = perf_counter()
        indexed_results, skipped_on_error = _run_batch_with_fallback(
            inferer=main_inferer,
            rect_input_imgs=valid_rect_input_imgs,
            mechanical_infos=valid_mechanical_infos,
            light_image_batches=valid_light_image_batches,
            sample_ids=valid_sample_ids,
            num_strs=valid_num_strs,
        )
        batch_elapsed = perf_counter() - batch_start

        total_inferred_samples += len(indexed_results)
        total_skipped_error += len(skipped_on_error)

        print(
            f"batch[{batch_index}] inferred={len(indexed_results)} "
            f"skipped_no_chip={len(sample_ids) - len(valid_indices)} skipped_error={len(skipped_on_error)} "
            f"elapsed_s={batch_elapsed:.3f}"
        )

        for sample_id, num_str, error_msg in skipped_on_error:
            print(f"  skipped sample_id={sample_id} num_str={num_str} error={error_msg}")
            exported_skipped_errors.append(
                {
                    "sample_id": sample_id,
                    "num_str": num_str,
                    "error": error_msg,
                }
            )

        for local_index, result in indexed_results:
            sample_id = valid_sample_ids[local_index]
            num_str = valid_num_strs[local_index]
            if not args.print_chip_results:
                summary = result.summary
                print(
                    f"  sample_id={sample_id} num_str={num_str} "
                    f"aligned={summary.aligned_count} OK={summary.ok_count} NG={summary.ng_count} "
                    f"INVALID={summary.invalid_count} total_s={result.timings.total_s:.3f}"
                )
            else:
                _print_result(
                    index=local_index,
                    sample_id=sample_id,
                    num_str=num_str,
                    result=result,
                )

            exported_samples.append(
                _build_sample_record(
                    result=result,
                    sample_id=sample_id,
                    num_str=num_str,
                    predict_input_root=args.predict_input_root,
                    save_predict_input=args.save_predict_input,
                    workspace_root=workspace_root,
                )
            )

        if args.max_samples > 0 and total_inferred_samples >= args.max_samples:
            break

    run_elapsed = perf_counter() - run_start
    throughput = total_inferred_samples / run_elapsed if run_elapsed > 0 else 0.0

    exported = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": _safe_relpath(args.root_dir, workspace_root),
        "yolo_config": _safe_relpath(args.yolo_config, workspace_root),
        "predict_input_root": _safe_relpath(args.predict_input_root, workspace_root),
        "summary": {
            "batches_processed": total_batches,
            "samples_seen": total_samples_seen,
            "samples_inferred": total_inferred_samples,
            "samples_skipped_no_chip": total_skipped_no_chip,
            "samples_skipped_error": total_skipped_error,
            "elapsed_s": run_elapsed,
            "inferred_samples_per_s": throughput,
        },
        "args": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "shuffle": args.shuffle,
            "unsafe_missing_ok": args.unsafe_missing_ok,
            "light_read_workers": args.light_read_workers,
            "max_batches": args.max_batches,
            "max_samples": args.max_samples,
            "strict_align": args.strict_align,
            "trace_batches": args.trace_batches,
            "save_predict_input": args.save_predict_input,
            "x_scale": args.x_scale,
            "y_scale": args.y_scale,
            "mech_delta_x": args.mech_delta_x,
            "mech_delta_y": args.mech_delta_y,
        },
        "samples": exported_samples,
        "skipped_errors": exported_skipped_errors,
    }

    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(exported, file, ensure_ascii=False, indent=2, default=_json_default)

    print("\n===== Run Summary =====")
    print(f"batches_processed={total_batches}")
    print(f"samples_seen={total_samples_seen}")
    print(f"samples_inferred={total_inferred_samples}")
    print(f"samples_skipped_no_chip={total_skipped_no_chip}")
    print(f"samples_skipped_error={total_skipped_error}")
    print(f"elapsed_s={run_elapsed:.3f}")
    print(f"inferred_samples_per_s={throughput:.2f}")
    print(f"export_json={_safe_relpath(args.output_json, workspace_root)}")


if __name__ == "__main__":
    main()

"""
python Detectors/rect_detector/run_main_inferer_from_raw_files.py 48AMA/imgs/S26F20082-02 \
    --batch-size 4 --num-workers 4 --persistent-workers --prefetch-factor 2 --light-read-workers 4 \
    --print-chip-results

python Detectors/rect_detector/run_main_inferer_from_raw_files.py 48AMA/imgs/S26F20082-02 --batch-size 4 --num-workers 4 --persistent-workers --prefetch-factor 2 --light-read-workers 4 --print-chip-results --predict-input-root temp/predict_input_verify --output-json temp/main_inferer_from_raw_files_verify.json --max-batches 10
"""
