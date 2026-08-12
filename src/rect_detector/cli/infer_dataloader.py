from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from rect_detector.export.dataloader_export import (
    build_export_payload,
    build_sample_record,
    safe_relpath,
    write_export_json,
)
from rect_detector.extract_chip_rects import RectInferer
from rect_detector.main_inferer import MainInferer
from rect_detector.pipeline.dataloader_runner import (
    is_no_chip_sample,
    prepare_light_batches,
    print_sample_result,
    run_batch_with_fallback,
)
from rect_detector.raw_batch_datasetV2 import build_raw_batch_dataloader
from rect_detector.yolo_inferers import CombinedYoloInferers


def _default_yolo_config_path() -> Path:
    config_name = "combined_yolo_inferers_config.example.json"
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[3] / config_name,
        Path.cwd() / "Detectors" / "rect_detector" / config_name,
        module_path.with_name(config_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(
    argv: list[str] | None = None,
    inferer: MainInferer | None = None,
) -> MainInferer:
    parser = argparse.ArgumentParser(description="Run MainInferer in DataLoader mode from raw files.")
    parser.add_argument("root_dir", type=Path, help="Dataset root directory, e.g. 48AMA/imgs/S26F20082-02")
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
        default=_default_yolo_config_path(),
        help="CombinedYoloInferers JSON config path.",
    )
    parser.add_argument("--x-scale", type=float, default=0.30)
    parser.add_argument("--y-scale", type=float, default=0.50)
    parser.add_argument("--threshold", type=int, default=18)
    parser.add_argument("--x-dilate", type=int, default=45)
    parser.add_argument("--y-dilate", type=int, default=14)
    parser.add_argument("--min-width", type=int, default=600)
    parser.add_argument("--max-width", type=int, default=950)
    parser.add_argument("--min-height", type=int, default=90)
    parser.add_argument("--max-height", type=int, default=220)
    parser.add_argument("--min-aspect", type=float, default=3.6)
    parser.add_argument("--max-aspect", type=float, default=8.5)
    parser.add_argument("--min-area", type=int, default=2500)
    parser.add_argument("--margin", type=int, default=100)
    parser.add_argument("--mech-delta-x", type=float, default=1421.0)
    parser.add_argument("--mech-delta-y", type=float, default=283.0)
    parser.add_argument(
        "--delta-x-pixel",
        type=float,
        default=950.0,
        help="Expected horizontal center spacing between adjacent chips in image pixels.",
    )
    parser.add_argument(
        "--delta-y-pixel",
        type=float,
        default=185.0,
        help="Expected vertical center spacing between adjacent chips in image pixels.",
    )
    parser.add_argument(
        "--strict-align",
        action="store_true",
        help="Require rect and mechanical counts to match; default allows partial alignment.",
    )
    parser.add_argument("--allow-x-reverse", action="store_true")
    parser.add_argument("--allow-y-reverse", action="store_true")
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
        default=Path("outputs") / "predict_input",
        help="Root directory for saved model.predict input crops.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs") / "json" / "main_inferer_dataloader_results.json",
        help="Final merged JSON output path.",
    )
    args = parser.parse_args(argv)

    project_root = _project_root()
    if not args.yolo_config.is_absolute():
        args.yolo_config = (project_root / args.yolo_config).resolve()
    if not args.predict_input_root.is_absolute():
        args.predict_input_root = (project_root / args.predict_input_root).resolve()
    if not args.output_json.is_absolute():
        args.output_json = (project_root / args.output_json).resolve()

    workspace_root = Path.cwd()
    args.predict_input_root.mkdir(parents=True, exist_ok=True)

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

    if inferer is None:
        rect_inferer = RectInferer(
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            threshold=args.threshold,
            x_dilate=args.x_dilate,
            y_dilate=args.y_dilate,
            min_width=args.min_width,
            max_width=args.max_width,
            min_height=args.min_height,
            max_height=args.max_height,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            min_area=args.min_area,
            margin=args.margin,
            mech_delta_x=args.mech_delta_x,
            mech_delta_y=args.mech_delta_y,
            delta_x_pixel=args.delta_x_pixel,
            delta_y_pixel=args.delta_y_pixel,
        )
        combined_yolo_inferers = CombinedYoloInferers.from_json(args.yolo_config)
        inferer = MainInferer(
            rect_inferer=rect_inferer,
            yolo_inferers=combined_yolo_inferers,
            allow_partial=not args.strict_align,
            allow_x_reverse=args.allow_x_reverse,
            allow_y_reverse=args.allow_y_reverse,
        )

    # The shared inferer retains model weights, while these settings are specific
    # to the product currently being processed.
    inferer.allow_partial = not args.strict_align
    inferer.allow_x_reverse = args.allow_x_reverse
    inferer.allow_y_reverse = args.allow_y_reverse
    inferer.yolo_inferers.set_trace_batching(args.trace_batches)
    inferer.yolo_inferers.set_predict_input_dir(args.predict_input_root)
    inferer.yolo_inferers.set_save_predict_input(args.save_predict_input)

    total_batches = 0
    total_samples_seen = 0
    total_samples_inferred = 0
    total_samples_skipped_no_chip = 0
    total_samples_skipped_error = 0
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

        total_batches += 1
        total_samples_seen += len(sample_ids)

        valid_indices: list[int] = []
        for i, (rect_input_img, mechanical_infos) in enumerate(zip(rect_input_imgs, mechanical_infos_batch)):
            if is_no_chip_sample(rect_input_img=rect_input_img, mechanical_infos=mechanical_infos):
                total_samples_skipped_no_chip += 1
                continue
            valid_indices.append(i)

        if not valid_indices:
            print(f"batch[{batch_index}] skipped: all {len(sample_ids)} samples have no chips")
            continue

        valid_rect_input_imgs = [rect_input_imgs[i] for i in valid_indices]
        valid_mechanical_infos = [mechanical_infos_batch[i] for i in valid_indices]
        valid_light_image_batches = prepare_light_batches([light_image_batches[i] for i in valid_indices])
        valid_sample_ids = [sample_ids[i] for i in valid_indices]
        valid_num_strs = [num_strs[i] for i in valid_indices]

        infer_start = perf_counter()
        indexed_results, skipped_on_error = run_batch_with_fallback(
            inferer=inferer,
            rect_input_imgs=valid_rect_input_imgs,
            mechanical_infos=valid_mechanical_infos,
            light_image_batches=valid_light_image_batches,
            sample_ids=valid_sample_ids,
            num_strs=valid_num_strs,
        )
        infer_elapsed = perf_counter() - infer_start
        total_samples_inferred += len(indexed_results)
        total_samples_skipped_error += len(skipped_on_error)

        print(
            f"batch[{batch_index}] inferred={len(indexed_results)} "
            f"skipped_no_chip={len(sample_ids) - len(valid_indices)} skipped_error={len(skipped_on_error)} "
            f"elapsed_s={infer_elapsed:.3f}"
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

        for local_valid_index, result in indexed_results:
            sample_id = valid_sample_ids[local_valid_index]
            num_str = valid_num_strs[local_valid_index]
            print_sample_result(
                result=result,
                sample_id=sample_id,
                num_str=num_str,
            )
            exported_samples.append(
                build_sample_record(
                    result=result,
                    sample_id=sample_id,
                    num_str=num_str,
                    predict_input_root=args.predict_input_root,
                    save_predict_input=args.save_predict_input,
                    workspace_root=workspace_root,
                )
            )

        if args.max_samples > 0 and total_samples_inferred >= args.max_samples:
            break

    run_elapsed = perf_counter() - run_start

    payload = build_export_payload(
        workspace_root=workspace_root,
        root_dir=args.root_dir,
        yolo_config=args.yolo_config,
        predict_input_root=args.predict_input_root,
        output_args={
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
            "allow_x_reverse": args.allow_x_reverse,
            "allow_y_reverse": args.allow_y_reverse,
            "trace_batches": args.trace_batches,
            "save_predict_input": args.save_predict_input,
            "x_scale": args.x_scale,
            "y_scale": args.y_scale,
            "threshold": args.threshold,
            "x_dilate": args.x_dilate,
            "y_dilate": args.y_dilate,
            "min_width": args.min_width,
            "max_width": args.max_width,
            "min_height": args.min_height,
            "max_height": args.max_height,
            "min_aspect": args.min_aspect,
            "max_aspect": args.max_aspect,
            "min_area": args.min_area,
            "margin": args.margin,
            "mech_delta_x": args.mech_delta_x,
            "mech_delta_y": args.mech_delta_y,
            "delta_x_pixel": args.delta_x_pixel,
            "delta_y_pixel": args.delta_y_pixel,
        },
        total_batches=total_batches,
        total_samples_seen=total_samples_seen,
        total_samples_inferred=total_samples_inferred,
        total_samples_skipped_no_chip=total_samples_skipped_no_chip,
        total_samples_skipped_error=total_samples_skipped_error,
        elapsed_s=run_elapsed,
        samples=exported_samples,
        skipped_errors=exported_skipped_errors,
    )
    write_export_json(args.output_json, payload)

    throughput = total_samples_inferred / run_elapsed if run_elapsed > 0 else 0.0
    print("\n===== Run Summary =====")
    print(f"batches_processed={total_batches}")
    print(f"samples_seen={total_samples_seen}")
    print(f"samples_inferred={total_samples_inferred}")
    print(f"samples_skipped_no_chip={total_samples_skipped_no_chip}")
    print(f"samples_skipped_error={total_samples_skipped_error}")
    print(f"elapsed_s={run_elapsed:.3f}")
    print(f"inferred_samples_per_s={throughput:.2f}")
    print(f"export_json={safe_relpath(args.output_json, workspace_root)}")
    return inferer


if __name__ == "__main__":
    main()

"""
Example:
python Detectors/rect_detector/run_main_inferer_dataloader.py 48AMA/imgs/S26F20082-02 \
    --batch-size 4 --num-workers 4 --persistent-workers --prefetch-factor 2 --light-read-workers 4
"""
