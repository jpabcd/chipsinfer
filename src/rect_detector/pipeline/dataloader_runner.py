from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rect_detector.main_inferer import MainInferResult, MainInferer


def is_no_chip_sample(rect_input_img: np.ndarray, mechanical_infos: Sequence[Any]) -> bool:
    return rect_input_img.size == 0 or len(mechanical_infos) == 0


def prepare_light_batches(light_image_batches: Sequence[Mapping[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
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


def run_batch_with_fallback(
    inferer: MainInferer,
    rect_input_imgs: Sequence[np.ndarray],
    mechanical_infos: Sequence[Sequence[Any]],
    light_image_batches: Sequence[Mapping[str, np.ndarray]],
    sample_ids: Sequence[int],
    num_strs: Sequence[str],
) -> tuple[list[tuple[int, MainInferResult]], list[tuple[int, str, str]]]:
    """Run batch inference and fallback to per-sample mode if batch fails."""
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


def print_sample_result(result: MainInferResult, sample_id: int, num_str: str) -> None:
    summary = result.summary
    print(
        f"  sample_id={sample_id} num_str={num_str} "
        f"aligned={summary.aligned_count} OK={summary.ok_count} NG={summary.ng_count} "
        f"INVALID={summary.invalid_count} total_s={result.timings.total_s:.3f}"
    )
