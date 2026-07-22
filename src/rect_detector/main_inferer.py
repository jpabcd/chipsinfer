from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from rect_detector.align_chip_rects import AlignResult, AlignedRect, MechanicalInfo, align_rects_to_mechanical_txt
from rect_detector.extract_chip_rects import RectInferer
from rect_detector.rect_models import Rect_Info
from rect_detector.yolo_inferers import (
    CombinedYoloInferers,
    CombinedYoloResult,
    CropBox,
    YoloPrediction,
)


ChipKey = tuple[int, int]


@dataclass(frozen=True)
class MainInferenceTimings:
    rect_detection_s: float
    alignment_s: float
    four_light_yolo_s: float
    result_build_s: float
    total_s: float


@dataclass(frozen=True)
class MainInferenceSummary:
    detected_rect_count: int
    mechanical_record_count: int
    aligned_count: int
    unmatched_rect_count: int
    unmatched_mechanical_count: int
    ok_count: int
    ng_count: int
    invalid_count: int
    alignment_rmse_um: float
    alignment_max_residual_um: float
    is_partial_alignment: bool


@dataclass(frozen=True)
class MainChipInfo:
    chip_key: ChipKey
    mechanical_info: MechanicalInfo
    aligned_rect: AlignedRect
    crop_box: CropBox | None
    light_predictions: dict[str, YoloPrediction | None]
    final_status: str
    final_class: str
    decision_reason: str
    trigger_light: str | None

    @property
    def rect_info(self) -> Rect_Info:
        return self.aligned_rect.rect

    @property
    def alignment_residual_um(self) -> float:
        return self.aligned_rect.residual_um


@dataclass
class MainInferResult:
    rects: list[Rect_Info]
    align_result: AlignResult
    yolo_result: CombinedYoloResult
    chips: dict[ChipKey, MainChipInfo]
    summary: MainInferenceSummary
    timings: MainInferenceTimings
    warnings: list[str] = field(default_factory=list)

    def chip(self, n_image_num: int, n_index: int) -> MainChipInfo:
        return self.chips[(n_image_num, n_index)]


class MainInferer:
    """Run rect detection, mechanical alignment, and four-light YOLO once."""

    def __init__(
        self,
        rect_inferer: RectInferer,
        yolo_inferers: CombinedYoloInferers,
        allow_partial: bool = True,
        allow_x_reverse: bool = False,
        allow_y_reverse: bool = False,
    ) -> None:
        self.rect_inferer = rect_inferer
        self.yolo_inferers = yolo_inferers
        self.allow_partial = allow_partial
        self.allow_x_reverse = allow_x_reverse
        self.allow_y_reverse = allow_y_reverse

    def __call__(
        self,
        rect_input_img: np.ndarray,
        mechanical_info: str | Path | MechanicalInfo | Sequence[MechanicalInfo],
        light_images: Mapping[str, np.ndarray],
    ) -> MainInferResult:
        return self.batch_infer(
            rect_input_imgs=[rect_input_img],
            mechanical_infos=[mechanical_info],
            light_image_batches=[light_images],
        )[0]

    def batch_infer(
        self,
        rect_input_imgs: Sequence[np.ndarray],
        mechanical_infos: Sequence[str | Path | MechanicalInfo | Sequence[MechanicalInfo]],
        light_image_batches: Sequence[Mapping[str, np.ndarray]],
    ) -> list[MainInferResult]:
        if len(rect_input_imgs) != len(mechanical_infos) or len(rect_input_imgs) != len(light_image_batches):
            raise ValueError(
                "rect_input_imgs, mechanical_infos, and light_image_batches must have the same length"
            )

        rects_list: list[list[Rect_Info]] = []
        align_results: list[AlignResult] = []
        rect_detection_times: list[float] = []
        alignment_times: list[float] = []

        for rect_input_img in rect_input_imgs:
            stage_start = perf_counter()
            rects = self.rect_inferer(rect_input_img)
            rect_detection_times.append(perf_counter() - stage_start)
            rects_list.append(rects)

        for rects, mechanical_info in zip(rects_list, mechanical_infos):
            stage_start = perf_counter()
            align_results.append(
                align_rects_to_mechanical_txt(
                    rects=rects,
                    mechanical_txt=mechanical_info,
                    mech_delta_x=self.rect_inferer.mech_delta_x,
                    mech_delta_y=self.rect_inferer.mech_delta_y,
                    delta_x_pixel=self.rect_inferer.delta_x_pixel,
                    delta_y_pixel=self.rect_inferer.delta_y_pixel,
                    allow_partial=self.allow_partial,
                    allow_x_reverse=self.allow_x_reverse,
                    allow_y_reverse=self.allow_y_reverse,
                )
            )
            alignment_times.append(perf_counter() - stage_start)

        stage_start = perf_counter()
        yolo_results = self.yolo_inferers.batch_infer(
            light_image_batches=light_image_batches,
            align_results=align_results,
        )
        four_light_yolo_s = perf_counter() - stage_start
        per_sample_yolo_times = self._allocate_yolo_time(four_light_yolo_s, yolo_results)

        results: list[MainInferResult] = []
        for rects, align_result, yolo_result, rect_detection_s, alignment_s, yolo_s in zip(
            rects_list,
            align_results,
            yolo_results,
            rect_detection_times,
            alignment_times,
            per_sample_yolo_times,
        ):
            stage_start = perf_counter()
            chips, warnings = self._build_chip_infos(
                align_result=align_result,
                yolo_result=yolo_result,
            )
            result_build_s = perf_counter() - stage_start
            warnings.extend(self._alignment_warnings(align_result))

            status_counts = {"OK": 0, "NG": 0, "INVALID": 0}
            for chip in chips.values():
                status_counts[chip.final_status] += 1

            summary = MainInferenceSummary(
                detected_rect_count=len(rects),
                mechanical_record_count=len(align_result.matches) + len(align_result.unmatched_mechs),
                aligned_count=len(align_result.matches),
                unmatched_rect_count=len(align_result.unmatched_rects),
                unmatched_mechanical_count=len(align_result.unmatched_mechs),
                ok_count=status_counts["OK"],
                ng_count=status_counts["NG"],
                invalid_count=status_counts["INVALID"] + len(align_result.unmatched_mechs),
                alignment_rmse_um=align_result.rmse_um,
                alignment_max_residual_um=align_result.max_residual_um,
                is_partial_alignment=align_result.is_partial,
            )
            total_s = rect_detection_s + alignment_s + yolo_s + result_build_s
            results.append(
                MainInferResult(
                    rects=rects,
                    align_result=align_result,
                    yolo_result=yolo_result,
                    chips=chips,
                    summary=summary,
                    timings=MainInferenceTimings(
                        rect_detection_s=rect_detection_s,
                        alignment_s=alignment_s,
                        four_light_yolo_s=yolo_s,
                        result_build_s=result_build_s,
                        total_s=total_s,
                    ),
                    warnings=warnings,
                )
            )
        return results

    def _build_chip_infos(
        self,
        align_result: AlignResult,
        yolo_result: CombinedYoloResult,
    ) -> tuple[dict[ChipKey, MainChipInfo], list[str]]:
        light_order = tuple(self.yolo_inferers.inferers)
        predictions_by_light = {
            light_name: light_result.predictions_by_rect_id
            for light_name, light_result in yolo_result.per_light.items()
        }
        chips: dict[ChipKey, MainChipInfo] = {}
        warnings: list[str] = []

        for aligned_rect in sorted(align_result.matches, key=lambda match: match.rect.id):
            mech = aligned_rect.mech
            chip_key = (mech.nImageNum, mech.nIndex)
            if chip_key in chips:
                raise ValueError(f"duplicate chip key in alignment result: {chip_key}")

            light_predictions = {
                light_name: predictions_by_light.get(light_name, {}).get(aligned_rect.rect.id)
                for light_name in light_order
            }
            crop_box, crop_consistent = self._shared_crop_box(light_predictions)
            final_status, final_class, reason, trigger_light = self._decide_final_result(
                light_predictions=light_predictions,
                crop_consistent=crop_consistent,
            )
            if final_status == "INVALID":
                missing_lights = [name for name, prediction in light_predictions.items() if prediction is None]
                warnings.append(
                    f"chip={chip_key} invalid: reason={reason}, missing_lights={missing_lights}"
                )

            chips[chip_key] = MainChipInfo(
                chip_key=chip_key,
                mechanical_info=mech,
                aligned_rect=aligned_rect,
                crop_box=crop_box,
                light_predictions=light_predictions,
                final_status=final_status,
                final_class=final_class,
                decision_reason=reason,
                trigger_light=trigger_light,
            )
        return chips, warnings

    @staticmethod
    def _shared_crop_box(
        light_predictions: Mapping[str, YoloPrediction | None],
    ) -> tuple[CropBox | None, bool]:
        boxes = [prediction.crop_box for prediction in light_predictions.values() if prediction is not None]
        if not boxes:
            return None, False
        first = boxes[0]
        return first, all(box == first for box in boxes[1:])

    @staticmethod
    def _decide_final_result(
        light_predictions: Mapping[str, YoloPrediction | None],
        crop_consistent: bool,
    ) -> tuple[str, str, str, str | None]:
        if any(prediction is None for prediction in light_predictions.values()):
            return "INVALID", "", "Missing_Light_Result", None
        if not crop_consistent:
            return "INVALID", "", "Inconsistent_Crop_Box", None

        predictions = [prediction for prediction in light_predictions.values() if prediction is not None]
        broken_predictions = [prediction for prediction in predictions if prediction.has_broken]
        if broken_predictions:
            trigger = max(broken_predictions, key=lambda prediction: prediction.broken_conf)
            return "NG", "broken", "Broken_Class", trigger.light_name

        ng_predictions = [prediction for prediction in predictions if prediction.pred_status == "NG"]
        if ng_predictions:
            trigger = max(ng_predictions, key=lambda prediction: prediction.max_area_ratio)
            return "NG", trigger.pred_class, trigger.decision_reason, trigger.light_name
        return "OK", "OK", "All_Lights_OK", None

    @staticmethod
    def _alignment_warnings(align_result: AlignResult) -> list[str]:
        warnings: list[str] = []
        if align_result.unmatched_rects:
            warnings.append(
                "unmatched rect ids: " + ",".join(str(rect.id) for rect in align_result.unmatched_rects)
            )
        if align_result.unmatched_mechs:
            warnings.append(
                "unmatched mechanical nIndex values: "
                + ",".join(str(mech.nIndex) for mech in align_result.unmatched_mechs)
            )
        return warnings

    @staticmethod
    def _allocate_yolo_time(
        total_yolo_s: float,
        yolo_results: Sequence[CombinedYoloResult],
    ) -> list[float]:
        weights = [
            sum(len(light_result.predictions) for light_result in result.per_light.values())
            for result in yolo_results
        ]
        total_weight = sum(weights)
        if total_weight <= 0:
            if not yolo_results:
                return []
            even_share = total_yolo_s / len(yolo_results)
            return [even_share for _ in yolo_results]
        return [total_yolo_s * weight / total_weight for weight in weights]


# Compatibility alias matching the requested class spelling.
Main_Inferer = MainInferer


__all__ = [
    "ChipKey",
    "MainChipInfo",
    "MainInferResult",
    "MainInferenceSummary",
    "MainInferenceTimings",
    "MainInferer",
    "Main_Inferer",
]
