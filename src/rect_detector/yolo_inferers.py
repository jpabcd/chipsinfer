from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rect_detector.align_chip_rects import AlignResult, AlignedRect

try:
    from ultralytics import YOLO
except ImportError as exc:  # The module can still be tested with injected models.
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_CROP_WIDTH = 795
DEFAULT_CROP_HEIGHT = 161
DEFAULT_BROKEN_SUPPORTED_LIGHTS = ("light_1", "light_3")
DEFAULT_NOLINE_SUPPORTED_LIGHTS = ("light_2",)
_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, dict[int, str]]] = {}


@dataclass(frozen=True)
class YoloInfererConfig:
    light_name: str
    weight: str
    broken_supported_lights: tuple[str, ...] = DEFAULT_BROKEN_SUPPORTED_LIGHTS
    noline_supported_lights: tuple[str, ...] = DEFAULT_NOLINE_SUPPORTED_LIGHTS
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 320
    device: str | int = "0"
    quantize: int | None = 16
    use_engine: bool = True
    yolo_batch: int = 256
    yolo_stream: bool = True
    yolo_stream_batch: int = 128
    yolo_warmup: int = 100
    area_ratio_threshold: float = 0.02
    crop_width: int = DEFAULT_CROP_WIDTH
    crop_height: int = DEFAULT_CROP_HEIGHT
    draw: bool = False
    save_predict_input: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.light_name:
            raise ValueError("light_name cannot be empty")
        if not self.weight:
            raise ValueError(f"weight cannot be empty for {self.light_name}")
        if self.crop_width <= 0 or self.crop_height <= 0:
            raise ValueError("crop_width and crop_height must be positive")
        if self.imgsz <= 0 or self.yolo_batch <= 0 or self.yolo_stream_batch <= 0:
            raise ValueError("imgsz and YOLO batch sizes must be positive")

    @property
    def active_batch(self) -> int:
        return self.yolo_stream_batch if self.yolo_stream else self.yolo_batch


@dataclass(frozen=True)
class CombinedYoloInferersConfig:
    lights: tuple[YoloInfererConfig, ...]

    def __post_init__(self) -> None:
        if len(self.lights) != 4:
            raise ValueError(f"CombinedYoloInferers requires exactly four lights, got {len(self.lights)}")
        names = [config.light_name for config in self.lights]
        if len(set(names)) != len(names):
            raise ValueError(f"light_name values must be unique: {names}")
    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CombinedYoloInferersConfig":
        shared = {
            "broken_supported_lights": tuple(
                str(value) for value in config.get("broken_supported_lights", DEFAULT_BROKEN_SUPPORTED_LIGHTS)
            ),
            "noline_supported_lights": tuple(
                str(value) for value in config.get("noline_supported_lights", DEFAULT_NOLINE_SUPPORTED_LIGHTS)
            ),
            "conf": float(config.get("conf", 0.25)),
            "iou": float(config.get("iou", 0.45)),
            "imgsz": int(config.get("imgsz", 320)),
            "device": config.get("device", "0"),
            "quantize": config.get("quantize", 16),
            "use_engine": bool(config.get("use_engine", True)),
            "yolo_batch": int(config.get("yolo_batch", 256)),
            "yolo_stream": bool(config.get("yolo_stream", True)),
            "yolo_stream_batch": int(config.get("yolo_stream_batch", 128)),
            "yolo_warmup": int(config.get("yolo_warmup", 100)),
            "area_ratio_threshold": float(config.get("area_ratio_threshold", 0.02)),
            "crop_width": int(config.get("crop_width", DEFAULT_CROP_WIDTH)),
            "crop_height": int(config.get("crop_height", DEFAULT_CROP_HEIGHT)),
            "draw": bool(config.get("draw", False)),
            "save_predict_input": bool(config.get("save_predict_input", False)),
            "verbose": bool(config.get("verbose", False)),
        }
        raw_lights = config.get("lights")
        if not isinstance(raw_lights, Sequence) or isinstance(raw_lights, (str, bytes)):
            raise ValueError("config 'lights' must be a list of four {name, weight} objects")

        lights: list[YoloInfererConfig] = []
        for index, light in enumerate(raw_lights):
            if not isinstance(light, Mapping):
                raise ValueError(f"lights[{index}] must be an object")
            lights.append(
                YoloInfererConfig(
                    light_name=str(light.get("name", "")).strip(),
                    weight=str(light.get("weight", "")).strip(),
                    **shared,
                )
            )
        return cls(lights=tuple(lights))

    @classmethod
    def from_json(cls, path: str | Path) -> "CombinedYoloInferersConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_mapping(json.load(file))


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class YoloDetection:
    box: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    area_ratio: float


@dataclass(frozen=True)
class YoloPrediction:
    light_name: str
    aligned_rect: AlignedRect
    crop_box: CropBox
    pred_status: str
    pred_class: str
    decision_reason: str
    detections: tuple[YoloDetection, ...]
    best_class: str = ""
    best_conf: float = 0.0
    max_area_class: str = ""
    max_area_conf: float = 0.0
    max_area_ratio: float = 0.0
    has_broken: bool = False
    broken_conf: float = 0.0
    yolo_ms: float = 0.0

    @property
    def rect_id(self) -> int:
        return self.aligned_rect.rect.id

    @property
    def mechanical_index(self) -> int:
        return self.aligned_rect.mech.nIndex

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rect_id"] = self.rect_id
        data["mechanical_index"] = self.mechanical_index
        return data


@dataclass(frozen=True)
class SkippedCrop:
    light_name: str
    aligned_rect: AlignedRect
    crop_box: CropBox
    reason: str


@dataclass
class LightYoloResult:
    light_name: str
    predictions: list[YoloPrediction] = field(default_factory=list)
    skipped: list[SkippedCrop] = field(default_factory=list)
    drawn_crops_by_rect_id: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def predictions_by_rect_id(self) -> dict[int, YoloPrediction]:
        return {prediction.rect_id: prediction for prediction in self.predictions}


@dataclass
class CombinedChipPrediction:
    aligned_rect: AlignedRect
    light_predictions: dict[str, YoloPrediction | None]


@dataclass
class CombinedYoloResult:
    per_light: dict[str, LightYoloResult]
    per_chip: dict[int, CombinedChipPrediction]


def _normalize_model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        if image.dtype == np.uint16:
            return (image / 256).clip(0, 255).astype(np.uint8)
        info = np.iinfo(image.dtype)
        if info.max <= 255 and info.min >= 0:
            return image.astype(np.uint8)
        scaled = (image.astype(np.float32) - info.min) * (255.0 / (info.max - info.min))
        return scaled.clip(0, 255).astype(np.uint8)

    values = image.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    minimum = float(values[finite].min())
    maximum = float(values[finite].max())
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((values - minimum) * (255.0 / (maximum - minimum))).clip(0, 255).astype(np.uint8)


def _to_yolo_image(image: np.ndarray) -> np.ndarray:
    image = _to_uint8(np.asarray(image))
    if image.ndim == 2:
        return np.ascontiguousarray(np.repeat(image[:, :, None], 3, axis=2))
    if image.ndim == 3 and image.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(image, 3, axis=2))
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)
    raise ValueError(f"unsupported image shape for YOLO: {image.shape}")


def _engine_path(config: YoloInfererConfig) -> Path:
    model_path = Path(config.weight)
    quantize_tag = str(config.quantize).strip().lower().replace(" ", "_") or "none"
    return model_path.with_name(
        f"{model_path.stem}_q{quantize_tag}_b{config.active_batch}_imgsz{config.imgsz}.engine"
    )


def _load_model(config: YoloInfererConfig) -> tuple[Any, dict[int, str]]:
    if YOLO is None:
        raise RuntimeError("ultralytics is required unless a model is injected into YoloInferer") from _YOLO_IMPORT_ERROR
    model_path = Path(config.weight)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model file does not exist: {model_path}")

    load_path = model_path
    if config.use_engine:
        try:
            load_path = _engine_path(config)
            if not load_path.exists():
                source_model = YOLO(str(model_path), task="detect")
                exported = source_model.export(
                    format="engine",
                    dynamic=True,
                    batch=config.active_batch,
                    quantize=config.quantize,
                    imgsz=config.imgsz,
                    device=config.device,
                    nms=True,
                )
                exported_path = Path(str(exported))
                if exported_path.exists() and exported_path.resolve() != load_path.resolve():
                    os.replace(exported_path, load_path)
            if not load_path.exists():
                raise RuntimeError(f"TensorRT export did not create the expected engine: {load_path}")
        except Exception as exc:
            print(
                f"Warning: TensorRT setup failed for {config.light_name}; "
                f"falling back to {model_path}. Error: {exc}"
            )
            load_path = model_path

    model = YOLO(str(load_path), task="detect")
    return model, _normalize_model_names(model)


class YoloInferer:
    """Run one light-specific YOLO model on fixed-size crops from an AlignResult."""

    def __init__(
        self,
        config: YoloInfererConfig,
        model: Any | None = None,
        model_names: Mapping[int, str] | None = None,
    ) -> None:
        self.config = config
        if model is None:
            cache_key = (
                config.weight,
                config.use_engine,
                config.imgsz,
                str(config.device),
                config.quantize,
                config.active_batch,
            )
            if cache_key not in _MODEL_CACHE:
                _MODEL_CACHE[cache_key] = _load_model(config)
                self._warmup(_MODEL_CACHE[cache_key][0])
            model, cached_names = _MODEL_CACHE[cache_key]
            model_names = model_names or cached_names
        self.model = model
        self.model_names = dict(model_names or _normalize_model_names(model))
        self.trace_batching = False
        self.draw = bool(config.draw)
        self.draw_dir = Path("temp")
        self.save_predict_input = bool(config.save_predict_input)
        self.predict_input_dir = Path("temp") / "predict_input"
        if self.draw:
            self._ensure_draw_dir()
        if self.save_predict_input:
            self._ensure_predict_input_dir()
        inverse_names = {name: class_id for class_id, name in self.model_names.items()}
        self.has_broken_logic = config.light_name in config.broken_supported_lights
        self.broken_class_id = inverse_names.get("broken") if self.has_broken_logic else None
        self.is_noline_supported = config.light_name in config.noline_supported_lights

    def _ensure_draw_dir(self) -> None:
        self.draw_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_predict_input_dir(self) -> None:
        self.predict_input_dir.mkdir(parents=True, exist_ok=True)

    def set_predict_input_dir(self, directory: str | Path) -> None:
        self.predict_input_dir = Path(directory)
        if self.save_predict_input:
            self._ensure_predict_input_dir()

    def _warmup(self, model: Any) -> None:
        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        for _ in range(max(0, self.config.yolo_warmup)):
            self._predict(model=model, images=[dummy], batch=1, stream=False)

    def _predict(
        self,
        model: Any,
        images: Sequence[np.ndarray],
        batch: int,
        stream: bool,
    ) -> Iterable[Any]:
        kwargs = {
            "source": images,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "device": self.config.device,
            "save": False,
            "stream": stream,
            "verbose": self.config.verbose,
            "batch": batch,
            'rect': True
        }
        if self.config.quantize is not None:
            kwargs["quantize"] = self.config.quantize
        return model.predict(**kwargs)

    def _crop_box(self, aligned_rect: AlignedRect) -> CropBox:
        box = aligned_rect.rect.box
        center_x = box.x + box.w / 2.0
        center_y = box.y + box.h / 2.0
        x1 = int(round(center_x - self.config.crop_width / 2.0))
        y1 = int(round(center_y - self.config.crop_height / 2.0))
        return CropBox(x1=x1, y1=y1, x2=x1 + self.config.crop_width, y2=y1 + self.config.crop_height)

    def _prepare_crops(
        self,
        input_img: np.ndarray,
        align_result: AlignResult,
    ) -> tuple[list[np.ndarray], list[AlignedRect], list[CropBox], list[SkippedCrop]]:
        image = np.asarray(input_img)
        if image.ndim not in (2, 3):
            raise ValueError(f"input_img must be HxW or HxWxC, got {image.shape}")
        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        matches: list[AlignedRect] = []
        crop_boxes: list[CropBox] = []
        skipped: list[SkippedCrop] = []

        for aligned_rect in sorted(align_result.matches, key=lambda match: match.rect.id):
            crop_box = self._crop_box(aligned_rect)
            if crop_box.x1 < 0 or crop_box.y1 < 0 or crop_box.x2 > width or crop_box.y2 > height:
                skipped.append(
                    SkippedCrop(
                        light_name=self.config.light_name,
                        aligned_rect=aligned_rect,
                        crop_box=crop_box,
                        reason="crop_out_of_bounds",
                    )
                )
                continue
            crop = image[crop_box.y1 : crop_box.y2, crop_box.x1 : crop_box.x2]
            crops.append(_to_yolo_image(crop))
            matches.append(aligned_rect)
            crop_boxes.append(crop_box)
        return crops, matches, crop_boxes, skipped

    def _analyse_result(self, result: Any, aligned_rect: AlignedRect, crop_box: CropBox) -> YoloPrediction:
        boxes = getattr(result, "boxes", None)
        detections: list[YoloDetection] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = _tensor_to_numpy(boxes.xyxy, np.float32).reshape(-1, 4)
            class_ids = _tensor_to_numpy(boxes.cls, np.int32).reshape(-1)
            confidences = _tensor_to_numpy(boxes.conf, np.float32).reshape(-1)
            crop_area = max(crop_box.width * crop_box.height, 1)
            for box, class_id, confidence in zip(xyxy, class_ids, confidences):
                width = max(0.0, float(box[2] - box[0]))
                height = max(0.0, float(box[3] - box[1]))
                class_name = self.model_names.get(int(class_id), str(int(class_id)))
                if class_name == "noline" and not self.is_noline_supported:
                    continue
                detections.append(
                    YoloDetection(
                        box=tuple(float(value) for value in box),
                        class_id=int(class_id),
                        class_name=class_name,
                        confidence=float(confidence),
                        area_ratio=width * height / crop_area,
                    )
                )

        best = max(detections, key=lambda item: item.confidence, default=None)
        largest = max(detections, key=lambda item: item.area_ratio, default=None)
        broken = [item for item in detections if item.class_id == self.broken_class_id]
        has_broken = bool(broken)
        if not detections:
            status, pred_class, reason = "OK", "OK", "NoBox"
        elif has_broken:
            status, pred_class, reason = "NG", "broken", "Broken_Class"
        elif largest is not None and largest.area_ratio >= self.config.area_ratio_threshold:
            status, pred_class, reason = "NG", largest.class_name, "Area"
        else:
            status, pred_class, reason = "OK", "OK", "AreaBelowThreshold"

        speed = getattr(result, "speed", {}) or {}
        yolo_ms = sum(float(speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess"))
        return YoloPrediction(
            light_name=self.config.light_name,
            aligned_rect=aligned_rect,
            crop_box=crop_box,
            pred_status=status,
            pred_class=pred_class,
            decision_reason=reason,
            detections=tuple(detections),
            best_class=best.class_name if best else "",
            best_conf=best.confidence if best else 0.0,
            max_area_class=largest.class_name if largest else "",
            max_area_conf=largest.confidence if largest else 0.0,
            max_area_ratio=largest.area_ratio if largest else 0.0,
            has_broken=has_broken,
            broken_conf=max((item.confidence for item in broken), default=0.0),
            yolo_ms=yolo_ms,
        )

    def _draw_prediction(self, crop_img: np.ndarray, result: Any, prediction: YoloPrediction) -> np.ndarray:
        canvas = _to_yolo_image(crop_img).copy()
        color = (0, 0, 255) if prediction.pred_status == "NG" else (0, 255, 0)

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = _tensor_to_numpy(boxes.xyxy, np.float32).reshape(-1, 4)
            class_ids = _tensor_to_numpy(boxes.cls, np.int32).reshape(-1)
            confidences = _tensor_to_numpy(boxes.conf, np.float32).reshape(-1)
            height, width = canvas.shape[:2]
            for box, class_id, confidence in zip(xyxy, class_ids, confidences):
                x1 = int(np.clip(round(float(box[0])), 0, width - 1))
                y1 = int(np.clip(round(float(box[1])), 0, height - 1))
                x2 = int(np.clip(round(float(box[2])), 0, width - 1))
                y2 = int(np.clip(round(float(box[3])), 0, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                if cv2 is not None:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.model_names.get(int(class_id), str(int(class_id)))}:{float(confidence):.2f}"
                    cv2.putText(
                        canvas,
                        label,
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    canvas[y1:y1 + 2, x1:x2 + 1] = color
                    canvas[max(0, y2 - 1):y2 + 1, x1:x2 + 1] = color
                    canvas[y1:y2 + 1, x1:x1 + 2] = color
                    canvas[y1:y2 + 1, max(0, x2 - 1):x2 + 1] = color
        return canvas

    def _draw_output_path(self, prediction: YoloPrediction) -> Path:
        mech = prediction.aligned_rect.mech
        mx_text = f"{float(mech.MX):.3f}".replace(".", "p")
        my_text = f"{float(mech.MY):.3f}".replace(".", "p")
        file_name = (
            f"{prediction.light_name}_MX{mx_text}_MY{my_text}_"
            f"img{mech.nImageNum}_idx{mech.nIndex}_rect{prediction.rect_id}.png"
        )
        return self.draw_dir / file_name

    def _save_drawn_crop(self, prediction: YoloPrediction, drawn_crop: np.ndarray) -> None:
        self._ensure_draw_dir()
        out_path = self._draw_output_path(prediction)
        if cv2 is not None:
            cv2.imwrite(str(out_path), cv2.cvtColor(drawn_crop, cv2.COLOR_RGB2BGR))
            return
        if Image is not None:
            Image.fromarray(drawn_crop).save(out_path)
            return
        raise RuntimeError("draw=True requires either cv2 or Pillow to save images")

    def _predict_input_output_path(self, prediction: YoloPrediction) -> Path:
        mech = prediction.aligned_rect.mech
        mx_text = f"{float(mech.MX):.3f}".replace(".", "p")
        my_text = f"{float(mech.MY):.3f}".replace(".", "p")
        file_name = (
            f"{prediction.light_name}_MX{mx_text}_MY{my_text}_"
            f"img{mech.nImageNum}_idx{mech.nIndex}_rect{prediction.rect_id}.png"
        )
        return self.predict_input_dir / file_name

    def _save_predict_input(self, prediction: YoloPrediction, crop_img: np.ndarray) -> None:
        self._ensure_predict_input_dir()
        out_path = self._predict_input_output_path(prediction)
        if cv2 is not None:
            cv2.imwrite(str(out_path), cv2.cvtColor(_to_yolo_image(crop_img), cv2.COLOR_RGB2BGR))
            return
        if Image is not None:
            Image.fromarray(_to_yolo_image(crop_img)).save(out_path)
            return
        raise RuntimeError("save_predict_input=True requires either cv2 or Pillow to save images")

    def __call__(self, input_img: np.ndarray, align_result: AlignResult) -> LightYoloResult:
        return self.batch_infer([input_img], [align_result])[0]

    def batch_infer(
        self,
        input_imgs: Sequence[np.ndarray],
        align_results: Sequence[AlignResult],
    ) -> list[LightYoloResult]:
        if len(input_imgs) != len(align_results):
            raise ValueError(
                f"{self.config.light_name}: input_imgs count ({len(input_imgs)}) "
                f"!= align_results count ({len(align_results)})"
            )

        light_results: list[LightYoloResult] = []
        flat_crops: list[np.ndarray] = []
        flat_meta: list[tuple[int, AlignedRect, CropBox]] = []
        per_sample_crop_counts: list[int] = []

        for batch_index, (input_img, align_result) in enumerate(zip(input_imgs, align_results)):
            crops, matches, crop_boxes, skipped = self._prepare_crops(input_img, align_result)
            light_results.append(LightYoloResult(light_name=self.config.light_name, skipped=skipped))
            per_sample_crop_counts.append(len(crops))
            flat_crops.extend(crops)
            flat_meta.extend(
                (batch_index, aligned_rect, crop_box)
                for aligned_rect, crop_box in zip(matches, crop_boxes)
            )

        if self.trace_batching:
            total_crops = len(flat_crops)
            active_batch = self.config.active_batch
            predict_calls = (total_crops + active_batch - 1) // active_batch if total_crops else 0
            print(
                f"YOLO batch plan light={self.config.light_name} samples={len(input_imgs)} "
                f"per_sample_crops={per_sample_crop_counts} total_crops={total_crops} "
                f"active_batch={active_batch} predict_calls={predict_calls}"
            )

        if not flat_crops:
            return light_results

        active_batch = self.config.active_batch
        for start in range(0, len(flat_crops), active_batch):
            batch_crops = flat_crops[start : start + active_batch]
            batch_meta = flat_meta[start : start + active_batch]
            if self.trace_batching:
                sample_indices = sorted({batch_index for batch_index, _, _ in batch_meta})
                print(
                    f"YOLO batch execute light={self.config.light_name} start={start} "
                    f"crop_count={len(batch_crops)} sample_indices={sample_indices}"
                )
            raw_results = self._predict(
                model=self.model,
                images=batch_crops,
                batch=min(active_batch, len(batch_crops)),
                stream=self.config.yolo_stream,
            )
            batch_results = list(raw_results)
            if len(batch_results) != len(batch_crops):
                raise RuntimeError(
                    f"{self.config.light_name}: YOLO returned {len(batch_results)} results for {len(batch_crops)} crops"
                )
            for result, crop_img, (batch_index, aligned_rect, crop_box) in zip(
                batch_results,
                batch_crops,
                batch_meta,
            ):
                prediction = self._analyse_result(result, aligned_rect, crop_box)
                light_results[batch_index].predictions.append(prediction)
                if self.save_predict_input:
                    self._save_predict_input(prediction, crop_img)
                if self.draw:
                    drawn_crop = self._draw_prediction(
                        crop_img,
                        result,
                        prediction,
                    )
                    light_results[batch_index].drawn_crops_by_rect_id[prediction.rect_id] = drawn_crop
                    self._save_drawn_crop(prediction, drawn_crop)
        return light_results


class CombinedYoloInferers:
    """Apply one AlignResult to four registered light images and models."""

    def __init__(
        self,
        config: CombinedYoloInferersConfig | Mapping[str, Any] | str | Path,
        models: Mapping[str, Any] | None = None,
        model_names: Mapping[str, Mapping[int, str]] | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            resolved_config = CombinedYoloInferersConfig.from_json(config)
        elif isinstance(config, CombinedYoloInferersConfig):
            resolved_config = config
        else:
            resolved_config = CombinedYoloInferersConfig.from_mapping(config)
        self.config = resolved_config
        models = models or {}
        model_names = model_names or {}
        self.inferers = {
            light.light_name: YoloInferer(
                config=light,
                model=models.get(light.light_name),
                model_names=model_names.get(light.light_name),
            )
            for light in resolved_config.lights
        }

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "CombinedYoloInferers":
        return cls(config=CombinedYoloInferersConfig.from_json(path), **kwargs)

    def set_trace_batching(self, enabled: bool) -> None:
        for inferer in self.inferers.values():
            inferer.trace_batching = enabled

    def set_draw(self, enabled: bool) -> None:
        for inferer in self.inferers.values():
            inferer.draw = enabled
            if enabled:
                inferer._ensure_draw_dir()

    def set_save_predict_input(self, enabled: bool) -> None:
        for inferer in self.inferers.values():
            inferer.save_predict_input = enabled
            if enabled:
                inferer._ensure_predict_input_dir()

    def set_predict_input_dir(self, directory: str | Path) -> None:
        for inferer in self.inferers.values():
            inferer.set_predict_input_dir(directory)

    def __call__(
        self,
        light_images: Mapping[str, np.ndarray],
        align_result: AlignResult,
    ) -> CombinedYoloResult:
        return self.batch_infer([light_images], [align_result])[0]

    def batch_infer(
        self,
        light_image_batches: Sequence[Mapping[str, np.ndarray]],
        align_results: Sequence[AlignResult],
    ) -> list[CombinedYoloResult]:
        if len(light_image_batches) != len(align_results):
            raise ValueError(
                f"light_image_batches count ({len(light_image_batches)}) "
                f"!= align_results count ({len(align_results)})"
            )

        for light_images in light_image_batches:
            self._validate_light_images(light_images)

        per_light_batches = {
            name: self.inferers[name].batch_infer(
                [light_images[name] for light_images in light_image_batches],
                align_results,
            )
            for name in self.inferers
        }
        combined_results: list[CombinedYoloResult] = []
        for batch_index, align_result in enumerate(align_results):
            per_light = {
                name: per_light_batches[name][batch_index]
                for name in self.inferers
            }
            combined_results.append(self._combine_per_light_results(per_light, align_result))
        return combined_results

    def _validate_light_images(self, light_images: Mapping[str, np.ndarray]) -> None:
        expected_names = set(self.inferers)
        actual_names = set(light_images)
        if actual_names != expected_names:
            raise ValueError(
                f"light_images keys must be {sorted(expected_names)}, got {sorted(actual_names)}"
            )

        image_shapes = {name: np.asarray(light_images[name]).shape[:2] for name in expected_names}
        if len(set(image_shapes.values())) != 1:
            raise ValueError(f"all four light images must have identical HxW shapes: {image_shapes}")

    def _combine_per_light_results(
        self,
        per_light: Mapping[str, LightYoloResult],
        align_result: AlignResult,
    ) -> CombinedYoloResult:
        predictions_by_light = {
            name: result.predictions_by_rect_id
            for name, result in per_light.items()
        }
        per_chip: dict[int, CombinedChipPrediction] = {}
        for aligned_rect in sorted(align_result.matches, key=lambda match: match.rect.id):
            mechanical_index = aligned_rect.mech.nIndex
            if mechanical_index in per_chip:
                raise ValueError(f"duplicate mechanical nIndex in AlignResult: {mechanical_index}")
            per_chip[mechanical_index] = CombinedChipPrediction(
                aligned_rect=aligned_rect,
                light_predictions={
                    name: predictions_by_light[name].get(aligned_rect.rect.id)
                    for name in self.inferers
                },
            )
        return CombinedYoloResult(per_light=per_light, per_chip=per_chip)


def _tensor_to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


# Compatibility aliases matching the requested component names.
yolo_inferer = YoloInferer
combined_yolo_inferers = CombinedYoloInferers


__all__ = [
    "CombinedChipPrediction",
    "CombinedYoloInferers",
    "CombinedYoloInferersConfig",
    "CombinedYoloResult",
    "CropBox",
    "LightYoloResult",
    "SkippedCrop",
    "YoloDetection",
    "YoloInferer",
    "YoloInfererConfig",
    "YoloPrediction",
    "combined_yolo_inferers",
    "yolo_inferer",
]
