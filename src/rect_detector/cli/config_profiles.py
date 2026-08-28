from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_PRODUCT_SECTIONS = {"name", "yolo_config", "raw_image", "rect_detection", "alignment", "wafer_map"}
_RUNTIME_SECTIONS = {"paths", "line", "inference", "outputs"}

_PRODUCT_OPTION_MAP = {
    "x_scale": "--x-scale",
    "y_scale": "--y-scale",
    "threshold": "--threshold",
    "x_dilate": "--x-dilate",
    "y_dilate": "--y-dilate",
    "min_width": "--min-width",
    "max_width": "--max-width",
    "min_height": "--min-height",
    "max_height": "--max-height",
    "min_aspect": "--min-aspect",
    "max_aspect": "--max-aspect",
    "min_area": "--min-area",
    "margin": "--margin",
    "mech_delta_x": "--mech-delta-x",
    "mech_delta_y": "--mech-delta-y",
    "delta_x_pixel": "--delta-x-pixel",
    "delta_y_pixel": "--delta-y-pixel",
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_json_config(path: Path, project_root: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved_path = path if path.is_absolute() else (project_root / path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved_path}")
    try:
        data = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} {resolved_path}: {exc}") from exc
    return resolved_path, _require_mapping(data, label)


def validate_product_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - _PRODUCT_SECTIONS
    if unknown:
        raise ValueError(f"Unknown product config fields: {sorted(unknown)}")
    for section in ("raw_image", "rect_detection", "alignment", "wafer_map"):
        if section in config:
            _require_mapping(config[section], f"product config '{section}'")

    raw_image = config.get("raw_image", {})
    rect = config.get("rect_detection", {})
    alignment = config.get("alignment", {})
    allowed_rect = set(_PRODUCT_OPTION_MAP) - {
        "mech_delta_x", "mech_delta_y", "delta_x_pixel", "delta_y_pixel"
    }
    allowed_alignment = {
        "mech_delta_x", "mech_delta_y", "delta_x_pixel", "delta_y_pixel",
        "strict_align", "allow_x_reverse", "allow_y_reverse",
    }
    unknown_rect = set(rect) - allowed_rect
    unknown_alignment = set(alignment) - allowed_alignment
    unknown_raw_image = set(raw_image) - {"width", "height", "dtype", "byte_order"}
    unknown_wafer = set(config.get("wafer_map", {})) - {"chip_aspect", "figsize"}
    if unknown_raw_image:
        raise ValueError(f"Unknown raw_image fields: {sorted(unknown_raw_image)}")
    if unknown_rect:
        raise ValueError(f"Unknown rect_detection fields: {sorted(unknown_rect)}")
    if unknown_alignment:
        raise ValueError(f"Unknown alignment fields: {sorted(unknown_alignment)}")
    if unknown_wafer:
        raise ValueError(f"Unknown wafer_map fields: {sorted(unknown_wafer)}")
    for field in ("width", "height"):
        if field in raw_image and (
            isinstance(raw_image[field], bool)
            or not isinstance(raw_image[field], int)
            or raw_image[field] <= 0
        ):
            raise ValueError(f"product raw_image.{field} must be a positive integer")
    if "dtype" in raw_image and raw_image["dtype"] not in {"auto", "uint8", "uint16"}:
        raise ValueError("product raw_image.dtype must be 'auto', 'uint8', or 'uint16'")
    if "byte_order" in raw_image and raw_image["byte_order"] not in {"little", "big"}:
        raise ValueError("product raw_image.byte_order must be 'little' or 'big'")


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - _RUNTIME_SECTIONS
    if unknown:
        raise ValueError(f"Unknown runtime config fields: {sorted(unknown)}")
    allowed = {
        "paths": {"imgs_root", "output_root", "summary_json"},
        "line": {
            "order_by", "reverse", "max_products", "skip_incomplete",
            "skip_completed", "fail_fast", "watch", "rescan_interval", "dry_run",
        },
        "inference": {
            "batch_size", "num_workers", "persistent_workers", "prefetch_factor",
            "light_read_workers", "max_batches", "max_samples", "unsafe_missing_ok",
            "trace_batches",
        },
        "outputs": {
            "defect_report",
            "auto_external_xy_csv",
            "save_predict_input",
            "save_predict_input_only_with_boxes",
            "save_predict_input_on_any_light_ng",
        },
    }
    for section, allowed_fields in allowed.items():
        values = _require_mapping(config.get(section, {}), f"runtime config '{section}'")
        unknown_fields = set(values) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown runtime {section} fields: {sorted(unknown_fields)}")


def product_config_args(config: Mapping[str, Any], project_root: Path) -> list[str]:
    validate_product_config(config)
    args: list[str] = []

    yolo_config = config.get("yolo_config")
    if yolo_config:
        yolo_path = Path(str(yolo_config))
        if not yolo_path.is_absolute():
            yolo_path = (project_root / yolo_path).resolve()
        args.extend(["--yolo-config", str(yolo_path)])

    raw_image = config.get("raw_image", {})
    for field, option in (
        ("width", "--raw-image-width"),
        ("height", "--raw-image-height"),
        ("dtype", "--raw-dtype"),
        ("byte_order", "--raw-byte-order"),
    ):
        if field in raw_image:
            args.extend([option, str(raw_image[field])])

    for section_name in ("rect_detection", "alignment"):
        section = config.get(section_name, {})
        for field, option in _PRODUCT_OPTION_MAP.items():
            if field in section:
                args.extend([option, str(section[field])])

    alignment = config.get("alignment", {})
    for field, option in (
        ("strict_align", "--strict-align"),
        ("allow_x_reverse", "--allow-x-reverse"),
        ("allow_y_reverse", "--allow-y-reverse"),
    ):
        if bool(alignment.get(field, False)):
            args.append(option)

    wafer_map = config.get("wafer_map", {})
    if "chip_aspect" in wafer_map:
        args.extend(["--wafer-map-chip-aspect", str(wafer_map["chip_aspect"])])
    if "figsize" in wafer_map:
        figsize = wafer_map["figsize"]
        if not isinstance(figsize, list) or len(figsize) != 2:
            raise ValueError("product wafer_map.figsize must be [width, height]")
        args.extend(["--wafer-map-figsize", str(figsize[0]), str(figsize[1])])
    return args


def runtime_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_runtime_config(config)
    paths = config.get("paths", {})
    line = config.get("line", {})
    inference = config.get("inference", {})
    outputs = config.get("outputs", {})
    return {
        "imgs_root": paths.get("imgs_root", "../../48AMA/imgs"),
        "output_root": paths.get("output_root"),
        "summary_json": paths.get("summary_json"),
        "order_by": line.get("order_by", "mtime"),
        "reverse": bool(line.get("reverse", False)),
        "max_products": int(line.get("max_products", 0)),
        "skip_incomplete": bool(line.get("skip_incomplete", True)),
        "skip_completed": bool(line.get("skip_completed", True)),
        "fail_fast": bool(line.get("fail_fast", False)),
        "watch": bool(line.get("watch", False)),
        "rescan_interval": float(line.get("rescan_interval", 30.0)),
        "dry_run": bool(line.get("dry_run", False)),
        "defect_report": bool(outputs.get("defect_report", True)),
        "auto_external_xy_csv": bool(outputs.get("auto_external_xy_csv", True)),
        "save_predict_input": bool(outputs.get("save_predict_input", False)),
        "save_predict_input_only_with_boxes": bool(
            outputs.get("save_predict_input_only_with_boxes", False)
        ),
        "save_predict_input_on_any_light_ng": bool(
            outputs.get("save_predict_input_on_any_light_ng", False)
        ),
        "inference": inference,
    }


def runtime_inference_args(config: Mapping[str, Any]) -> list[str]:
    inference = runtime_defaults(config)["inference"]
    args: list[str] = []
    scalar_options = {
        "batch_size": "--batch-size",
        "num_workers": "--num-workers",
        "prefetch_factor": "--prefetch-factor",
        "light_read_workers": "--light-read-workers",
        "max_batches": "--max-batches",
        "max_samples": "--max-samples",
    }
    for field, option in scalar_options.items():
        if field in inference:
            args.extend([option, str(inference[field])])
    boolean_options = {
        "persistent_workers": ("--persistent-workers", "--no-persistent-workers"),
        "unsafe_missing_ok": ("--unsafe-missing-ok", None),
        "trace_batches": ("--trace-batches", None),
    }
    for field, (true_option, false_option) in boolean_options.items():
        if field in inference:
            if bool(inference[field]):
                args.append(true_option)
            elif false_option:
                args.append(false_option)
    return args
