import argparse
import glob
import hashlib
import json
import mimetypes
import os
import pickle
import re
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image

try:
    import orjson
except ImportError:
    orjson = None


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ANNOTATION_DIR = APP_DIR / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "annotations.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSIFICATION_VERDICT_MAP = {
    "OK": "分类正确",
    "NG": "分类错误",
    "分类正确": "分类正确",
    "分类错误": "分类错误",
}
INFER_MODEL_PREDICTION_MAP = {
    "OK": "合格品",
    "NG": "缺陷品",
    "INVALID": "未知",
}
DEFAULT_RESULT_JSON = APP_DIR.parent / "outputs" / "json" / "main_inferer_dataloader_results.json"
IMAGE_META_CACHE = {}
INFERENCE_ITEMS_CACHE = {}
INFERENCE_BASE_CACHE = {}
INFERENCE_CACHE_LOCK = threading.RLock()
PROJECT_DIR = APP_DIR.parent
CONFIGURED_RESULT_JSON = ""
JSON_OPTIONS_TXT = APP_DIR / "json_candidates.txt"
INFERENCE_DISK_CACHE_DIR = APP_DIR / ".inference_cache"
MODEL_PREDICTION_BY_PATH = {}


def clear_inference_runtime_cache(clear_disk_cache=False):
    IMAGE_META_CACHE.clear()
    MODEL_PREDICTION_BY_PATH.clear()
    with INFERENCE_CACHE_LOCK:
        INFERENCE_ITEMS_CACHE.clear()
        INFERENCE_BASE_CACHE.clear()

    removed_disk_files = 0
    if clear_disk_cache and INFERENCE_DISK_CACHE_DIR.exists():
        for pattern in ("*.pkl", "*.tmp"):
            for cache_file in INFERENCE_DISK_CACHE_DIR.glob(pattern):
                try:
                    cache_file.unlink()
                    removed_disk_files += 1
                except OSError:
                    continue
    return removed_disk_files


def normalize_path_text(value):
    return str(value).replace("\\", "/").rstrip("/")


def path_basename(value):
    return os.path.basename(normalize_path_text(value)).lower()


def annotation_lookup_key(value):
    parts = normalize_path_text(value).lower().split("/")
    for index, part in enumerate(parts):
        if re.fullmatch(r"light_\d+", part):
            # Identity starts at light_x and includes sample/chip directories,
            # so equal filenames in other lights or samples remain independent.
            return "/".join(parts[index:])
    return f"unknown::{path_basename(value)}"


def get_model_prediction(image_path):
    stem = Path(image_path).stem
    tokens = [token.upper() for token in stem.split("_")]
    if "OK" in tokens:
        return "合格品"
    if "NG" in tokens:
        return "缺陷品"
    return "未知"


def get_light_type(image_path):
    parts = [part.lower() for part in str(image_path).replace("\\", "/").split("/")]
    for part in parts:
        if re.fullmatch(r"light_\d+", part):
            return part

    stem_tokens = Path(image_path).stem.lower().split("_")
    for index, token in enumerate(stem_tokens[:-1]):
        if token == "light" and stem_tokens[index + 1].isdigit():
            return f"light_{stem_tokens[index + 1]}"
    return "unknown"


def resolve_result_path(result_json_path, image_path, project_dir=None):
    candidate = Path(str(image_path)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    # When the caller explicitly provides imgroot, path_ is defined relative
    # to that directory. Avoid probing all workspace ancestors for every item.
    if project_dir:
        return (Path(project_dir).expanduser() / str(image_path)).resolve()

    result_json_resolved = resolve_user_supplied_path(result_json_path)
    result_parent = result_json_resolved.parent

    roots = []
    if project_dir:
        roots.append(Path(project_dir).expanduser())
    roots.extend([
        Path.cwd(),
        APP_DIR,
        APP_DIR.parent,
        result_parent,
    ])

    # Try every ancestor of the result JSON directory, so paths like
    # outputs/run_xxx/... can be resolved from rect_detector or workspace root.
    for parent in result_parent.parents:
        roots.append(parent)

    seen = set()
    deduped_roots = []
    for root in roots:
        root_key = str(root.resolve())
        if root_key in seen:
            continue
        seen.add(root_key)
        deduped_roots.append(root)

    image_path_text = str(image_path)
    for root in deduped_roots:
        merged = (root / image_path_text).resolve()
        if merged.exists():
            return merged
    return (Path.cwd() / image_path_text).resolve()


def resolve_user_supplied_path(path_value):
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    roots = (
        Path.cwd(),
        APP_DIR,
        APP_DIR.parent,
        APP_DIR.parent.parent,
        APP_DIR.parent.parent.parent,
    )
    for root in roots:
        merged = (root / candidate).resolve()
        if merged.exists():
            return merged
    return (Path.cwd() / candidate).resolve()


def resolve_json_option_path(path_value, txt_file_path=None, project_dir=None):
    candidate = Path(str(path_value)).expanduser()
    roots = []

    if candidate.is_absolute():
        return candidate.resolve()

    if txt_file_path:
        roots.append(Path(txt_file_path).expanduser().resolve().parent)
    if project_dir:
        roots.append(Path(project_dir).expanduser().resolve())
    roots.extend([
        Path.cwd(),
        APP_DIR,
        APP_DIR.parent,
    ])

    seen = set()
    for root in roots:
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        merged = (root / candidate).resolve()
        if merged.exists():
            return merged
    return (Path.cwd() / candidate).resolve()


def load_json_options_from_txt(txt_path, project_dir=None):
    txt_file = resolve_user_supplied_path(txt_path)
    if not txt_file.is_file():
        raise FileNotFoundError(f"TXT 文件不存在：{txt_file}")

    try:
        lines = txt_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = txt_file.read_text(encoding="gbk").splitlines()

    options = []
    seen = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Allow CSV-ish lines and quoted paths.
        line = line.rstrip(",;")
        line = line.strip().strip('"').strip("'")
        if not line:
            continue

        resolved = resolve_json_option_path(line, txt_file_path=txt_file, project_dir=project_dir)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)

        if resolved.suffix.lower() != ".json":
            continue

        options.append({
            "path": key,
            "name": resolved.name,
            "exists": resolved.is_file(),
        })

    # Existing files first, then lexicographic path order for stable UI.
    options.sort(key=lambda item: (not item["exists"], item["path"]))
    return options


def get_image_meta_cached(image_path):
    try:
        stat = os.stat(image_path)
    except OSError:
        return 0, 0

    cache_key = str(image_path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = IMAGE_META_CACHE.get(cache_key)
    if cached and cached.get("stamp") == stamp:
        return cached.get("width", 0), cached.get("height", 0)

    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError:
        width, height = 0, 0

    IMAGE_META_CACHE[cache_key] = {
        "stamp": stamp,
        "width": width,
        "height": height,
    }
    return width, height


def clamp_region(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    return {
        "x": round(x1),
        "y": round(y1),
        "w": round(x2 - x1),
        "h": round(y2 - y1),
    }


def detection_regions(raw_output, width, height):
    regions = []
    for detection in raw_output.get("detections", []):
        if not isinstance(detection, dict) or not isinstance(detection.get("box"), list):
            continue
        region = clamp_region(detection["box"], width, height)
        region.update({
            "label": f"{detection.get('class_name', 'object')} {float(detection.get('confidence', 0)):.3f}",
            "stroke": "#b42318",
            "fill": None,
        })
        regions.append(region)
    return regions


def analyse_regions(raw_output, analysed_output, width, height):
    status = analysed_output.get("pred_status", "INVALID")
    colors = {
        "OK": "#20744a",
        "NG": "#b42318",
        "INVALID": "#9a6700",
    }
    stroke = colors.get(status, colors["INVALID"])

    regions = [{
        "x": 0,
        "y": 0,
        "w": width,
        "h": height,
        "label": "",
        "stroke": stroke,
        "fill": None,
    }]

    for detection in detection_regions(raw_output, width, height):
        detection = {
            **detection,
            "stroke": "#20744a" if status == "OK" else stroke,
            "fill": None,
        }
        regions.append(detection)

    return regions


def compact_raw_output(raw_output):
    detections = []
    for detection in raw_output.get("detections", []):
        if not isinstance(detection, dict) or not isinstance(detection.get("box"), list):
            continue
        detections.append({
            "box": detection.get("box", []),
            "class_name": detection.get("class_name", "object"),
            "confidence": detection.get("confidence", 0),
        })

    rect_box = (((raw_output.get("aligned_rect") or {}).get("rect") or {}).get("box") or {})
    return {
        "detections": detections,
        "crop_box": raw_output.get("crop_box") or {},
        "aligned_rect": {"rect": {"box": rect_box}},
    }


def _load_inference_base_items(result_json_path, project_dir, stamp):
    """Parse and normalize the result JSON once, independent of UI filters."""
    result_json_path = resolve_user_supplied_path(result_json_path)
    base_key = (str(result_json_path), stamp, str(project_dir or ""), "base-v2")
    with INFERENCE_CACHE_LOCK:
        cached = INFERENCE_BASE_CACHE.get(base_key)
    if cached is not None:
        return cached

    if orjson is not None:
        payload = orjson.loads(result_json_path.read_bytes())
    else:
        with result_json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

    items = []
    for sample in payload.get("samples", []):
        for chip in sample.get("chips", []):
            yolo_block = chip.get("yolo") or {}
            for light_name, light_data in yolo_block.items():
                raw = (
                    light_data.get("raw_trt_output")
                    or light_data.get("raw_output")
                    or light_data.get("raw")
                )
                image_reference = (
                    light_data.get("path_")
                    or light_data.get("predict_input")
                    or light_data.get("image_path")
                    or light_data.get("path")
                )
                if not isinstance(raw, dict) or not image_reference:
                    continue

                image_path = resolve_result_path(result_json_path, image_reference, project_dir)
                if not project_dir and not image_path.is_file():
                    continue

                analysed = (
                    light_data.get("analyzed_output")
                    or light_data.get("analyse_output")
                    or {}
                )
                current_light_prediction = INFER_MODEL_PREDICTION_MAP.get(
                    analysed.get("pred_status"), "未知"
                )
                chip_key = chip.get("chip_key", {}) or {}
                mechanical = chip.get("mechanical_columns") or {}
                searchable = " ".join(map(str, [
                    image_path.name,
                    sample.get("sample_id", ""),
                    sample.get("num_str", ""),
                    light_name,
                    mechanical.get("MX", ""),
                    mechanical.get("MY", ""),
                    (chip.get("final") or {}).get("status", ""),
                    (chip.get("final") or {}).get("class", ""),
                    (chip.get("final") or {}).get("reason", ""),
                    chip_key.get("nImageNum", ""),
                    chip_key.get("nIndex", ""),
                ])).lower()

                original_path = str(image_path.resolve())
                MODEL_PREDICTION_BY_PATH[original_path] = current_light_prediction
                MODEL_PREDICTION_BY_PATH[os.path.normcase(original_path)] = current_light_prediction
                items.append({
                    "id": quote(original_path, safe=""),
                    "name": image_path.name,
                    "originalPath": original_path,
                    "displayPath": original_path,
                    "imageUrl": "/api/image?path=" + quote(original_path, safe=""),
                    "width": 0,
                    "height": 0,
                    "modelPrediction": current_light_prediction,
                    "modelReason": ", ".join(analysed.get("decision_reason", []))
                    if isinstance(analysed.get("decision_reason"), list)
                    else analysed.get("decision_reason", ""),
                    "lightType": light_name,
                    "sampleId": sample.get("sample_id", ""),
                    "numStr": sample.get("num_str", ""),
                    "chipKey": chip_key,
                    "predictionOverlay": {"detection": [], "analyse": []},
                    "_searchText": searchable,
                    "_rawOutput": compact_raw_output(raw),
                    "_analysedOutput": analysed,
                })

    with INFERENCE_CACHE_LOCK:
        INFERENCE_BASE_CACHE[base_key] = items
    return items


def load_inference_items(result_json_path, light_type, model_prediction_filter, keyword, image_search, project_dir=None):
    result_json_path = resolve_user_supplied_path(result_json_path)
    try:
        stat = result_json_path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None

    cache_key = (
        "filtered-v2",
        str(result_json_path),
        stamp,
        light_type,
        model_prediction_filter,
        (keyword or "").lower(),
        (image_search or "").lower(),
        str(project_dir or ""),
    )
    cached = INFERENCE_ITEMS_CACHE.get(cache_key)
    if cached is not None:
        for cached_item in cached:
            cached_path = cached_item.get("originalPath", "")
            cached_prediction = cached_item.get("modelPrediction", "")
            if cached_path and cached_prediction:
                MODEL_PREDICTION_BY_PATH[cached_path] = cached_prediction
                MODEL_PREDICTION_BY_PATH[os.path.normcase(cached_path)] = cached_prediction
        return cached

    cache_id = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    disk_cache_path = INFERENCE_DISK_CACHE_DIR / f"{cache_id}.pkl"
    try:
        with disk_cache_path.open("rb") as file:
            cached = pickle.load(file)
        if isinstance(cached, list):
            for cached_item in cached:
                cached_path = cached_item.get("originalPath", "")
                cached_prediction = cached_item.get("modelPrediction", "")
                if cached_path and cached_prediction:
                    MODEL_PREDICTION_BY_PATH[cached_path] = cached_prediction
                    MODEL_PREDICTION_BY_PATH[os.path.normcase(cached_path)] = cached_prediction
            INFERENCE_ITEMS_CACHE[cache_key] = cached
            return cached
    except (OSError, EOFError, pickle.PickleError, ValueError):
        pass

    keyword_lower = (keyword or "").lower()
    search_lower = (image_search or "").lower()
    base_items = _load_inference_base_items(result_json_path, project_dir, stamp)
    items = []
    for base_item in base_items:
        if light_type != "All" and base_item.get("lightType") != light_type:
            continue
        if model_prediction_filter != "All" and base_item.get("modelPrediction") != model_prediction_filter:
            continue
        if keyword_lower and keyword_lower not in base_item.get("name", "").lower():
            continue
        if search_lower and search_lower not in base_item.get("_searchText", ""):
            continue
        items.append(base_item)

    INFERENCE_ITEMS_CACHE[cache_key] = items
    try:
        INFERENCE_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_cache_path = disk_cache_path.with_suffix(".tmp")
        with temp_cache_path.open("wb") as file:
            pickle.dump(items, file, protocol=pickle.HIGHEST_PROTOCOL)
        temp_cache_path.replace(disk_cache_path)
    except OSError:
        pass
    return items


def empty_stats():
    return {
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "skipped": 0,
    }


def finalize_stats(stats):
    false_positive_denominator = stats["fp"] + stats["tn"]
    false_negative_denominator = stats["fn"] + stats["tp"]
    total = stats["tn"] + stats["fp"] + stats["fn"] + stats["tp"]
    return {
        **stats,
        "total": total,
        "falsePositiveRate": stats["fp"] / false_positive_denominator if false_positive_denominator else None,
        "falseNegativeRate": stats["fn"] / false_negative_denominator if false_negative_denominator else None,
        "hzFalsePositiveRate": stats["fp"] / total if total else None,
        "hzFalseNegativeRate": stats["fn"] / total if total else None,
        "falsePositiveDenominator": false_positive_denominator,
        "falseNegativeDenominator": false_negative_denominator,
    }


def accumulate_stats(stats, path, annotation):
    normalized = normalize_annotation(annotation)
    verdict = normalized.get("verdict", "")
    model_prediction = (
        normalized.get("modelPrediction")
        or MODEL_PREDICTION_BY_PATH.get(path)
        or MODEL_PREDICTION_BY_PATH.get(os.path.normcase(path))
        or get_model_prediction(path)
    )

    if verdict not in ("分类正确", "分类错误") or model_prediction not in ("合格品", "缺陷品"):
        stats["skipped"] += 1
        return

    if model_prediction == "合格品" and verdict == "分类正确":
        stats["tn"] += 1
    elif model_prediction == "合格品" and verdict == "分类错误":
        stats["fn"] += 1
    elif model_prediction == "缺陷品" and verdict == "分类错误":
        stats["fp"] += 1
    elif model_prediction == "缺陷品" and verdict == "分类正确":
        stats["tp"] += 1


def get_confusion_cell(path, annotation):
    stats = empty_stats()
    accumulate_stats(stats, path, annotation)
    if stats["skipped"]:
        return ""
    for cell in ("tp", "fn", "fp", "tn"):
        if stats[cell]:
            return cell.upper()
    return ""


def matches_confusion_filter(path, annotations, confusion_cell, confusion_light, annotation_index=None):
    confusion_cell = (confusion_cell or "").upper()
    confusion_light = confusion_light or ""
    if confusion_cell in ("", "ALL"):
        return True
    if confusion_cell not in ("TP", "FN", "FP", "TN"):
        return True
    if confusion_light and confusion_light != "All" and get_light_type(path) != confusion_light:
        return False
    annotation = get_annotation_for_image(path, annotations, annotation_index)
    return get_confusion_cell(path, annotation) == confusion_cell


def build_confusion_basename_set(annotations, confusion_cell, confusion_light):
    confusion_cell = (confusion_cell or "").upper()
    confusion_light = confusion_light or ""
    if confusion_cell not in ("TP", "FN", "FP", "TN"):
        return None

    matched = set()
    for annotation_path, annotation in annotations.items():
        if confusion_light and confusion_light != "All" and get_light_type(annotation_path) != confusion_light:
            continue
        if get_confusion_cell(annotation_path, annotation) == confusion_cell:
            matched.add(annotation_lookup_key(annotation_path))
    return matched


def load_annotations():
    if not ANNOTATION_FILE.exists():
        return {}
    try:
        with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_annotations(data):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = ANNOTATION_FILE.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_file.replace(ANNOTATION_FILE)


def clear_saved_annotations():
    """Start a new review run without discarding the parsed inference cache."""
    save_annotations({})


def normalize_annotation(annotation):
    if not isinstance(annotation, dict):
        annotation = {}

    issues = annotation.get("detectionIssues")
    if not isinstance(issues, list):
        old_type = annotation.get("defectType", "")
        issues = [old_type] if old_type else []

    clean_issues = []
    for issue in issues:
        if issue in ("漏检", "错检") and issue not in clean_issues:
            clean_issues.append(issue)

    miss_regions = annotation.get("missRegions", [])
    if not isinstance(miss_regions, list):
        miss_regions = []

    false_regions = annotation.get("falseRegions", [])
    if not isinstance(false_regions, list):
        false_regions = []

    green_defect_regions = annotation.get("greenDefectRegions", [])
    if not isinstance(green_defect_regions, list):
        green_defect_regions = []

    inference_removed_regions = annotation.get("inferenceRemovedRegions", [])
    if not isinstance(inference_removed_regions, list):
        inference_removed_regions = []
    inference_removed_regions = [
        item for item in inference_removed_regions
        if isinstance(item, dict)
    ]

    inference_regions = annotation.get("inferenceRegions", [])
    if not isinstance(inference_regions, list):
        inference_regions = []
    inference_regions = [
        item for item in inference_regions
        if isinstance(item, dict)
    ]

    verdict = CLASSIFICATION_VERDICT_MAP.get(annotation.get("verdict", ""), annotation.get("verdict", ""))
    model_prediction = annotation.get("modelPrediction", annotation.get("model_prediction", ""))
    model_prediction = INFER_MODEL_PREDICTION_MAP.get(model_prediction, model_prediction)
    if model_prediction not in ("合格品", "缺陷品"):
        model_prediction = ""
    green_defect = bool(annotation.get("greenDefect", False))
    if annotation.get("logicVerdict") == "NG" or green_defect_regions:
        green_defect = True
    if verdict != "分类错误":
        green_defect = False
        green_defect_regions = []

    if miss_regions and "漏检" not in clean_issues:
        clean_issues.append("漏检")
    if false_regions and "错检" not in clean_issues:
        clean_issues.append("错检")

    return {
        "verdict": verdict,
        "modelPrediction": model_prediction,
        "greenDefect": green_defect,
        "greenDefectRegions": green_defect_regions,
        "inferenceRegions": inference_regions,
        "inferenceRemovedRegions": inference_removed_regions,
        "detectionIssues": clean_issues,
        "defectType": clean_issues[0] if clean_issues else "",
        "missRegions": miss_regions,
        "falseRegions": false_regions,
        "note": annotation.get("note", ""),
        "imageName": annotation.get("imageName", ""),
        "updatedAt": annotation.get("updatedAt", ""),
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value):
    return max(0.0, min(1.0, value))


def _normalize_region_for_app1(region, image_width, image_height, default_class_name=""):
    if not isinstance(region, dict):
        return None

    x = _to_float(region.get("x"))
    y = _to_float(region.get("y"))
    w = _to_float(region.get("w"))
    h = _to_float(region.get("h"))
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None

    # Existing annotations are typically absolute pixels. If values are already
    # normalized, keep them as-is; otherwise convert using image dimensions.
    looks_normalized = (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 <= w <= 1.0
        and 0.0 <= h <= 1.0
        and x + w <= 1.000001
        and y + h <= 1.000001
    )
    if not looks_normalized:
        if image_width <= 0 or image_height <= 0:
            return None
        x = x / image_width
        y = y / image_height
        w = w / image_width
        h = h / image_height

    x = _clamp01(x)
    y = _clamp01(y)
    w = _clamp01(w)
    h = _clamp01(h)
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    if w <= 0 or h <= 0:
        return None

    return {
        "x": round(x, 6),
        "y": round(y, 6),
        "w": round(w, 6),
        "h": round(h, 6),
        "className": str(region.get("className") or region.get("label") or default_class_name or ""),
        "note": str(region.get("note") or ""),
    }


def _region_signature(region):
    x = _to_float(region.get("x")) or 0.0
    y = _to_float(region.get("y")) or 0.0
    w = _to_float(region.get("w")) or 0.0
    h = _to_float(region.get("h")) or 0.0
    return "|".join([
        f"{x:.6f}",
        f"{y:.6f}",
        f"{w:.6f}",
        f"{h:.6f}",
        str(region.get("label") or ""),
    ])


def annotation_to_app1_compatible_row(path, annotation, inference_regions=None):
    normalized = normalize_annotation(annotation)
    verdict = normalized.get("verdict", "")
    if not isinstance(inference_regions, list):
        inference_regions = []
    persisted_inference_regions = normalized.get("inferenceRegions", []) if isinstance(normalized.get("inferenceRegions", []), list) else []
    candidate_inference_regions = inference_regions or persisted_inference_regions
    has_any_regions = any(
        isinstance(normalized.get(key), list) and len(normalized.get(key)) > 0
        for key in ("greenDefectRegions", "missRegions", "falseRegions")
    )
    has_any_regions = has_any_regions or len(candidate_inference_regions) > 0
    status = "good" if verdict == "分类正确" else ("bad" if verdict == "分类错误" else "")
    if has_any_regions:
        # app1 clears regions when status=good. Keep boxes visible after import.
        status = "bad"

    removed_signatures = {
        _region_signature(region)
        for region in normalized.get("inferenceRemovedRegions", [])
        if isinstance(region, dict)
    }

    filtered_inference_regions = []
    for region in candidate_inference_regions:
        if not isinstance(region, dict):
            continue
        if _region_signature(region) in removed_signatures:
            continue
        filtered_inference_regions.append(region)

    image_width, image_height = get_image_meta_cached(path)
    regions = []
    exported_inference_regions = []
    if status == "bad":
        for default_name, blocks in (
            ("绿色缺陷", normalized.get("greenDefectRegions", [])),
            ("漏检", normalized.get("missRegions", [])),
            ("错检", normalized.get("falseRegions", [])),
        ):
            for region in blocks if isinstance(blocks, list) else []:
                normalized_region = _normalize_region_for_app1(
                    region,
                    image_width=image_width,
                    image_height=image_height,
                    default_class_name=default_name,
                )
                if normalized_region:
                    if default_name == "漏检":
                        normalized_region["source"] = "miss_region"
                    elif default_name == "错检":
                        normalized_region["source"] = "false_region"
                    elif default_name == "绿色缺陷":
                        normalized_region["source"] = "green_defect_region"
                    regions.append(normalized_region)

        for region in filtered_inference_regions:
            normalized_region = _normalize_region_for_app1(
                region,
                image_width=image_width,
                image_height=image_height,
                default_class_name="推理框",
            )
            if not normalized_region:
                continue
            normalized_region["source"] = "inference_detection"
            if region.get("note"):
                normalized_region["note"] = str(region.get("note"))
            exported_inference_regions.append(dict(normalized_region))
            regions.append(normalized_region)

        # Deduplicate highly similar boxes while preserving insertion order.
        seen = set()
        deduped = []
        for region in regions:
            key = (
                round(float(region.get("x", 0)), 6),
                round(float(region.get("y", 0)), 6),
                round(float(region.get("w", 0)), 6),
                round(float(region.get("h", 0)), 6),
                str(region.get("className", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(region)
        regions = deduped

    return {
        "originalPath": path,
        "imagePath": path,
        "imageName": normalized.get("imageName") or path_basename(path),
        "status": status,
        "regions": [] if status == "good" else regions,
        "inferenceRegions": [] if status == "good" else exported_inference_regions,
        "note": normalized.get("note", ""),
        "updatedAt": normalized.get("updatedAt", ""),
        # Preserve existing export fields for backward compatibility.
        **normalized,
    }


def build_inference_detection_region_index(result_json_path, project_dir=None):
    items = load_inference_items(
        result_json_path=result_json_path,
        light_type="All",
        model_prediction_filter="All",
        keyword="",
        image_search="",
        project_dir=project_dir,
    )
    indexed = {}
    for item in items:
        original_path = item.get("originalPath", "")
        if not original_path:
            continue
        width, height = get_image_meta_cached(original_path)
        if width <= 0 or height <= 0:
            continue
        raw_output = item.get("_rawOutput") or {}
        source_regions = []
        for region in detection_regions(raw_output, width, height):
            if not isinstance(region, dict):
                continue
            label = str(region.get("label") or "").strip()
            class_name = label.split(" ", 1)[0] if label else "推理框"
            source_regions.append({
                "x": region.get("x", 0),
                "y": region.get("y", 0),
                "w": region.get("w", 0),
                "h": region.get("h", 0),
                "className": class_name,
                "note": label,
            })

        key = os.path.normcase(original_path)
        indexed[key] = source_regions
    return indexed


def annotations_from_import(payload):
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        rows = payload["items"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = [
            {"originalPath": path, **annotation}
            for path, annotation in payload.items()
            if isinstance(annotation, dict)
        ]
    else:
        return {}

    imported = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        original_path = row.get("originalPath", "")
        if not original_path:
            continue
        annotation = normalize_annotation(row)
        annotation["imageName"] = annotation.get("imageName") or path_basename(original_path)
        imported[original_path] = annotation
    return imported


def auto_crop_image(image_path, base_dir, padding=5):
    rel_path = os.path.relpath(image_path, base_dir)
    cache_path = os.path.join(base_dir, "cropped_cache", rel_path)

    if os.path.exists(cache_path):
        return cache_path

    return image_path


def get_filtered_images(light_type, keyword, base_dir):
    keyword = (keyword or "").lower()
    if light_type == "All":
        patterns = [
            os.path.join(base_dir, "*", "yolo_pred_images", f"*{ext}")
            for ext in IMAGE_EXTENSIONS
        ]
    else:
        patterns = [
            os.path.join(base_dir, light_type, "yolo_pred_images", f"*{ext}")
            for ext in IMAGE_EXTENSIONS
        ]

    images = []
    for pattern in patterns:
        images.extend(glob.glob(pattern))

    return sorted(
        img for img in images
        if keyword in os.path.basename(img).lower()
    )


def annotation_matches_image(image_path, annotation_path, annotation):
    image_path_text = normalize_path_text(image_path).lower()
    annotation_path_text = normalize_path_text(annotation_path).lower()
    image_name = path_basename(image_path)
    annotation_name = path_basename(annotation.get("imageName") or annotation_path)
    return (
        image_path_text == annotation_path_text
        or (
            get_light_type(image_path) == get_light_type(annotation_path)
            and image_name == annotation_name
        )
    )


def build_annotation_index(annotations):
    indexed = {}
    for annotation_path, annotation in annotations.items():
        lookup_key = annotation_lookup_key(annotation_path)
        if lookup_key and lookup_key not in indexed:
            indexed[lookup_key] = annotation
    return indexed


def get_annotation_for_image(image_path, annotations, annotation_index=None):
    exact = annotations.get(image_path)
    if exact:
        return exact

    if annotation_index is not None:
        indexed = annotation_index.get(annotation_lookup_key(image_path))
        if indexed:
            return indexed
    return {}


def image_matches_search(image_path, annotations, search_text, annotation_index=None):
    search_text = (search_text or "").strip().lower()
    if not search_text:
        return True

    candidates = [image_path, os.path.basename(image_path)]
    annotation = get_annotation_for_image(image_path, annotations, annotation_index)
    if annotation:
        normalized = normalize_annotation(annotation)
        candidates.extend([
            normalized.get("imageName", ""),
            normalized.get("note", ""),
            normalized.get("verdict", ""),
            "有缺陷但框是绿色的" if normalized.get("greenDefect") else "",
            " ".join(normalized.get("detectionIssues", [])) if isinstance(normalized.get("detectionIssues"), list) else "",
        ])

    return any(search_text in str(candidate).lower() for candidate in candidates)


def order_json_first(paths, annotations, annotation_index=None):
    annotated = []
    plain = []
    for path in paths:
        if get_annotation_for_image(path, annotations, annotation_index):
            annotated.append(path)
        else:
            plain.append(path)
    return annotated + plain


def shuffle_paths(paths, seed):
    seed = seed or "default"
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(f"{seed}|{path}".encode("utf-8")).hexdigest()
    )


def make_image_path_export_items(paths, base_dir):
    """Build a portable path manifest for the currently filtered images."""
    absolute_base_dir = Path(base_dir).expanduser().resolve() if base_dir else None
    items = []
    for path in paths:
        absolute_path = Path(path).expanduser().resolve()
        source = ""
        if absolute_base_dir:
            try:
                source = absolute_path.parent.relative_to(absolute_base_dir).as_posix()
            except ValueError:
                source = os.path.relpath(absolute_path.parent, absolute_base_dir).replace("\\", "/")
        items.append({
            "imagePath": str(absolute_path),
            "imageName": absolute_path.name,
            "source": source,
        })
    return items


def json_bytes(payload):
    if orjson is not None:
        return orjson.dumps(payload)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class InspectionHandler(BaseHTTPRequestHandler):
    server_version = "InspectionWeb/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(STATIC_DIR / "index.html")
        elif parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/").lstrip("/")
            self.serve_file(STATIC_DIR / rel)
        elif parsed.path == "/api/images":
            self.handle_images(parse_qs(parsed.query))
        elif parsed.path == "/api/image":
            self.handle_image(parse_qs(parsed.query))
        elif parsed.path == "/api/annotations":
            self.handle_annotations_export(parse_qs(parsed.query))
        elif parsed.path == "/api/export-image-paths":
            self.handle_image_paths_export(parse_qs(parsed.query))
        elif parsed.path == "/api/json-options":
            self.handle_json_options(parse_qs(parsed.query))
        elif parsed.path == "/api/stats":
            self.handle_stats()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotation":
            self.handle_save_annotation()
        elif parsed.path == "/api/annotations/restore":
            self.handle_annotations_restore()
        elif parsed.path == "/api/annotations/bulk-default-correct":
            self.handle_bulk_default_correct()
        elif parsed.path == "/api/annotations/import":
            self.handle_annotations_import()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_file(self, path):
        try:
            resolved = Path(path).resolve()
            if not str(resolved).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            data = resolved.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        # HTML/JS/CSS are edited while the inspection app is running. Do not
        # let the browser keep an old control layout or request-building code.
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json_download(self, payload, filename):
        data = json_bytes(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_images(self, query):
        result_json = query.get("result_json", [""])[0].strip() or CONFIGURED_RESULT_JSON
        project_dir = query.get("project_dir", [""])[0].strip() or str(PROJECT_DIR)
        base_dir = query.get("base_dir", [""])[0].strip()
        light_type = query.get("light_type", ["All"])[0] or "All"
        keyword = query.get("keyword", [""])[0]
        page = max(1, int(float(query.get("page", ["1"])[0] or 1)))
        num_col = max(1, int(float(query.get("num_col", ["2"])[0] or 2)))
        num_row = max(1, int(float(query.get("num_row", ["30"])[0] or 30)))
        shuffle_enabled = query.get("shuffle", ["true"])[0].lower() in ("1", "true", "yes", "on")
        shuffle_seed = query.get("shuffle_seed", ["default"])[0]
        json_first = query.get("json_first", ["true"])[0].lower() in ("1", "true", "yes", "on")
        clear_cache = query.get("clear_cache", ["false"])[0].lower() in ("1", "true", "yes", "on")
        image_search = query.get("image_search", [""])[0]
        model_prediction_filter = query.get("model_prediction", ["All"])[0] or "All"
        confusion_cell = query.get("confusion_cell", [""])[0].upper()
        confusion_light = query.get("confusion_light", [""])[0]

        per_page = num_col * num_row
        cache_cleared = False
        removed_disk_files = 0
        if clear_cache and result_json:
            # The review cache is the saved annotation file. Keep the parsed
            # inference cache so a large result JSON can still use app1-style
            # fast reloads; its path/mtime/size cache key prevents stale JSON.
            clear_saved_annotations()
            cache_cleared = True
        annotations = load_annotations()
        annotation_index = build_annotation_index(annotations)
        source = ""

        try:
            if result_json:
                items_all = load_inference_items(
                    result_json,
                    light_type,
                    model_prediction_filter,
                    keyword,
                    image_search,
                    project_dir,
                )
                source = str(resolve_user_supplied_path(result_json))
            else:
                if not base_dir:
                    self.send_json({"error": "请填写推理结果 JSON 或图片根目录。"}, HTTPStatus.BAD_REQUEST)
                    return

                base_dir = os.path.abspath(os.path.expanduser(base_dir))
                if not os.path.isdir(base_dir):
                    self.send_json({"error": f"图片根目录不存在：{base_dir}"}, HTTPStatus.BAD_REQUEST)
                    return

                source = base_dir
                original_paths = get_filtered_images(light_type, keyword, base_dir)
                original_paths = [
                    path for path in original_paths
                    if image_matches_search(path, annotations, image_search, annotation_index)
                ]
                if model_prediction_filter != "All":
                    original_paths = [
                        path for path in original_paths
                        if get_model_prediction(path) == model_prediction_filter
                    ]

                items_all = []
                for original_path in original_paths:
                    display_path = auto_crop_image(original_path, base_dir)
                    try:
                        with Image.open(display_path) as img:
                            width, height = img.size
                    except Exception:
                        width, height = 0, 0

                    items_all.append({
                        "id": quote(original_path, safe=""),
                        "name": os.path.basename(original_path),
                        "modelPrediction": get_model_prediction(original_path),
                        "modelReason": "",
                        "originalPath": original_path,
                        "displayPath": display_path,
                        "imageUrl": "/api/image?path=" + quote(display_path, safe=""),
                        "width": width,
                        "height": height,
                        "predictionOverlay": {"detection": [], "analyse": []},
                    })
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": f"读取数据失败：{error}"}, HTTPStatus.BAD_REQUEST)
            return

        filtered_items = []
        confusion_basename_set = build_confusion_basename_set(annotations, confusion_cell, confusion_light)
        need_confusion_match = bool(confusion_cell and confusion_cell not in ("ALL",))
        annotation_paths = set(annotations.keys())
        annotation_key_set = set(annotation_index.keys())

        def has_annotation_quick(original_path):
            return (original_path in annotation_paths) or (annotation_lookup_key(original_path) in annotation_key_set)

        for item in items_all:
            original_path = item.get("originalPath", "")
            if not original_path:
                continue
            if confusion_basename_set is not None:
                if annotation_lookup_key(original_path) not in confusion_basename_set:
                    continue
            elif need_confusion_match and not matches_confusion_filter(original_path, annotations, confusion_cell, confusion_light, annotation_index):
                continue
            item["_hasAnnotation"] = has_annotation_quick(original_path)
            filtered_items.append(item)

        if shuffle_enabled:
            filtered_items = sorted(
                filtered_items,
                key=lambda item: hashlib.sha256(f"{shuffle_seed}|{item.get('originalPath', '')}".encode("utf-8")).hexdigest(),
            )
        if json_first:
            filtered_items = sorted(filtered_items, key=lambda item: 0 if item.get("_hasAnnotation") else 1)

        total_pages = max(1, (len(filtered_items) + per_page - 1) // per_page)
        page = min(page, total_pages)
        start = (page - 1) * per_page
        page_items = filtered_items[start:start + per_page]
        items = []

        for base_item in page_items:
            item = dict(base_item)
            width, height = get_image_meta_cached(item.get("originalPath", ""))
            item["width"] = width
            item["height"] = height
            raw_output = item.get("_rawOutput") or {}
            analysed_output = item.get("_analysedOutput") or {}
            item["predictionOverlay"] = {
                    "detection": detection_regions(raw_output, width, height),
                    "analyse": analyse_regions(raw_output, analysed_output, width, height),
            }
            item["annotation"] = normalize_annotation(
                get_annotation_for_image(item.get("originalPath", ""), annotations, annotation_index)
            )
            item.pop("_rawOutput", None)
            item.pop("_analysedOutput", None)
            item.pop("_hasAnnotation", None)
            items.append(item)

        batch_w = items[0].get("width", 1920) if items else 1920
        batch_h = items[0].get("height", 1080) if items else 1080

        self.send_json({
            "items": items,
            "page": page,
            "totalPages": total_pages,
            "total": len(filtered_items),
            "batchWidth": batch_w,
            "batchHeight": batch_h,
            "baseDir": source,
            "projectDir": project_dir,
            "resultJson": result_json,
            "shuffle": shuffle_enabled,
            "shuffleSeed": shuffle_seed,
            "jsonFirst": json_first,
            "imageSearch": image_search,
            "modelPredictionFilter": model_prediction_filter,
            "confusionCell": confusion_cell,
            "confusionLight": confusion_light,
            "cacheCleared": cache_cleared,
            "cacheFilesRemoved": removed_disk_files,
        })

    def handle_image(self, query):
        raw_path = query.get("path", [""])[0]
        image_path = os.path.abspath(os.path.expanduser(unquote(raw_path)))
        if not os.path.isfile(image_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        if Path(image_path).suffix.lower() not in IMAGE_EXTENSIONS:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unsupported image type")
            return

        try:
            data = Path(image_path).read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return

        content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_save_annotation(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "评价数据格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        original_path = payload.get("originalPath", "")
        if not original_path:
            self.send_json({"error": "缺少图片路径。"}, HTTPStatus.BAD_REQUEST)
            return

        verdict = CLASSIFICATION_VERDICT_MAP.get(payload.get("verdict", ""), payload.get("verdict", ""))
        if verdict not in ("分类正确", "分类错误"):
            self.send_json({"error": "请先选择“分类正确”或“分类错误”。"}, HTTPStatus.BAD_REQUEST)
            return

        detection_issues = payload.get("detectionIssues", [])
        if not isinstance(detection_issues, list):
            detection_issues = []

        annotation = normalize_annotation({
            "verdict": verdict,
            "greenDefect": payload.get("greenDefect", False),
            "greenDefectRegions": payload.get("greenDefectRegions", []),
            "inferenceRegions": payload.get("inferenceRegions", []),
            "inferenceRemovedRegions": payload.get("inferenceRemovedRegions", []),
            "detectionIssues": detection_issues,
            "missRegions": payload.get("missRegions", []),
            "falseRegions": payload.get("falseRegions", []),
            "note": payload.get("note", ""),
            "imageName": os.path.basename(original_path),
            "updatedAt": payload.get("updatedAt", ""),
        })
        annotations = load_annotations()
        annotations[original_path] = annotation
        save_annotations(annotations)
        self.send_json({"ok": True, "annotation": annotation})

    def handle_annotations_restore(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "恢复数据格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list) or not entries:
            self.send_json({"error": "缺少可恢复的评价记录。"}, HTTPStatus.BAD_REQUEST)
            return

        annotations = load_annotations()
        restored = 0
        deleted = 0
        skipped = 0

        for entry in entries:
            if not isinstance(entry, dict):
                skipped += 1
                continue

            original_path = entry.get("originalPath", "")
            if not isinstance(original_path, str) or not original_path:
                skipped += 1
                continue

            if entry.get("annotation") is None:
                if original_path in annotations:
                    del annotations[original_path]
                    deleted += 1
                continue

            annotation = entry.get("annotation")
            if not isinstance(annotation, dict):
                skipped += 1
                continue

            normalized = normalize_annotation(annotation)
            normalized["imageName"] = normalized.get("imageName") or os.path.basename(original_path)
            annotations[original_path] = normalized
            restored += 1

        if restored or deleted:
            save_annotations(annotations)

        self.send_json({
            "ok": True,
            "restored": restored,
            "deleted": deleted,
            "skipped": skipped,
            "total": len(annotations),
        })

    def handle_annotations_import(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "导入 JSON 格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        imported = annotations_from_import(payload)
        if not imported:
            self.send_json({"error": "没有找到可导入的评价记录。"}, HTTPStatus.BAD_REQUEST)
            return

        save_annotations(imported)

        self.send_json({
            "ok": True,
            "imported": len(imported),
            "total": len(imported),
        })

    def handle_bulk_default_correct(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "批量保存数据格式错误。"}, HTTPStatus.BAD_REQUEST)
            return

        paths = payload.get("paths", [])
        if not isinstance(paths, list) or not paths:
            self.send_json({"error": "缺少当前页图片路径。"}, HTTPStatus.BAD_REQUEST)
            return

        annotations = load_annotations()
        updated_paths = []
        skipped_paths = []

        for original_path in paths:
            if not isinstance(original_path, str) or not original_path:
                continue

            existing = normalize_annotation(get_annotation_for_image(original_path, annotations))
            if existing.get("verdict") in ("分类正确", "分类错误"):
                skipped_paths.append(original_path)
                continue

            annotations[original_path] = normalize_annotation({
                "verdict": "分类正确",
                "modelPrediction": MODEL_PREDICTION_BY_PATH.get(original_path, ""),
                "greenDefect": False,
                "greenDefectRegions": [],
                "detectionIssues": [],
                "missRegions": [],
                "falseRegions": [],
                "note": payload.get("note", ""),
                "imageName": os.path.basename(original_path),
                "updatedAt": payload.get("updatedAt", ""),
            })
            updated_paths.append(original_path)

        if updated_paths:
            save_annotations(annotations)

        self.send_json({
            "ok": True,
            "updated": len(updated_paths),
            "skipped": len(skipped_paths),
            "updatedPaths": updated_paths,
            "skippedPaths": skipped_paths,
        })

    def handle_annotations_export(self, query=None):
        query = query or {}
        result_json = query.get("result_json", [""])[0].strip() or CONFIGURED_RESULT_JSON
        project_dir = query.get("project_dir", [""])[0].strip() or str(PROJECT_DIR)

        inference_regions_by_path = {}
        if result_json:
            try:
                inference_regions_by_path = build_inference_detection_region_index(
                    result_json_path=result_json,
                    project_dir=project_dir,
                )
            except Exception:
                inference_regions_by_path = {}

        annotations = load_annotations()
        rows = []
        for path, annotation in annotations.items():
            rows.append(annotation_to_app1_compatible_row(
                path,
                annotation,
                inference_regions=inference_regions_by_path.get(os.path.normcase(path), []),
            ))
        self.send_json_download({"count": len(rows), "items": rows}, "inspection_annotations.json")

    def handle_image_paths_export(self, query):
        """Export all paths matching the active filters, not only the visible page."""
        result_json = query.get("result_json", [""])[0].strip() or CONFIGURED_RESULT_JSON
        project_dir = query.get("project_dir", [""])[0].strip() or str(PROJECT_DIR)
        base_dir = query.get("base_dir", [""])[0].strip()
        light_type = query.get("light_type", ["All"])[0] or "All"
        keyword = query.get("keyword", [""])[0]
        image_search = query.get("image_search", [""])[0]
        model_prediction_filter = query.get("model_prediction", ["All"])[0] or "All"
        confusion_cell = query.get("confusion_cell", [""])[0].upper()
        confusion_light = query.get("confusion_light", [""])[0]

        annotations = load_annotations()
        annotation_index = build_annotation_index(annotations)
        source_root = base_dir

        try:
            if result_json:
                result_json_path = resolve_user_supplied_path(result_json)
                items = load_inference_items(
                    result_json_path,
                    light_type,
                    model_prediction_filter,
                    keyword,
                    image_search,
                    project_dir,
                )
                paths = [item.get("originalPath", "") for item in items if item.get("originalPath")]
                if not source_root:
                    source_root = project_dir or str(result_json_path.parent)
            else:
                if not base_dir:
                    self.send_json({"error": "请填写推理结果 JSON 或图片根目录。"}, HTTPStatus.BAD_REQUEST)
                    return
                source_root = os.path.abspath(os.path.expanduser(base_dir))
                if not os.path.isdir(source_root):
                    self.send_json({"error": f"图片根目录不存在：{source_root}"}, HTTPStatus.BAD_REQUEST)
                    return
                paths = get_filtered_images(light_type, keyword, source_root)
                paths = [
                    path for path in paths
                    if image_matches_search(path, annotations, image_search, annotation_index)
                ]
                if model_prediction_filter != "All":
                    paths = [
                        path for path in paths
                        if get_model_prediction(path) == model_prediction_filter
                    ]

            confusion_basename_set = build_confusion_basename_set(
                annotations, confusion_cell, confusion_light
            )
            filtered_paths = []
            for path in paths:
                if confusion_basename_set is not None:
                    if annotation_lookup_key(path) not in confusion_basename_set:
                        continue
                elif confusion_cell and confusion_cell != "ALL" and not matches_confusion_filter(
                    path, annotations, confusion_cell, confusion_light, annotation_index
                ):
                    continue
                filtered_paths.append(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": f"读取数据失败：{error}"}, HTTPStatus.BAD_REQUEST)
            return

        self.send_json_download(
            {
                "count": len(filtered_paths),
                "items": make_image_path_export_items(filtered_paths, source_root),
            },
            "filtered_image_paths.json",
        )

    def handle_json_options(self, query):
        project_dir = query.get("project_dir", [""])[0].strip() or str(PROJECT_DIR)

        try:
            items = load_json_options_from_txt(str(JSON_OPTIONS_TXT), project_dir=project_dir)
        except FileNotFoundError:
            self.send_json({
                "count": 0,
                "items": [],
                "warning": f"固定候选 TXT 不存在：{JSON_OPTIONS_TXT}",
            })
            return
        except (OSError, ValueError) as error:
            self.send_json({"error": f"读取 TXT 失败：{error}"}, HTTPStatus.BAD_REQUEST)
            return

        self.send_json({
            "count": len(items),
            "items": items,
        })

    def handle_stats(self):
        annotations = load_annotations()
        backfilled = 0
        for path, annotation in annotations.items():
            model_prediction = MODEL_PREDICTION_BY_PATH.get(path) or MODEL_PREDICTION_BY_PATH.get(os.path.normcase(path))
            if model_prediction and not normalize_annotation(annotation).get("modelPrediction"):
                annotation["modelPrediction"] = model_prediction
                backfilled += 1
        if backfilled:
            save_annotations(annotations)
        overall = empty_stats()
        by_light = {}

        for path, annotation in annotations.items():
            accumulate_stats(overall, path, annotation)
            light_type = get_light_type(path)
            by_light.setdefault(light_type, empty_stats())
            accumulate_stats(by_light[light_type], path, annotation)

        self.send_json({
            **finalize_stats(overall),
            "backfilled": backfilled,
            "overall": finalize_stats(overall),
            "byLight": {
                light_type: finalize_stats(stats)
                for light_type, stats in sorted(by_light.items())
            },
        })


def main():
    global PROJECT_DIR, CONFIGURED_RESULT_JSON
    parser = argparse.ArgumentParser(description="Industrial image inspection web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7868, type=int)
    parser.add_argument("--result-json", default=str(DEFAULT_RESULT_JSON))
    parser.add_argument(
        "--project-dir",
        "--image-root",
        "--dir",
        dest="project_dir",
        default=str(PROJECT_DIR),
        help="用于拼接 JSON 中 path_ 相对路径的工程目录",
    )
    args = parser.parse_args()

    PROJECT_DIR = resolve_user_supplied_path(args.project_dir)
    if not PROJECT_DIR.is_dir():
        raise SystemExit(f"工程目录不存在：{PROJECT_DIR}")
    CONFIGURED_RESULT_JSON = str(resolve_user_supplied_path(args.result_json))

    server = ThreadingHTTPServer((args.host, args.port), InspectionHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Inspection web app running at {url}", flush=True)
    print(f"Annotations: {ANNOTATION_FILE}", flush=True)
    print(f"Result JSON: {CONFIGURED_RESULT_JSON}", flush=True)
    print(f"Project directory: {PROJECT_DIR}", flush=True)
    if not Path(CONFIGURED_RESULT_JSON).is_file():
        print("Result JSON not found yet; enter a valid JSON path in the web page when ready.", flush=True)
    else:
        threading.Thread(
            target=load_inference_items,
            args=(CONFIGURED_RESULT_JSON, "All", "All", "", "", str(PROJECT_DIR)),
            daemon=True,
            name="inference-cache-warmup",
        ).start()
        print("Inference cache warm-up started in background.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
