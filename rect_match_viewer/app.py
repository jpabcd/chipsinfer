from __future__ import annotations

import base64
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rect_detector.align_chip_rects import align_rects_to_mechanical_txt, parse_mechanical_txt  # noqa: E402
from rect_detector.extract_chip_rects import RectInferer  # noqa: E402
from rect_detector.raw_batch_datasetV2 import read_gray_raw  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
LAST_RESULT: dict = {}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def image_data_url(image: np.ndarray, max_side: int = 1200) -> str:
    image = np.asarray(image)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if image.ndim == 2:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("failed to encode image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def draw_result(image: np.ndarray, align_result) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    matched_rect_ids = set()
    for item in align_result.matches:
        box = item.rect.box
        matched_rect_ids.add(item.rect.id)
        color = (0, 190, 0) if item.residual_um <= 10 else (0, 210, 220)
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, 5)
        label = f"#{item.rect.id} MX={item.mech.MX:g} MY={item.mech.MY:g} R={item.residual_um:.1f}um"
        cv2.putText(canvas, label, (box.x, max(28, box.y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    for rect in align_result.unmatched_rects:
        box = rect.box
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), (0, 0, 255), 6)
        cv2.putText(canvas, f"#{rect.id} UNMATCHED", (box.x, max(28, box.y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def run_match(payload: dict) -> dict:
    config_path = resolve_path(str(payload["config_path"]))
    msk_path = resolve_path(str(payload["msk_path"]))
    txt_path = resolve_path(str(payload.get("txt_path") or msk_path.with_suffix(".txt")))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = config.get("raw_image", {})
    rect_config = config.get("rect_detection", {})
    alignment = config.get("alignment", {})
    image = read_gray_raw(
        msk_path,
        width=int(raw.get("width", 5120)),
        height=int(raw.get("height", 5120)),
        dtype=str(raw.get("dtype", "auto")),
        byte_order=str(raw.get("byte_order", "little")),
    )
    inferer = RectInferer(
        **rect_config,
        mech_delta_x=float(alignment["mech_delta_x"]),
        mech_delta_y=float(alignment["mech_delta_y"]),
        delta_x_pixel=float(alignment["delta_x_pixel"]),
        delta_y_pixel=float(alignment["delta_y_pixel"]),
    )
    rects = inferer(image)
    mechanical_infos = parse_mechanical_txt(txt_path)
    if not rects:
        return {
            "status": "no_rects",
            "message": "MSK 中没有找到满足当前 rect_detection 参数的前景矩形，因此未执行坐标匹配。",
            "config_path": str(config_path),
            "msk_path": str(msk_path),
            "txt_path": str(txt_path),
            "config": config,
            "txt": txt_path.read_text(encoding="utf-8", errors="replace"),
            "msk_image": image_data_url(image),
            "overlay_image": image_data_url(image),
            "summary": {
                "rect_count": 0,
                "mechanical_count": len(mechanical_infos),
                "matched_count": 0,
                "unmatched_rect_count": 0,
                "unmatched_mechanical_count": len(mechanical_infos),
                "rmse_um": None,
                "max_residual_um": None,
                "x_reversed": False,
                "y_reversed": False,
            },
            "records": [],
        }
    align_result = align_rects_to_mechanical_txt(
        rects,
        txt_path,
        mech_delta_x=float(alignment["mech_delta_x"]),
        mech_delta_y=float(alignment["mech_delta_y"]),
        delta_x_pixel=float(alignment["delta_x_pixel"]),
        delta_y_pixel=float(alignment["delta_y_pixel"]),
        allow_partial=not bool(alignment.get("strict_align", False)),
        allow_x_reverse=bool(alignment.get("allow_x_reverse", False)),
        allow_y_reverse=bool(alignment.get("allow_y_reverse", False)),
    )
    records = []
    for item in align_result.matches:
        box = item.rect.box
        records.append({
            "rect_id": item.rect.id,
            "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
            "mx": item.mech.MX,
            "my": item.mech.MY,
            "residual_um": item.residual_um,
            "matched": True,
        })
    records.extend({
        "rect_id": rect.id,
        "box": {"x": rect.box.x, "y": rect.box.y, "w": rect.box.w, "h": rect.box.h},
        "matched": False,
    } for rect in align_result.unmatched_rects)
    return {
        "config_path": str(config_path),
        "msk_path": str(msk_path),
        "txt_path": str(txt_path),
        "config": config,
        "txt": txt_path.read_text(encoding="utf-8", errors="replace"),
        "msk_image": image_data_url(image),
        "overlay_image": image_data_url(draw_result(image, align_result)),
        "summary": {
            "rect_count": len(rects),
            "mechanical_count": len(align_result.matches) + len(align_result.unmatched_mechs),
            "matched_count": len(align_result.matches),
            "unmatched_rect_count": len(align_result.unmatched_rects),
            "unmatched_mechanical_count": len(align_result.unmatched_mechs),
            "rmse_um": align_result.rmse_um,
            "max_residual_um": align_result.max_residual_um,
            "x_reversed": align_result.x_reversed,
            "y_reversed": align_result.y_reversed,
        },
        "records": records,
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/result":
            self.send_json(LAST_RESULT)
            return
        if parsed.path == "/":
            target = STATIC_DIR / "index.html"
        elif parsed.path.startswith("/static/"):
            target = STATIC_DIR / parsed.path.removeprefix("/static/")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.resolve().is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        global LAST_RESULT
        if urlparse(self.path).path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            LAST_RESULT = run_match(payload)
            self.send_json(LAST_RESULT)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize MSK rectangle detection and TXT alignment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Rect Match Viewer: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
