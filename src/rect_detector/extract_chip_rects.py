from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from rect_detector.align_chip_rects import AlignResult, align_rects_to_mechanical_txt
from rect_detector.rect_models import Box, Rect_Info


ImgInput = str | Path | Image.Image | np.ndarray


def _as_cv_image(img: ImgInput) -> np.ndarray:
    """Return an OpenCV image: uint8 grayscale or BGR."""
    if isinstance(img, Image.Image):
        rgb = np.asarray(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(img, np.ndarray):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (3, 4)):
            return np.ascontiguousarray(img)
        raise ValueError("numpy img must be HxW, HxWx3 (BGR), or HxWx4 (BGRA)")

    image = cv2.imread(str(Path(img)), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image: {img}")
    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    image = _to_uint8(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    image = _to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _scale_int(value: int | float, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * scale)))


def _box_from_xyxy(x1: int, y1: int, x2: int, y2: int) -> Box:
    return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1, x2=x2, y2=y2)


class RectInferer:
    """Detect complete chip rectangles and draw annotations."""

    def __init__(
        self,
        scale: float = 0.5,
        threshold: int = 18,
        x_dilate: int = 45,
        y_dilate: int = 14,
        min_width: int = 600,
        max_width: int = 950,
        min_height: int = 90,
        max_height: int = 220,
        min_aspect: float = 3.6,
        max_aspect: float = 8.5,
        min_area: int = 2500,
        margin: int = 100,
        mech_delta_x: float | None = None,
        mech_delta_y: float | None = None,
        delta_x_pixel: float | None = 950.0,
        delta_y_pixel: float | None = 185.0,
        x_scale: float | None = None,
        y_scale: float | None = None,
    ) -> None:
        resolved_x_scale = scale if x_scale is None else x_scale
        resolved_y_scale = scale if y_scale is None else y_scale
        if scale <= 0 or resolved_x_scale <= 0 or resolved_y_scale <= 0:
            raise ValueError("scale, x_scale, and y_scale must be positive")
        if (mech_delta_x is None) != (mech_delta_y is None):
            raise ValueError("mech_delta_x and mech_delta_y must be provided together, or both be None")
        if (delta_x_pixel is None) != (delta_y_pixel is None):
            raise ValueError("delta_x_pixel and delta_y_pixel must be provided together, or both be None")
        if delta_x_pixel is not None and (delta_x_pixel <= 0 or delta_y_pixel <= 0):
            raise ValueError("delta_x_pixel and delta_y_pixel must be positive")
        self.scale = scale
        self.x_scale = resolved_x_scale
        self.y_scale = resolved_y_scale
        self.threshold = threshold
        self.x_dilate = x_dilate
        self.y_dilate = y_dilate
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.margin = margin
        self.mech_delta_x = mech_delta_x
        self.mech_delta_y = mech_delta_y
        self.delta_x_pixel = delta_x_pixel
        self.delta_y_pixel = delta_y_pixel

    def __call__(self, input_img: ImgInput) -> list[Rect_Info]:
        original = _as_cv_image(input_img)
        original_h, original_w = original.shape[:2]
        detect_img = self._resize_for_detection(original)
        scaled_rects = self._detect_on_scaled_image(detect_img)

        rect_infos: list[Rect_Info] = []
        for idx, (scaled_box, area, aspect) in enumerate(scaled_rects, 1):
            box = self._scaled_box_to_original_box(scaled_box, (original_w, original_h))
            rect_infos.append(
                Rect_Info(
                    id=idx,
                    box=box,
                    scaled_box=scaled_box,
                    area=area,
                    aspect=round(aspect, 4),
                )
            )
        return rect_infos

    def align_rect(
        self,
        input_img: ImgInput,
        mechanical_info: str | Path,
        mech_delta_x: float | None = None,
        mech_delta_y: float | None = None,
        delta_x_pixel: float | None = None,
        delta_y_pixel: float | None = None,
        allow_partial: bool = False,
        allow_x_reverse: bool = False,
        allow_y_reverse: bool = False,
    ) -> AlignResult:
        """
        Detect rects and align them to mechanical-coordinate TXT information.

        mechanical_info can be either a txt file path or already-read txt
        content. The returned object is align_chip_rects.AlignResult.
        """
        delta_x = self.mech_delta_x if mech_delta_x is None else mech_delta_x
        delta_y = self.mech_delta_y if mech_delta_y is None else mech_delta_y
        pixel_delta_x = self.delta_x_pixel if delta_x_pixel is None else delta_x_pixel
        pixel_delta_y = self.delta_y_pixel if delta_y_pixel is None else delta_y_pixel
        rects = self(input_img)
        return align_rects_to_mechanical_txt(
            rects,
            mechanical_info,
            mech_delta_x=delta_x,
            mech_delta_y=delta_y,
            delta_x_pixel=pixel_delta_x,
            delta_y_pixel=pixel_delta_y,
            allow_partial=allow_partial,
            allow_x_reverse=allow_x_reverse,
            allow_y_reverse=allow_y_reverse,
        )

    def draw(
        self,
        input_img: ImgInput,
        rects: list[Rect_Info] | None = None,
        align: bool = False,
        mechanical_info: str | Path | None = None,
        align_result: AlignResult | None = None,
        mech_delta_x: float | None = None,
        mech_delta_y: float | None = None,
        delta_x_pixel: float | None = None,
        delta_y_pixel: float | None = None,
        allow_partial: bool = False,
        allow_x_reverse: bool = False,
        allow_y_reverse: bool = False,
        outline_color: tuple[int, int, int] = (0, 0, 255),
        outline_width: int = 4,
        show_index: bool = True,
        show_mechanical_xy: bool = True,
    ) -> np.ndarray:
        image = _as_cv_image(input_img)

        if align_result is not None:
            rect_infos = [match.rect for match in align_result.matches] + list(align_result.unmatched_rects)
        else:
            start = time.time()
            rect_infos = rects if rects is not None else self(input_img)
            print(f"Detection took {time.time() - start:.2f}s")
            start = time.time()
            if align:
                if mechanical_info is None:
                    raise ValueError("mechanical_info is required when align=True and align_result is not provided")
                delta_x = self.mech_delta_x if mech_delta_x is None else mech_delta_x
                delta_y = self.mech_delta_y if mech_delta_y is None else mech_delta_y
                pixel_delta_x = self.delta_x_pixel if delta_x_pixel is None else delta_x_pixel
                pixel_delta_y = self.delta_y_pixel if delta_y_pixel is None else delta_y_pixel
                align_result = align_rects_to_mechanical_txt(
                    rect_infos,
                    mechanical_info,
                    mech_delta_x=delta_x,
                    mech_delta_y=delta_y,
                    delta_x_pixel=pixel_delta_x,
                    delta_y_pixel=pixel_delta_y,
                    allow_partial=allow_partial,
                    allow_x_reverse=allow_x_reverse,
                    allow_y_reverse=allow_y_reverse,
                )
            print(f"Alignment took {time.time() - start:.2f}s")
        mech_by_rect_id = {}
        if align and align_result is not None:
            mech_by_rect_id = {match.rect.id: match.mech for match in align_result.matches}

        annotated = _to_bgr(image)
        image_h = annotated.shape[0]
        for rect in rect_infos:
            box = rect.box
            cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), outline_color, outline_width)
            if show_index:
                cv2.putText(
                    annotated, str(rect.id), (box.x, max(14, box.y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, outline_color, 1, cv2.LINE_AA,
                )
            if show_mechanical_xy and rect.id in mech_by_rect_id:
                mech = mech_by_rect_id[rect.id]
                label = f"({mech.MX:.0f}, {mech.MY:.0f})"
                cv2.putText(
                    annotated, label, (box.x, min(image_h - 5, box.y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 1, cv2.LINE_AA,
                )
        return annotated

    def _resize_for_detection(self, image: np.ndarray) -> np.ndarray:
        if self.x_scale == 1 and self.y_scale == 1:
            return image
        height, width = image.shape[:2]
        scaled_size = (
            max(1, int(round(width * self.x_scale))),
            max(1, int(round(height * self.y_scale))),
        )
        return cv2.resize(image, scaled_size, interpolation=cv2.INTER_NEAREST)

    def _scaled_params(self) -> dict[str, int | float]:
        return {
            "x_dilate": _scale_int(self.x_dilate, self.x_scale),
            "y_dilate": _scale_int(self.y_dilate, self.y_scale),
            "min_width": _scale_int(self.min_width, self.x_scale),
            "max_width": _scale_int(self.max_width, self.x_scale),
            "min_height": _scale_int(self.min_height, self.y_scale),
            "max_height": _scale_int(self.max_height, self.y_scale),
            "min_area": _scale_int(self.min_area, self.x_scale * self.y_scale),
            "margin_x": _scale_int(self.margin, self.x_scale, minimum=10),
            "margin_y": _scale_int(self.margin, self.y_scale, minimum=10),
        }

    def _detect_on_scaled_image(self, image: np.ndarray) -> list[tuple[Box, int, float]]:
        params = self._scaled_params()
        gray = _to_gray(image)
        _, foreground = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        fg_x, fg_y, fg_w, fg_h = cv2.boundingRect(foreground)
        if fg_w == 0 or fg_h == 0:
            return []

        margin_x = int(params["margin_x"])
        margin_y = int(params["margin_y"])
        crop_x1 = max(0, fg_x - margin_x)
        crop_y1 = max(0, fg_y - margin_y)
        crop_x2 = min(gray.shape[1], fg_x + fg_w + margin_x)
        crop_y2 = min(gray.shape[0], fg_y + fg_h + margin_y)

        cropped_foreground = foreground[crop_y1:crop_y2, crop_x1:crop_x2]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2 * int(params["x_dilate"]) + 1, 2 * int(params["y_dilate"]) + 1),
        )
        grouped_mask = cv2.dilate(cropped_foreground, kernel, iterations=1)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(grouped_mask, connectivity=8)

        rects: list[tuple[Box, int, float]] = []
        for label in range(1, component_count):
            gx1 = int(stats[label, cv2.CC_STAT_LEFT])
            gy1 = int(stats[label, cv2.CC_STAT_TOP])
            gw = int(stats[label, cv2.CC_STAT_WIDTH])
            gh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            gx2, gy2 = gx1 + gw, gy1 + gh
            original_region = cropped_foreground[gy1:gy2, gx1:gx2]
            rx, ry, rw, rh = cv2.boundingRect(original_region)
            if rw == 0 or rh == 0:
                continue

            x1 = rx + gx1 + crop_x1
            y1 = ry + gy1 + crop_y1
            x2 = x1 + rw
            y2 = y1 + rh
            box = _box_from_xyxy(x1, y1, x2, y2)
            aspect = (box.w / self.x_scale) / max(box.h / self.y_scale, 1e-12)

            if not (int(params["min_width"]) <= box.w <= int(params["max_width"])):
                continue
            if not (int(params["min_height"]) <= box.h <= int(params["max_height"])):
                continue
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            if area < int(params["min_area"]):
                continue

            rects.append((box, int(area), aspect))

        return sorted(rects, key=lambda item: (item[0].y, item[0].x))

    def _scaled_box_to_original_box(self, box: Box, original_size: tuple[int, int]) -> Box:
        if self.x_scale == 1 and self.y_scale == 1:
            return box
        original_w, original_h = original_size
        x1 = max(0, min(original_w, int(round(box.x / self.x_scale))))
        y1 = max(0, min(original_h, int(round(box.y / self.y_scale))))
        x2 = max(0, min(original_w, int(round(box.x2 / self.x_scale))))
        y2 = max(0, min(original_h, int(round(box.y2 / self.y_scale))))
        return _box_from_xyxy(x1, y1, x2, y2)


def detect_chip_rects(img: ImgInput, **kwargs) -> list[Rect_Info]:
    return RectInferer(**kwargs)(img)


def annotate_chip_rects(img: ImgInput, **kwargs) -> np.ndarray:
    return RectInferer(**kwargs).draw(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and annotate complete chip rectangles.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("annotated_chips.png"))
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--x-scale", type=float, default=None)
    parser.add_argument("--y-scale", type=float, default=None)
    parser.add_argument("--threshold", type=int, default=18)
    parser.add_argument("--x-dilate", type=int, default=45)
    parser.add_argument("--y-dilate", type=int, default=14)
    parser.add_argument("--align-txt", type=Path, default=None)
    parser.add_argument("--mech-delta-x", type=float, default=None)
    parser.add_argument("--mech-delta-y", type=float, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-x-reverse", action="store_true")
    parser.add_argument("--allow-y-reverse", action="store_true")
    args = parser.parse_args()

    inferer = RectInferer(
        scale=args.scale,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        threshold=args.threshold,
        x_dilate=args.x_dilate,
        y_dilate=args.y_dilate,
        mech_delta_x=args.mech_delta_x,
        mech_delta_y=args.mech_delta_y,
    )
    rects = inferer(args.image)
    if args.align_txt is not None:
        align_result = inferer.align_rect(
            args.image,
            args.align_txt,
            allow_partial=args.allow_partial,
            allow_x_reverse=args.allow_x_reverse,
            allow_y_reverse=args.allow_y_reverse,
        )
        annotated = inferer.draw(args.image, align=True, align_result=align_result)
        print(f"aligned rmse_um={align_result.rmse_um:.3f}, max_residual_um={align_result.max_residual_um:.3f}")
        if align_result.is_partial:
            print(
                "partial alignment "
                f"matches={len(align_result.matches)}, "
                f"unmatched_rects={len(align_result.unmatched_rects)}, "
                f"unmatched_mechs={len(align_result.unmatched_mechs)}"
            )
        if align_result.used_delta_prior:
            print(
                "delta prior "
                f"dx={align_result.mech_delta_x:.3f}, dy={align_result.mech_delta_y:.3f}, "
                f"penalty_um={align_result.delta_penalty_um:.3f}"
            )
    else:
        annotated = inferer.draw(args.image, rects=rects)
    if not cv2.imwrite(str(args.out), annotated):
        raise OSError(f"failed to save image: {args.out}")

    print(f"detected {len(rects)} complete chips")
    print(f"saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
