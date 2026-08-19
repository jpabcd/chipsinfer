from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_raw_file_and_save_png(
	raw_path: str | Path,
	png_path: str | Path,
	width: int = 5120,
	height: int = 5120,
	dtype: str = "auto",
) -> Path:
	"""Read a headerless raw grayscale file and save it as a PNG image.

	Args:
		raw_path: Input raw file path.
		png_path: Output png file path.
		width: Image width.
		height: Image height.
		dtype: "auto", "uint8", or "uint16".

	Returns:
		Resolved output PNG path.
	"""
	raw_path = Path(raw_path)
	png_path = Path(png_path)

	if not raw_path.is_file():
		raise FileNotFoundError(f"raw file does not exist: {raw_path}")

	if width <= 0 or height <= 0:
		raise ValueError(f"width/height must be positive, got width={width}, height={height}")

	pixel_count = width * height
	file_size = raw_path.stat().st_size

	if dtype not in {"auto", "uint8", "uint16"}:
		raise ValueError("dtype must be one of: auto, uint8, uint16")

	if dtype == "auto":
		if file_size == pixel_count:
			resolved_dtype = np.uint8
		elif file_size == pixel_count * np.dtype(np.uint16).itemsize:
			resolved_dtype = np.uint16
		else:
			raise ValueError(
				f"unexpected file size for {raw_path}: {file_size} bytes; "
				f"expected {pixel_count} (uint8) or {pixel_count * 2} (uint16)"
			)
	else:
		resolved_dtype = np.uint8 if dtype == "uint8" else np.uint16
		expected_size = pixel_count * np.dtype(resolved_dtype).itemsize
		if file_size != expected_size:
			raise ValueError(
				f"file size mismatch for dtype={dtype}: actual {file_size} bytes, "
				f"expected {expected_size} bytes"
			)

	image = np.fromfile(raw_path, dtype=resolved_dtype)
	image = image.reshape(height, width)

	if resolved_dtype == np.uint16:
		# Convert to 8-bit grayscale using high-byte extraction.
		np.right_shift(image, 8, out=image)
		image = image.astype(np.uint8, copy=False)

	png_path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(image, mode="L").save(png_path)
	return png_path.resolve()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Read a raw file and save as PNG.")
	parser.add_argument("raw_path", type=Path, help="Input raw file path")
	parser.add_argument("png_path", type=Path, help="Output png file path")
	parser.add_argument("--width", type=int, default=5120, help="Image width")
	parser.add_argument("--height", type=int, default=5120, help="Image height")
	parser.add_argument(
		"--dtype",
		choices=["auto", "uint8", "uint16"],
		default="auto",
		help="Raw pixel dtype",
	)

	args = parser.parse_args()
	out = read_raw_file_and_save_png(
		raw_path=args.raw_path,
		png_path=args.png_path,
		width=args.width,
		height=args.height,
		dtype=args.dtype,
	)
	print(out)
