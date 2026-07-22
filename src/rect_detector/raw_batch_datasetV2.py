from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset

from rect_detector.align_chip_rects import MechanicalInfo, parse_mechanical_txt


IMAGE_WIDTH = 5120
IMAGE_HEIGHT = 5120
_IMAGE3_RE = re.compile(r"^IMAGE3_(\d+)\.raw$")


def read_gray_5120(path: str | Path) -> np.ndarray:
    """Read a headerless 5120x5120 uint8/uint16 file as uint8 grayscale."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file does not exist: {path}")

    pixel_count = IMAGE_WIDTH * IMAGE_HEIGHT
    file_size = path.stat().st_size
    if file_size == pixel_count:
        # image_u8 = np.fromfile(path, dtype=np.uint8)
        return np.fromfile(path, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    elif file_size == pixel_count * np.dtype(np.uint16).itemsize:
        image_u16 = np.fromfile(path, dtype=np.uint16)
        # Shift in place to avoid creating an extra temporary uint16 array.
        np.right_shift(image_u16, 8, out=image_u16)
        image_u8 = image_u16.astype(np.uint8, copy=False)
        return image_u8.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        raise ValueError(
            f"unexpected file size for {path}: {file_size} bytes; "
            f"expected {pixel_count} bytes (uint8) or {pixel_count * 2} bytes (uint16)"
        )


@dataclass(frozen=True)
class RawBatchSample:
    sample_id: int
    num_str: str
    rect_input_path: Path
    txt_path: Path
    light_1_path: Path
    light_2_path: Path
    light_3_path: Path
    light_4_path: Path


@lru_cache(maxsize=2048)
def _load_mechanical_infos_cached(txt_path: str) -> tuple[MechanicalInfo, ...]:
    return tuple(parse_mechanical_txt(Path(txt_path)))


class RawBatchDataset(Dataset):
    """Dataset indexed by IMAGE3_{num_str}.raw and its sibling Light1-4 files.

    __getitem__(idx) returns one sample dictionary:
    - rect_input_img: uint8 HxW array from Light1-raw/IMAGE1_{num_str}.msk
    - txt: Path to Light1-raw/IMAGE1_{num_str}.txt
    - mechanical_infos: cached tuple[MechanicalInfo, ...] parsed from txt
    - light_images: dict with light_1/light_2/light_3/light_4 uint8 HxW arrays
    """

    def __init__(
        self,
        root_dir: str | Path,
        strict: bool = True,
        light_read_workers: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.strict = bool(strict)
        self.light_read_workers = max(1, int(light_read_workers))
        self.light1_dir = self.root_dir / "Light1-raw"
        self.light2_dir = self.root_dir / "Light2-raw"
        self.light3_dir = self.root_dir / "Light3-raw"
        self.light4_dir = self.root_dir / "Light4-raw"

        self.samples = self._discover_samples()

    def _discover_samples(self) -> list[RawBatchSample]:
        if not self.light3_dir.is_dir():
            raise FileNotFoundError(f"Light3-raw directory does not exist: {self.light3_dir}")

        candidates: list[tuple[int, str, Path]] = []
        for path in self.light3_dir.glob("IMAGE3_*.raw"):
            match = _IMAGE3_RE.match(path.name)
            if match is None:
                continue
            num_str = match.group(1)
            candidates.append((int(num_str), num_str, path))

        candidates.sort(key=lambda item: item[0])
        samples: list[RawBatchSample] = []

        for sample_id, num_str, light3_path in candidates:
            rect_input_path = self.light1_dir / f"IMAGE1_{num_str}.msk"
            txt_path = self.light1_dir / f"IMAGE1_{num_str}.txt"
            light1_path = self.light1_dir / f"IMAGE1_{num_str}.raw"
            light2_path = self.light2_dir / f"IMAGE2_{num_str}.raw"
            light4_path = self.light4_dir / f"IMAGE4_{num_str}.raw"

            required_paths = {
                "rect_input_path": rect_input_path,
                "txt_path": txt_path,
                "light_1_path": light1_path,
                "light_2_path": light2_path,
                "light_3_path": light3_path,
                "light_4_path": light4_path,
            }
            if self.strict:
                missing = [f"{name}={path}" for name, path in required_paths.items() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"missing files for sample IMAGE3_{num_str}: " + ", ".join(missing)
                    )

            samples.append(
                RawBatchSample(
                    sample_id=sample_id,
                    num_str=num_str,
                    rect_input_path=rect_input_path,
                    txt_path=txt_path,
                    light_1_path=light1_path,
                    light_2_path=light2_path,
                    light_3_path=light3_path,
                    light_4_path=light4_path,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sample_images(self, sample: RawBatchSample) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        image_paths = {
            "rect_input": sample.rect_input_path,
            "light_1": sample.light_1_path,
            "light_2": sample.light_2_path,
            "light_3": sample.light_3_path,
            "light_4": sample.light_4_path,
        }
        if self.light_read_workers <= 1:
            images = {
                name: read_gray_5120(path)
                for name, path in image_paths.items()
            }
        else:
            # Read rect input and all light images in parallel to reduce per-sample IO wall time.
            with ThreadPoolExecutor(max_workers=min(self.light_read_workers, len(image_paths))) as pool:
                arrays = list(pool.map(read_gray_5120, image_paths.values()))
            images = dict(zip(image_paths.keys(), arrays))

        rect_input_img = images.pop("rect_input")
        return rect_input_img, images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        mechanical_infos = _load_mechanical_infos_cached(str(sample.txt_path))
        if not mechanical_infos:
            return {
                "sample_id": sample.sample_id,
                "num_str": sample.num_str,
                "rect_input_img": np.empty((0, 0), dtype=np.uint8),
                "txt": sample.txt_path,
                "mechanical_infos": mechanical_infos,
                "light_images": {
                    "light_1": np.empty((0, 0), dtype=np.uint8),
                    "light_2": np.empty((0, 0), dtype=np.uint8),
                    "light_3": np.empty((0, 0), dtype=np.uint8),
                    "light_4": np.empty((0, 0), dtype=np.uint8),
                },
            }

        rect_input_img, light_images = self._read_sample_images(sample)
        return {
            "sample_id": sample.sample_id,
            "num_str": sample.num_str,
            "rect_input_img": rect_input_img,
            "txt": sample.txt_path,
            "mechanical_infos": mechanical_infos,
            "light_images": light_images,
        }


def raw_batch_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Collate a list of dataset samples into dict-of-lists."""
    output: dict[str, list[Any]] = {
        "sample_ids": [],
        "num_strs": [],
        "rect_input_imgs": [],
        "txts": [],
        "mechanical_infos": [],
        "light_images": [],
    }
    for sample in batch:
        output["sample_ids"].append(sample["sample_id"])
        output["num_strs"].append(sample["num_str"])
        output["rect_input_imgs"].append(sample["rect_input_img"])
        output["txts"].append(sample["txt"])
        output["mechanical_infos"].append(sample.get("mechanical_infos", ()))
        output["light_images"].append(sample["light_images"])
    return output


def build_raw_batch_dataloader(
    root_dir: str | Path,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    strict: bool = True,
    light_read_workers: int = 1,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader that yields dicts of lists for batch inference."""
    dataset = RawBatchDataset(
        root_dir=root_dir,
        strict=strict,
        light_read_workers=light_read_workers,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "collate_fn": raw_batch_collate_fn,
        **dataloader_kwargs,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def benchmark_raw_batch_dataloader(
    root_dir: str | Path,
    batch_size: int = 2,
    num_batches: int = 5,
    num_workers: int = 0,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
    shuffle: bool = False,
    strict: bool = True,
    light_read_workers: int = 1,
) -> None:
    """Run a small loader benchmark and print timing plus batch structure."""
    dataloader = build_raw_batch_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        strict=strict,
        light_read_workers=light_read_workers,
    )

    start_time = time.perf_counter()
    processed_batches = 0
    processed_samples = 0

    for batch_index, batch in enumerate(dataloader):
        rect_input_imgs = list(batch["rect_input_imgs"])
        mechanical_infos = list(batch.get("mechanical_infos", [Path(txt) for txt in batch["txts"]]))
        light_image_batches = list(batch["light_images"])
        processed_batches += 1
        processed_samples += len(rect_input_imgs)

        if batch_index == 0:
            print("Batch structure")
            print(f"  keys: {sorted(batch.keys())}")
            print(f"  rect_input_imgs: {len(rect_input_imgs)}")
            print(f"  txts: {len(batch['txts'])}")
            print(f"  mechanical_infos: {len(mechanical_infos)}")
            print(f"  light_image_batches: {len(light_image_batches)}")
            if rect_input_imgs and rect_input_imgs[0] is not None:
                print(f"  rect_input_img[0].shape: {rect_input_imgs[0].shape}, dtype={rect_input_imgs[0].dtype}")
            if light_image_batches:
                first_light_batch = light_image_batches[0]
                print(f"  first light batch keys: {sorted(first_light_batch.keys())}")
                print(
                    f"  light_1 shape: {first_light_batch['light_1'].shape}, dtype={first_light_batch['light_1'].dtype}"
                )

        if processed_batches >= num_batches:
            break

    elapsed_s = time.perf_counter() - start_time
    samples_per_second = processed_samples / elapsed_s if elapsed_s > 0 else float("inf")
    print("Benchmark summary")
    print(f"  batches: {processed_batches}")
    print(f"  samples: {processed_samples}")
    print(f"  elapsed_s: {elapsed_s:.3f}")
    print(f"  samples_per_second: {samples_per_second:.2f}")


__all__ = [
    "benchmark_raw_batch_dataloader",
    "RawBatchDataset",
    "RawBatchSample",
    "build_raw_batch_dataloader",
    "raw_batch_collate_fn",
    "read_gray_5120",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the raw batch dataset and dataloader.")
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Dataset root directory, e.g. /workspace/.../48AMA/imgs/S26F20082-02",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
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
        default=1,
        help="Per-sample thread count for parallel light image reads (1 disables).",
    )
    args = parser.parse_args()

    benchmark_raw_batch_dataloader(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=args.shuffle,
        strict=not args.unsafe_missing_ok,
        light_read_workers=args.light_read_workers,
    )

    # python Detectors/rect_detector/raw_batch_dataset2.py 48AMA/imgs/S26F20082-02 --batch-size 2 --num-batches 121 --num-workers 2 --persistent-workers --prefetch-factor 2 --light-read-workers 4