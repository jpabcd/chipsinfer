from __future__ import annotations

from collections.abc import Iterable as IterableABC
from math import comb
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from rect_detector.rect_models import Rect_Info


@dataclass(frozen=True)
class MechanicalInfo:
    MX: float
    MY: float
    nBin: int
    nImageNum: int
    nIndex: int
    nDefectType: int
    nMultiDefectCount: int
    bIsNeedle: int
    fScore: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignedRect:
    rect: Rect_Info
    mech: MechanicalInfo
    predicted_MX: float
    predicted_MY: float
    residual_um: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rect"] = self.rect.to_dict()
        data["mech"] = self.mech.to_dict()
        return data


@dataclass(frozen=True)
class AlignResult:
    matches: list[AlignedRect]
    unmatched_rects: list[Rect_Info]
    unmatched_mechs: list[MechanicalInfo]
    affine_pixel_to_um: np.ndarray
    rmse_um: float
    max_residual_um: float
    x_reversed: bool
    y_reversed: bool
    is_partial: bool = False
    used_delta_prior: bool = False
    mech_delta_x: float | None = None
    mech_delta_y: float | None = None
    delta_penalty_um: float = 0.0


def parse_mechanical_txt(txt: str | Path) -> list[MechanicalInfo]:
    """
    Parse the mechanical-coordinate txt table.

    The input may be a file path or the raw txt content. Required columns are:
    MX MY nBin nImageNum nIndex nDefectType nMultiDefectCount bIsNeedle fScore
    """
    if isinstance(txt, Path):
        content = Path(txt).read_text(encoding="utf-8")
    elif isinstance(txt, str) and "\n" not in txt and "\r" not in txt and Path(txt).exists():
        content = Path(txt).read_text(encoding="utf-8")
    else:
        content = str(txt)

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []

    header_idx = None
    for idx, line in enumerate(lines):
        cols = line.split()
        if len(cols) >= 9 and cols[0] == "MX" and cols[1] == "MY":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("mechanical txt header not found")

    records: list[MechanicalInfo] = []
    for line in lines[header_idx + 1 :]:
        cols = line.split()
        if len(cols) < 9:
            continue
        try:
            records.append(
                MechanicalInfo(
                    MX=float(cols[0]),
                    MY=float(cols[1]),
                    nBin=int(cols[2]),
                    nImageNum=int(cols[3]),
                    nIndex=int(cols[4]),
                    nDefectType=int(cols[5]),
                    nMultiDefectCount=int(cols[6]),
                    bIsNeedle=int(cols[7]),
                    fScore=float(cols[8]),
                )
            )
        except ValueError:
            continue
    return records


def _coerce_mechanical_records(
    mechanical_txt: str | Path | MechanicalInfo | Iterable[MechanicalInfo],
) -> list[MechanicalInfo]:
    if isinstance(mechanical_txt, MechanicalInfo):
        return [mechanical_txt]
    if isinstance(mechanical_txt, IterableABC) and not isinstance(mechanical_txt, (str, bytes, Path)):
        mech_list = list(mechanical_txt)
        if not mech_list:
            return []
        if all(isinstance(item, MechanicalInfo) for item in mech_list):
            return mech_list
    return parse_mechanical_txt(mechanical_txt)


def align_rects_to_mechanical_txt(
    rects: Iterable[Rect_Info],
    mechanical_txt: str | Path | MechanicalInfo | Iterable[MechanicalInfo],
    mech_delta_x: float | None = None,
    mech_delta_y: float | None = None,
    delta_x_pixel: float | None = None,
    delta_y_pixel: float | None = None,
    delta_weight: float = 0.01,
    allow_partial: bool = False,
    allow_x_reverse: bool = False,
    allow_y_reverse: bool = False,
) -> AlignResult:
    """
    Match detected image rects to mechanical-coordinate records.

    Matching is based on relative layout. It does not use absolute pixel/um
    origin or unit equality. By default, it assumes MX increases from image
    left to right and MY increases from image top to bottom. If
    mech_delta_x/mech_delta_y are provided, they are used as a soft
    mechanical-grid prior.
    """
    if (mech_delta_x is None) != (mech_delta_y is None):
        raise ValueError("mech_delta_x and mech_delta_y must be provided together, or both be None")
    if mech_delta_x is not None and (mech_delta_x <= 0 or mech_delta_y <= 0):
        raise ValueError("mech_delta_x and mech_delta_y must be positive")
    if (delta_x_pixel is None) != (delta_y_pixel is None):
        raise ValueError("delta_x_pixel and delta_y_pixel must be provided together, or both be None")
    if delta_x_pixel is not None and (delta_x_pixel <= 0 or delta_y_pixel <= 0):
        raise ValueError("delta_x_pixel and delta_y_pixel must be positive")

    rect_list = list(rects)
    mech_list = _coerce_mechanical_records(mechanical_txt)
    if len(rect_list) != len(mech_list) and not allow_partial:
        raise ValueError(f"rect count ({len(rect_list)}) != mechanical count ({len(mech_list)})")
    if not rect_list:
        raise ValueError("no rects to align")
    if not mech_list:
        raise ValueError("no mechanical records to align")

    rect_points = np.array([_rect_center(rect) for rect in rect_list], dtype=np.float64)
    mech_points = np.array([(item.MX, item.MY) for item in mech_list], dtype=np.float64)

    partial_candidate = None
    if allow_partial and len(rect_list) != len(mech_list):
        partial_candidate = _try_column_constrained_partial_alignment(
            rect_points=rect_points,
            mech_points=mech_points,
            mech_delta_x=mech_delta_x,
            mech_delta_y=mech_delta_y,
            delta_x_pixel=delta_x_pixel,
            delta_y_pixel=delta_y_pixel,
            delta_weight=delta_weight,
            allow_x_reverse=allow_x_reverse,
            allow_y_reverse=allow_y_reverse,
        )

    if partial_candidate is not None:
        score, rmse, delta_penalty, pairs, affine, residuals, x_reversed, y_reversed = partial_candidate
    else:
        candidates = _build_alignment_candidates(
            rect_points=rect_points,
            mech_points=mech_points,
            mech_delta_x=mech_delta_x,
            mech_delta_y=mech_delta_y,
            delta_x_pixel=delta_x_pixel,
            delta_y_pixel=delta_y_pixel,
            delta_weight=delta_weight,
            allow_partial=allow_partial,
            allow_x_reverse=allow_x_reverse,
            allow_y_reverse=allow_y_reverse,
        )

        if not candidates:
            raise ValueError(
                "could not align by column structure; check whether rect count/column counts match the mechanical txt"
            )

        score, rmse, delta_penalty, pairs, affine, residuals, x_reversed, y_reversed = min(
            candidates, key=lambda item: item[0]
        )

    predicted_all = _apply_affine(rect_points, affine)

    matches: list[AlignedRect] = []
    for pair_idx, (rect_idx, mech_idx) in enumerate(pairs):
        pred_mx, pred_my = predicted_all[rect_idx]
        matches.append(
            AlignedRect(
                rect=rect_list[rect_idx],
                mech=mech_list[mech_idx],
                predicted_MX=float(pred_mx),
                predicted_MY=float(pred_my),
                residual_um=float(residuals[pair_idx]),
            )
        )

    matches.sort(key=lambda item: item.rect.id)
    matched_rect_indices = {rect_idx for rect_idx, _ in pairs}
    matched_mech_indices = {mech_idx for _, mech_idx in pairs}
    unmatched_rects = [rect for idx, rect in enumerate(rect_list) if idx not in matched_rect_indices]
    unmatched_mechs = [mech for idx, mech in enumerate(mech_list) if idx not in matched_mech_indices]

    return AlignResult(
        matches=matches,
        unmatched_rects=unmatched_rects,
        unmatched_mechs=unmatched_mechs,
        affine_pixel_to_um=affine,
        rmse_um=rmse,
        max_residual_um=float(np.max(residuals)),
        x_reversed=x_reversed,
        y_reversed=y_reversed,
        is_partial=bool(unmatched_rects or unmatched_mechs),
        used_delta_prior=mech_delta_x is not None,
        mech_delta_x=mech_delta_x,
        mech_delta_y=mech_delta_y,
        delta_penalty_um=float(delta_penalty),
    )


def _build_alignment_candidates(
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    mech_delta_x: float | None,
    mech_delta_y: float | None,
    delta_x_pixel: float | None,
    delta_y_pixel: float | None,
    delta_weight: float,
    allow_partial: bool,
    allow_x_reverse: bool,
    allow_y_reverse: bool,
) -> list[tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool]]:
    use_partial_matching = allow_partial and len(rect_points) != len(mech_points)
    if delta_x_pixel is None:
        rect_cols = _cluster_indices_1d(rect_points[:, 0])
    else:
        rect_cols = _cluster_indices_by_delta(rect_points[:, 0], delta_x_pixel)
    if mech_delta_x is None:
        mech_cols = _cluster_indices_1d(mech_points[:, 0])
    else:
        mech_cols = _cluster_indices_by_delta(mech_points[:, 0], mech_delta_x)
    candidates: list[tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool]] = []
    x_reverse_options = (False, True) if allow_x_reverse else (False,)
    y_reverse_options = (False, True) if allow_y_reverse else (False,)

    for x_reversed in x_reverse_options:
        ordered_mech_cols = list(reversed(mech_cols)) if x_reversed else mech_cols
        column_pairs = _match_column_groups(
            rect_cols,
            ordered_mech_cols,
            allow_partial=use_partial_matching,
        )
        if column_pairs is None:
            continue

        for y_reversed in y_reverse_options:
            pairs: list[tuple[int, int]] = []
            for rect_group, mech_group in column_pairs:
                rect_order = _sort_indices_by_axis_delta(
                    indices=rect_group,
                    values=rect_points[:, 1],
                    delta=delta_y_pixel,
                )
                mech_order = sorted(mech_group, key=lambda idx: mech_points[idx, 1], reverse=y_reversed)
                if use_partial_matching:
                    pairs.extend(_match_ordered_sequences(rect_order, mech_order))
                elif len(rect_order) == len(mech_order):
                    pairs.extend(zip(rect_order, mech_order))

            if not pairs:
                continue
            candidates.append(
                _score_alignment_pairs(
                    pairs=pairs,
                    rect_cols=rect_cols,
                    ordered_mech_cols=ordered_mech_cols,
                    rect_points=rect_points,
                    mech_points=mech_points,
                    mech_delta_x=mech_delta_x,
                    mech_delta_y=mech_delta_y,
                    delta_weight=delta_weight,
                    x_reversed=x_reversed,
                    y_reversed=y_reversed,
                )
            )

    return candidates


def _try_column_constrained_partial_alignment(
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    mech_delta_x: float | None,
    mech_delta_y: float | None,
    delta_x_pixel: float | None,
    delta_y_pixel: float | None,
    delta_weight: float,
    allow_x_reverse: bool,
    allow_y_reverse: bool,
    max_candidates: int = 2000,
    max_options_per_column: int = 256,
) -> tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool] | None:
    """Search missing positions only inside columns whose counts differ."""
    if delta_x_pixel is None:
        rect_cols = _cluster_indices_1d(rect_points[:, 0])
    else:
        rect_cols = _cluster_indices_by_delta(rect_points[:, 0], delta_x_pixel)
    mech_cols = (
        _cluster_indices_1d(mech_points[:, 0])
        if mech_delta_x is None
        else _cluster_indices_by_delta(mech_points[:, 0], mech_delta_x)
    )
    x_reverse_options = (False, True) if allow_x_reverse else (False,)
    y_reverse_options = (False, True) if allow_y_reverse else (False,)
    best = None

    for x_reversed in x_reverse_options:
        ordered_mech_cols = list(reversed(mech_cols)) if x_reversed else mech_cols
        if len(rect_cols) == len(ordered_mech_cols):
            column_pairs = list(zip(rect_cols, ordered_mech_cols))
        else:
            column_pairs = _match_column_groups(rect_cols, ordered_mech_cols, allow_partial=True)
        if column_pairs is None:
            continue

        for y_reversed in y_reverse_options:
            column_option_sets: list[list[list[tuple[int, int]]]] = []
            for rect_group, mech_group in column_pairs:
                rect_order = _sort_indices_by_axis_delta(
                    indices=rect_group,
                    values=rect_points[:, 1],
                    delta=delta_y_pixel,
                )
                mech_order = sorted(mech_group, key=lambda idx: mech_points[idx, 1], reverse=y_reversed)
                option_count = comb(max(len(rect_order), len(mech_order)), min(len(rect_order), len(mech_order)))
                if option_count > max_options_per_column:
                    # Do not materialize a combinatorial list of candidates.
                    # The caller falls back to the bounded monotonic matcher.
                    return None
                column_options = _ordered_column_pair_options(rect_order, mech_order)
                if not column_options:
                    continue
                column_option_sets.append(column_options)

            stable_selection = _select_column_options_from_stable_pairs(
                column_option_sets=column_option_sets,
                column_pairs=column_pairs,
                rect_points=rect_points,
                mech_points=mech_points,
            )
            if stable_selection is not None:
                pair_candidates = [stable_selection]
            else:
                pair_candidates = _expand_column_options(column_option_sets, max_candidates)
                if pair_candidates is None:
                    return None

            for pairs in pair_candidates:
                if not pairs:
                    continue
                candidate = _score_alignment_pairs(
                    pairs=pairs,
                    rect_cols=rect_cols,
                    ordered_mech_cols=ordered_mech_cols,
                    rect_points=rect_points,
                    mech_points=mech_points,
                    mech_delta_x=mech_delta_x,
                    mech_delta_y=mech_delta_y,
                    delta_weight=delta_weight,
                    x_reversed=x_reversed,
                    y_reversed=y_reversed,
                )
                best = candidate if best is None or candidate[0] < best[0] else best

    if best is not None and mech_delta_x is not None:
        max_reasonable_rmse = 0.1 * min(mech_delta_x, mech_delta_y)
        if best[1] > max_reasonable_rmse:
            return None
    return best


def _select_column_options_from_stable_pairs(
    column_option_sets: list[list[list[tuple[int, int]]]],
    column_pairs: list[tuple[list[int], list[int]]],
    rect_points: np.ndarray,
    mech_points: np.ndarray,
) -> list[tuple[int, int]] | None:
    """Use complete columns to resolve each incomplete column independently."""
    stable_pairs = [pair for options in column_option_sets if len(options) == 1 for pair in options[0]]
    if len(stable_pairs) < 3:
        return None

    stable_rect_indices = np.fromiter((pair[0] for pair in stable_pairs), dtype=np.intp, count=len(stable_pairs))
    stable_mech_indices = np.fromiter((pair[1] for pair in stable_pairs), dtype=np.intp, count=len(stable_pairs))
    stable_source = rect_points[stable_rect_indices]
    design = np.column_stack((stable_source, np.ones(len(stable_source), dtype=np.float64)))
    stable_column_count = sum(len(options) == 1 for options in column_option_sets)
    if stable_column_count >= 2 and np.linalg.matrix_rank(design) >= 3:
        initial_affine = _fit_affine(stable_source, mech_points[stable_mech_indices])
    else:
        initial_affine = _fit_axis_aligned_initial_affine(
            stable_rect_indices=stable_rect_indices,
            stable_mech_indices=stable_mech_indices,
            column_pairs=column_pairs,
            rect_points=rect_points,
            mech_points=mech_points,
        )
        if initial_affine is None:
            return None
    selected: list[tuple[int, int]] = []
    for options in column_option_sets:
        if len(options) == 1:
            selected.extend(options[0])
            continue
        best_option = min(
            options,
            key=lambda option: _pair_option_squared_error(option, rect_points, mech_points, initial_affine),
        )
        selected.extend(best_option)
    return selected


def _fit_axis_aligned_initial_affine(
    stable_rect_indices: np.ndarray,
    stable_mech_indices: np.ndarray,
    column_pairs: list[tuple[list[int], list[int]]],
    rect_points: np.ndarray,
    mech_points: np.ndarray,
) -> np.ndarray | None:
    rect_column_x = np.array([np.mean(rect_points[group, 0]) for group, _ in column_pairs], dtype=np.float64)
    mech_column_x = np.array([np.mean(mech_points[group, 0]) for _, group in column_pairs], dtype=np.float64)
    stable_rect_y = rect_points[stable_rect_indices, 1]
    stable_mech_y = mech_points[stable_mech_indices, 1]
    if len(rect_column_x) < 2 or np.ptp(rect_column_x) == 0 or np.ptp(stable_rect_y) == 0:
        return None

    x_scale, x_offset = np.polyfit(rect_column_x, mech_column_x, 1)
    y_scale, y_offset = np.polyfit(stable_rect_y, stable_mech_y, 1)
    return np.array(
        [[x_scale, 0.0], [0.0, y_scale], [x_offset, y_offset]],
        dtype=np.float64,
    )


def _pair_option_squared_error(
    pairs: list[tuple[int, int]],
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    affine: np.ndarray,
) -> float:
    rect_indices = np.fromiter((pair[0] for pair in pairs), dtype=np.intp, count=len(pairs))
    mech_indices = np.fromiter((pair[1] for pair in pairs), dtype=np.intp, count=len(pairs))
    difference = _apply_affine(rect_points[rect_indices], affine) - mech_points[mech_indices]
    return float(np.sum(difference * difference))


def _expand_column_options(
    column_option_sets: list[list[list[tuple[int, int]]]],
    max_candidates: int,
) -> list[list[tuple[int, int]]] | None:
    candidates: list[list[tuple[int, int]]] = [[]]
    for options in column_option_sets:
        expanded: list[list[tuple[int, int]]] = []
        for existing in candidates:
            for option in options:
                expanded.append(existing + option)
                if len(expanded) > max_candidates:
                    return None
        candidates = expanded
    return candidates


def _ordered_column_pair_options(
    rect_order: list[int],
    mech_order: list[int],
) -> list[list[tuple[int, int]]]:
    """Enumerate monotonic matches for one paired column."""
    rect_count = len(rect_order)
    mech_count = len(mech_order)
    if rect_count == 0 or mech_count == 0:
        return []
    if rect_count == mech_count:
        return [list(zip(rect_order, mech_order))]
    if rect_count > mech_count:
        return [list(zip(kept_rects, mech_order)) for kept_rects in combinations(rect_order, mech_count)]
    return [list(zip(rect_order, kept_mechs)) for kept_mechs in combinations(mech_order, rect_count)]


def _score_alignment_pairs(
    pairs: list[tuple[int, int]],
    rect_cols: list[list[int]],
    ordered_mech_cols: list[list[int]],
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    mech_delta_x: float | None,
    mech_delta_y: float | None,
    delta_weight: float,
    x_reversed: bool,
    y_reversed: bool,
) -> tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool]:
    rect_indices = np.fromiter((pair[0] for pair in pairs), dtype=np.intp, count=len(pairs))
    mech_indices = np.fromiter((pair[1] for pair in pairs), dtype=np.intp, count=len(pairs))
    source = rect_points[rect_indices]
    target = mech_points[mech_indices]
    affine = _fit_alignment_transform(source, target)
    residuals = np.linalg.norm(_apply_affine(source, affine) - target, axis=1)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    delta_penalty = 0.0
    if mech_delta_x is not None:
        delta_penalty = _delta_layout_penalty(
            pairs=pairs,
            rect_cols=rect_cols,
            ordered_mech_cols=ordered_mech_cols,
            rect_points=rect_points,
            mech_points=mech_points,
            mech_delta_x=mech_delta_x,
            mech_delta_y=mech_delta_y,
            y_reversed=y_reversed,
        )
    unmatched_count = len(rect_points) + len(mech_points) - 2 * len(pairs)
    score = rmse + delta_penalty * delta_weight + unmatched_count * 25.0
    return (score, rmse, delta_penalty, pairs, affine, residuals, x_reversed, y_reversed)


def _try_exhaustive_subset_alignment(
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    mech_delta_x: float | None,
    mech_delta_y: float | None,
    delta_x_pixel: float | None,
    delta_y_pixel: float | None,
    delta_weight: float,
    allow_x_reverse: bool,
    allow_y_reverse: bool,
    max_missing: int = 3,
    max_candidates: int = 2000,
    max_duration_s: float = 2.0,
) -> tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool] | None:
    rect_count = len(rect_points)
    mech_count = len(mech_points)
    missing = abs(rect_count - mech_count)
    if missing == 0 or missing > max_missing:
        return None

    best = None
    tried = 0
    start_time = perf_counter()

    if rect_count > mech_count:
        for keep_rect_indices in combinations(range(rect_count), mech_count):
            if perf_counter() - start_time > max_duration_s:
                return best
            tried += 1
            if tried > max_candidates:
                return best
            keep_rect = list(keep_rect_indices)
            candidates = _build_alignment_candidates(
                rect_points=rect_points[keep_rect],
                mech_points=mech_points,
                mech_delta_x=mech_delta_x,
                mech_delta_y=mech_delta_y,
                delta_x_pixel=delta_x_pixel,
                delta_y_pixel=delta_y_pixel,
                delta_weight=delta_weight,
                allow_partial=False,
                allow_x_reverse=allow_x_reverse,
                allow_y_reverse=allow_y_reverse,
            )
            for candidate in candidates:
                mapped = _map_subset_candidate(candidate, keep_rect, list(range(mech_count)))
                best = mapped if best is None or mapped[0] < best[0] else best
    else:
        for keep_mech_indices in combinations(range(mech_count), rect_count):
            if perf_counter() - start_time > max_duration_s:
                return best
            tried += 1
            if tried > max_candidates:
                return best
            keep_mech = list(keep_mech_indices)
            candidates = _build_alignment_candidates(
                rect_points=rect_points,
                mech_points=mech_points[keep_mech],
                mech_delta_x=mech_delta_x,
                mech_delta_y=mech_delta_y,
                delta_x_pixel=delta_x_pixel,
                delta_y_pixel=delta_y_pixel,
                delta_weight=delta_weight,
                allow_partial=False,
                allow_x_reverse=allow_x_reverse,
                allow_y_reverse=allow_y_reverse,
            )
            for candidate in candidates:
                mapped = _map_subset_candidate(candidate, list(range(rect_count)), keep_mech)
                best = mapped if best is None or mapped[0] < best[0] else best

    return best


def _map_subset_candidate(
    candidate: tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool],
    rect_index_map: list[int],
    mech_index_map: list[int],
) -> tuple[float, float, float, list[tuple[int, int]], np.ndarray, np.ndarray, bool, bool]:
    score, rmse, delta_penalty, pairs, affine, residuals, x_reversed, y_reversed = candidate
    mapped_pairs = [(rect_index_map[rect_idx], mech_index_map[mech_idx]) for rect_idx, mech_idx in pairs]
    return (score, rmse, delta_penalty, mapped_pairs, affine, residuals, x_reversed, y_reversed)


def _rect_center(rect: Rect_Info) -> tuple[float, float]:
    box = rect.box
    return (box.x + box.w / 2.0, box.y + box.h / 2.0)


def _cluster_indices_1d(values: np.ndarray) -> list[list[int]]:
    order = np.argsort(values)
    sorted_values = values[order]
    if len(sorted_values) <= 1:
        return [order.tolist()]

    gaps = np.diff(sorted_values)
    positive_gaps = gaps[gaps > 0]
    if len(positive_gaps) == 0:
        return [order.tolist()]

    positive_gaps = np.sort(positive_gaps)
    lower_half = positive_gaps[: max(1, (len(positive_gaps) + 1) // 2)]
    robust_small_gap = float(np.median(lower_half))
    range_gap = float(sorted_values[-1] - sorted_values[0]) * 0.03
    threshold = max(robust_small_gap * 4.0, range_gap)

    groups: list[list[int]] = [[int(order[0])]]
    for gap, idx in zip(gaps, order[1:]):
        if gap > threshold:
            groups.append([])
        groups[-1].append(int(idx))

    return groups


def _cluster_indices_by_delta(values: np.ndarray, delta: float) -> list[list[int]]:
    min_value = float(np.min(values))
    keys = np.rint((values - min_value) / delta).astype(int)
    grouped: dict[int, list[int]] = {}
    for idx, key in enumerate(keys.tolist()):
        grouped.setdefault(key, []).append(idx)
    return [sorted(group, key=lambda idx: values[idx]) for key, group in sorted(grouped.items())]


def _sort_indices_by_axis_delta(indices: list[int], values: np.ndarray, delta: float | None) -> list[int]:
    if delta is None:
        return sorted(indices, key=lambda idx: values[idx])
    min_value = float(np.min(values[indices]))
    return sorted(
        indices,
        key=lambda idx: (
            int(round((float(values[idx]) - min_value) / delta)),
            float(values[idx]),
        ),
    )


def _match_column_groups(
    rect_cols: list[list[int]],
    mech_cols: list[list[int]],
    allow_partial: bool,
) -> list[tuple[list[int], list[int]]] | None:
    if not allow_partial:
        if len(rect_cols) != len(mech_cols):
            return None
        if [len(group) for group in rect_cols] != [len(group) for group in mech_cols]:
            return None
        return list(zip(rect_cols, mech_cols))

    rows = len(rect_cols)
    cols = len(mech_cols)
    gap_cost = 8.0
    dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    prev: list[list[tuple[int, int] | None]] = [[None for _ in range(cols + 1)] for _ in range(rows + 1)]
    dp[0, 0] = 0.0

    for i in range(rows + 1):
        for j in range(cols + 1):
            if not np.isfinite(dp[i, j]):
                continue
            if i < rows and j < cols:
                cost = abs(len(rect_cols[i]) - len(mech_cols[j]))
                if dp[i, j] + cost < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = dp[i, j] + cost
                    prev[i + 1][j + 1] = (i, j)
            if i < rows:
                cost = gap_cost + len(rect_cols[i])
                if dp[i, j] + cost < dp[i + 1, j]:
                    dp[i + 1, j] = dp[i, j] + cost
                    prev[i + 1][j] = (i, j)
            if j < cols:
                cost = gap_cost + len(mech_cols[j])
                if dp[i, j] + cost < dp[i, j + 1]:
                    dp[i, j + 1] = dp[i, j] + cost
                    prev[i][j + 1] = (i, j)

    pairs: list[tuple[list[int], list[int]]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        step = prev[i][j]
        if step is None:
            return None
        pi, pj = step
        if i == pi + 1 and j == pj + 1:
            pairs.append((rect_cols[pi], mech_cols[pj]))
        i, j = pi, pj

    pairs.reverse()
    return pairs


def _match_ordered_sequences(rect_order: list[int], mech_order: list[int]) -> list[tuple[int, int]]:
    rows = len(rect_order)
    cols = len(mech_order)
    if rows == 0 or cols == 0:
        return []

    gap_cost = 1.0
    dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    prev: list[list[tuple[int, int] | None]] = [[None for _ in range(cols + 1)] for _ in range(rows + 1)]
    dp[0, 0] = 0.0

    for i in range(rows + 1):
        for j in range(cols + 1):
            if not np.isfinite(dp[i, j]):
                continue
            if i < rows and j < cols:
                cost = _relative_rank_cost(i, rows, j, cols)
                if dp[i, j] + cost < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = dp[i, j] + cost
                    prev[i + 1][j + 1] = (i, j)
            if i < rows and dp[i, j] + gap_cost < dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j] + gap_cost
                prev[i + 1][j] = (i, j)
            if j < cols and dp[i, j] + gap_cost < dp[i, j + 1]:
                dp[i, j + 1] = dp[i, j] + gap_cost
                prev[i][j + 1] = (i, j)

    pairs: list[tuple[int, int]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        step = prev[i][j]
        if step is None:
            break
        pi, pj = step
        if i == pi + 1 and j == pj + 1:
            pairs.append((rect_order[pi], mech_order[pj]))
        i, j = pi, pj

    pairs.reverse()
    return pairs


def _relative_rank_cost(rect_idx: int, rect_len: int, mech_idx: int, mech_len: int) -> float:
    rect_pos = rect_idx / max(rect_len - 1, 1)
    mech_pos = mech_idx / max(mech_len - 1, 1)
    return abs(rect_pos - mech_pos)


def _delta_layout_penalty(
    pairs: list[tuple[int, int]],
    rect_cols: list[list[int]],
    ordered_mech_cols: list[list[int]],
    rect_points: np.ndarray,
    mech_points: np.ndarray,
    mech_delta_x: float,
    mech_delta_y: float,
    y_reversed: bool,
) -> float:
    penalties: list[float] = []

    mech_col_centers = [float(np.mean(mech_points[group, 0])) for group in ordered_mech_cols]
    for left, right in zip(mech_col_centers, mech_col_centers[1:]):
        penalties.append(_distance_to_positive_multiple(abs(right - left), mech_delta_x))

    pair_lookup = dict(pairs)
    for rect_group in rect_cols:
        rect_order = sorted(rect_group, key=lambda idx: rect_points[idx, 1])
        mech_order = [pair_lookup[idx] for idx in rect_order if idx in pair_lookup]
        if y_reversed:
            mech_order = list(mech_order)
        for a, b in zip(mech_order, mech_order[1:]):
            penalties.append(_distance_to_positive_multiple(abs(mech_points[b, 1] - mech_points[a, 1]), mech_delta_y))

    return float(np.mean(penalties)) if penalties else 0.0


def _distance_to_positive_multiple(value: float, step: float) -> float:
    multiple = max(1, int(round(value / step)))
    return abs(value - multiple * step)


def _fit_affine(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    ones = np.ones((src_xy.shape[0], 1), dtype=np.float64)
    design = np.hstack([src_xy, ones])
    affine, *_ = np.linalg.lstsq(design, dst_xy, rcond=None)
    return affine


def _fit_alignment_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Fit the most constrained transform supported by the number of pairs.

    One pair can only determine translation. Two pairs determine a similarity
    transform (rotation, uniform scale, and translation). Three or more pairs
    use the existing unconstrained affine least-squares fit.
    """
    pair_count = len(src_xy)
    if pair_count == 0:
        raise ValueError("at least one matched pair is required to fit an alignment transform")
    if pair_count == 1:
        translation = dst_xy[0] - src_xy[0]
        return np.array(
            [[1.0, 0.0], [0.0, 1.0], translation],
            dtype=np.float64,
        )
    if pair_count == 2:
        return _fit_similarity_transform(src_xy, dst_xy)
    return _fit_affine(src_xy, dst_xy)


def _fit_similarity_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Fit a two-point rotation, uniform-scale, and translation transform."""
    source_delta = src_xy[1] - src_xy[0]
    target_delta = dst_xy[1] - dst_xy[0]
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))

    if source_length == 0.0 or target_length == 0.0:
        translation = np.mean(dst_xy - src_xy, axis=0)
        return np.array(
            [[1.0, 0.0], [0.0, 1.0], translation],
            dtype=np.float64,
        )

    source_angle = float(np.arctan2(source_delta[1], source_delta[0]))
    target_angle = float(np.arctan2(target_delta[1], target_delta[0]))
    angle = target_angle - source_angle
    scale = target_length / source_length
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    linear = scale * np.array(
        [[cos_angle, sin_angle], [-sin_angle, cos_angle]],
        dtype=np.float64,
    )
    translation = dst_xy[0] - src_xy[0] @ linear
    return np.vstack((linear, translation))


def _apply_affine(src_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ones = np.ones((src_xy.shape[0], 1), dtype=np.float64)
    design = np.hstack([src_xy, ones])
    return design @ affine
