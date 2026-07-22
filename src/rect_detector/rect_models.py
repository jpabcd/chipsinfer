from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Rect_Info:
    id: int
    box: Box
    scaled_box: Box
    area: int
    aspect: float
    score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
