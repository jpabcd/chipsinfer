from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryClassification:
    """One binary classifier result corresponding to one input ROI image."""

    label: str
    confidence: float
    is_positive: bool


class DummyBinaryClassifier:
    """Framework-independent placeholder for a batched binary classifier.

    This dummy model returns a configurable fixed result for every ROI. It is
    intentionally deterministic so the detector-to-classifier integration can
    be tested before a real model format is selected.
    """

    def __init__(
        self,
        positive_label: str = "positive",
        negative_label: str = "negative",
        positive: bool = True,
        confidence: float = 1.0,
    ) -> None:
        if not positive_label.strip() or not negative_label.strip():
            raise ValueError("classifier labels must be non-empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in the range [0, 1]")
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.positive = positive
        self.confidence = confidence

    def infer_batch(self, roi_images: Sequence[np.ndarray]) -> list[BinaryClassification]:
        """Classify all ROI images in one call and return results in input order."""
        label = self.positive_label if self.positive else self.negative_label
        result = BinaryClassification(
            label=label,
            confidence=self.confidence,
            is_positive=self.positive,
        )
        return [result for _ in roi_images]


__all__ = ["BinaryClassification", "DummyBinaryClassifier"]
