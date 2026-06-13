"""Modality-calibrated relevance scoring.

A retrieval score (bi-encoder cosine, cross-encoder logit, RRF) does not mean the same
thing across content modalities: a 0.6 on a code chunk, a table cell, and a prose
paragraph imply different probabilities of relevance, because the encoders were trained
predominantly on prose and each modality has its own score distribution. Standard fusion
ranks raw scores directly, so systematically mis-scored modalities are ranked unfairly.

This module learns, per ``(modality, score_source)``, a monotonic map from raw score to
P(relevant) via isotonic regression on a held-out split with gold labels, then maps
every candidate onto that common probability scale before ranking. It generalizes the
per-source threshold calibration of SSCC (which helped) to the heterogeneous-modality
axis. Buckets with too few samples or a single class fall back to the identity map, so
calibration never degrades a modality it cannot fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["CalibrationSample", "ModalityCalibrator", "infer_modality"]

# Below this many labeled samples (or with a single class) a bucket is left uncalibrated.
_MIN_SAMPLES = 25


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One labeled candidate: its bucket, raw score, and relevance (0/1)."""

    modality: str
    score_source: str
    score: float
    relevant: int


def infer_modality(*, level: int, filename: str, language: str | None) -> str:
    """Coarse modality of a retrieved node from its metadata.

    Summary/corpus nodes (level > 0) form their own bucket; leaves are typed by language
    (code) or filename extension (table / markdown / pdf / prose).
    """
    if level > 0:
        return "summary"
    if language:
        return "code"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in {"csv", "tsv", "xlsx", "xls"}:
        return "table"
    if ext in {"md", "markdown"}:
        return "markdown"
    if ext == "pdf":
        return "pdf"
    return "prose"


@dataclass(slots=True)
class ModalityCalibrator:
    """Per-(modality, score_source) calibration of relevance scores.

    ``method="platt"`` (default) fits a strictly-monotonic logistic map, which preserves
    the within-modality score order exactly and only shifts modalities relative to one
    another -- the intended effect. ``method="isotonic"`` is a flexible step-function
    fit but introduces score ties (plateaus) that can perturb tie-sensitive metrics.
    """

    method: str = "platt"
    _maps: dict[tuple[str, str], object] = field(default_factory=dict)

    def fit(self, samples: Sequence[CalibrationSample]) -> ModalityCalibrator:
        """Fit one calibrator per bucket with enough labeled, two-class data."""
        import numpy as np

        buckets: dict[tuple[str, str], list[CalibrationSample]] = {}
        for s in samples:
            buckets.setdefault((s.modality, s.score_source), []).append(s)
        for key, items in buckets.items():
            labels = {s.relevant for s in items}
            if len(items) < _MIN_SAMPLES or len(labels) < 2:
                continue  # too little signal -> identity (handled in calibrate)
            x = np.array([s.score for s in items], dtype=np.float64).reshape(-1, 1)
            y = np.array([s.relevant for s in items], dtype=np.float64)
            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                model: object = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.0, y_max=1.0
                ).fit(x.ravel(), y)
            else:  # platt: logistic regression on the raw score
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression().fit(x, y)
            self._maps[key] = model
        return self

    @property
    def fitted_buckets(self) -> list[tuple[str, str]]:
        return sorted(self._maps)

    def calibrate(self, score: float, modality: str, score_source: str) -> float:
        """Map a raw score to calibrated P(relevant); identity for unfitted buckets."""
        model = self._maps.get((modality, score_source))
        if model is None:
            return score
        if self.method == "isotonic":
            return float(model.predict([score])[0])  # type: ignore[attr-defined]
        return float(model.predict_proba([[score]])[0, 1])  # type: ignore[attr-defined]
