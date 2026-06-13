"""Contract tests for modality-calibrated relevance scoring."""

from __future__ import annotations

from sage.strategies.modality_fusion import (
    CalibrationSample,
    ModalityCalibrator,
    infer_modality,
)


def test_infer_modality_by_metadata():
    assert infer_modality(level=2, filename="x.pdf", language=None) == "summary"
    assert infer_modality(level=0, filename="m.py", language="python") == "code"
    assert infer_modality(level=0, filename="data.csv", language=None) == "table"
    assert infer_modality(level=0, filename="readme.md", language=None) == "markdown"
    assert infer_modality(level=0, filename="paper.pdf", language=None) == "pdf"
    assert infer_modality(level=0, filename="notes.txt", language=None) == "prose"


def test_calibration_corrects_a_miscalibrated_modality():
    # "table" scores are deflated: relevant tables score ~0.4, irrelevant prose ~0.5.
    # Raw ranking puts prose above relevant tables; calibration must invert that.
    samples: list[CalibrationSample] = []
    for i in range(60):
        rel = i % 2
        score = 0.4 if rel else 0.3  # relevant tables outscore irrelevant ones, but low
        samples.append(CalibrationSample("table", "cross_encoder", score, rel))
    for i in range(60):
        rel = i % 2
        score = 0.55 if rel else 0.5
        samples.append(CalibrationSample("prose", "cross_encoder", score, rel))

    cal = ModalityCalibrator().fit(samples)
    assert ("table", "cross_encoder") in cal.fitted_buckets
    # A relevant table (raw 0.4) should calibrate above an irrelevant prose (raw 0.5).
    table_rel = cal.calibrate(0.4, "table", "cross_encoder")
    prose_irrel = cal.calibrate(0.5, "prose", "cross_encoder")
    assert table_rel > prose_irrel
    # Calibrated values are probabilities in [0, 1].
    assert 0.0 <= table_rel <= 1.0 and 0.0 <= prose_irrel <= 1.0


def test_unfitted_bucket_is_identity():
    cal = ModalityCalibrator()  # nothing fitted
    assert cal.calibrate(0.73, "code", "bi_encoder") == 0.73
    # Too few samples -> bucket left as identity.
    cal.fit([CalibrationSample("code", "bi_encoder", 0.5, 1) for _ in range(5)])
    assert cal.calibrate(0.73, "code", "bi_encoder") == 0.73
