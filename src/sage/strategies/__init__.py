"""First-stage retrieval strategies and result fusion."""

from sage.strategies.fusion import merge_deduplicate, reciprocal_rank_fusion
from sage.strategies.hyde import HydeStrategy
from sage.strategies.modality_fusion import (
    CalibrationSample,
    ModalityCalibrator,
    infer_modality,
)
from sage.strategies.semantic import SemanticStrategy
from sage.strategies.step_back import StepBackStrategy

__all__ = [
    "CalibrationSample",
    "HydeStrategy",
    "ModalityCalibrator",
    "SemanticStrategy",
    "StepBackStrategy",
    "infer_modality",
    "merge_deduplicate",
    "reciprocal_rank_fusion",
]
