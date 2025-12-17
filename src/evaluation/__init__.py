"""
Evaluation module for the episodic memory pipeline.

Provides metrics and evaluation runners for assessing memory system quality.
"""
from .metrics import (
    RetrievalPrecisionMetric,
    FactConflictRateMetric,
    ConsolidationCompressionMetric,
    EvaluationMetrics,
)
from .runner import EvaluationRunner, EvaluationScenario, DiaryScenario

__all__ = [
    "RetrievalPrecisionMetric",
    "FactConflictRateMetric",
    "ConsolidationCompressionMetric",
    "EvaluationMetrics",
    "EvaluationRunner",
    "EvaluationScenario",
    "DiaryScenario",
]

