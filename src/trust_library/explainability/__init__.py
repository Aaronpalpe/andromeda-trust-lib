"""Explainability Module

Implements SHAP-based explainability evaluation.

Public API:
  - ExplainabilityPillar
  - compute_shap_based_metrics
"""

from .explainability import ExplainabilityPillar
from .explainability_metrics_core import compute_shap_based_metrics

__all__ = [
    "ExplainabilityPillar",
    "compute_shap_based_metrics",
]
