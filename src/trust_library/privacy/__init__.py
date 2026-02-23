"""
Privacy Module
===================
Modular privacy analysis package.

Structure
---------
privacy_metrics_core.py  - pure mathematical implementations
PrivacyPillar
"""

from .privacy import PrivacyPillar

# ─────────────────────────────────────────────────────────────
# Privacy Metrics
# ─────────────────────────────────────────────────────────────
from .privacy_metrics_core import (
    compute_epsilon_star,
    compute_shapr,
    compute_attribute_inference,
    compute_privacy_risk,
    compute_accuracy_ratio,
    compute_k_anonymity,
    compute_l_diversity,
    compute_t_closeness,
)

__all__ = [
    # Pillar
    "PrivacyPillar",

    # Metrics
    "compute_epsilon_star",
    "compute_shapr",
    "compute_attribute_inference",
    "compute_privacy_risk",
    "compute_accuracy_ratio",
    "compute_k_anonymity",
    "compute_l_diversity",
    "compute_t_closeness",
]