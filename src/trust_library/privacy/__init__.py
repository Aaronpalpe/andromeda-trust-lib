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
    epsilon_dp,
    epsilon_star,
    shapr,
    attribute_inference,
    privacy_risk,
    accuracy_ratio,
    k_anonymity,
    l_diversity,
    t_closeness,
)

__all__ = [
    # Pillar
    "PrivacyPillar",

    # Metrics
    "epsilon_dp",
    "epsilon_star",
    "shapr",
    "attribute_inference",
    "privacy_risk",
    "accuracy_ratio",
    "k_anonymity",
    "l_diversity",
    "t_closeness",
]