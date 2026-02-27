"""
Fairness Module
===================
Modular fairness analysis package.

Structure
---------
fairness_metrics_core.py  - pure mathematical implementations (no fairness library deps)
FairnessPillar
"""

from .fairness import FairnessPillar

# ─────────────────────────────────────────────────────────────
# Group Fairness Metrics
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    statistical_parity_difference,
    disparate_impact,
    equal_opportunity_difference,
    average_odds_difference,
    accuracy_parity,
    predictive_parity,
    treatment_equality,
)

# ─────────────────────────────────────────────────────────────
# Calibration Metrics
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    calibration_gap,
    well_calibration_error,
)

# ─────────────────────────────────────────────────────────────
# Inequality / Information-Theory Metrics
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    generalized_entropy_index,
    theil_index,
    coefficient_of_variation,
    kl_divergence,
)

# ─────────────────────────────────────────────────────────────
# Individual Fairness
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    individual_consistency,
)

# ─────────────────────────────────────────────────────────────
# Dataset / Distribution Metrics
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    class_balance,
    class_imbalance,
)

# ─────────────────────────────────────────────────────────────
# Model Fit Metrics
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    underfitting,
    overfitting,
)

# ─────────────────────────────────────────────────────────────
# Bias & Effect Size
# ─────────────────────────────────────────────────────────────
from .fairness_metrics_core import (
    bias_amplification,
    cohens_d,
    smoothed_edf,
)

__all__ = [
    # Pillar
    "FairnessPillar",

    # Group fairness
    "statistical_parity_difference",
    "disparate_impact",
    "equal_opportunity_difference",
    "average_odds_difference",
    "accuracy_parity",
    "predictive_parity",
    "treatment_equality",

    # Calibration
    "calibration_gap",
    "well_calibration_error",

    # Inequality / entropy
    "generalized_entropy_index",
    "theil_index",
    "coefficient_of_variation",
    "kl_divergence",

    # Individual fairness
    "individual_consistency",

    # Dataset metrics
    "class_balance",
    "class_imbalance",

    # Fit metrics
    "underfitting",
    "overfitting",

    # Bias / effect size
    "bias_amplification",
    "cohens_d",
    "smoothed_edf",
]