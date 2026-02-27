"""
Accountability Module
=====================
Modular accountability analysis package.

Structure
---------
accountability_metrics_core.py  - pure mathematical implementations (no fairness library deps)
AccountabilityPillar
"""

from .accountability import AccountabilityPillar

from .accountability_metrics_core import (
    train_test_split_ratio,
    train_test_split_mapping,
    count_missing_values,
    normalization_statistics,
    detect_normalization_type,
    regularization_mapping,
    factsheet_completeness,
)

__all__ = [
    "AccountabilityPillar",
    "train_test_split_ratio",
    "train_test_split_mapping",
    "count_missing_values",
    "normalization_statistics",
    "detect_normalization_type",
    "regularization_mapping",
    "factsheet_completeness",
]