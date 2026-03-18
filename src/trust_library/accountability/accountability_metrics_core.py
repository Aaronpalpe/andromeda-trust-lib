"""
accountability_metrics_core.py
==============================

Pure mathematical implementations of accountability metrics.

Design principles
-----------------
• No dependency on Pillars, Result objects, or configuration logic.
• Deterministic and side-effect free.
• Operates on raw data structures (pandas / numpy).
• Returns structured dictionaries with raw mathematical outputs.

These functions DO NOT perform scoring.
They only compute measurable quantities.
"""

from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import re
from math import isclose


# =============================================================================
# Train / Test Split
# =============================================================================

def train_test_split_ratio(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Dict[str, int]:
    """
    Compute percentage ratio of training vs test data.

    Returns
    -------
    {
        "train_ratio": int,
        "test_ratio": int,
    }
    """

    n_train: int = len(train_data)
    n_test: int = len(test_data)
    total: int = n_train + n_test

    if total == 0:
        return {"train_ratio": 0, "test_ratio": 0}

    train_ratio: int = round(n_train / total * 100)
    test_ratio: int = 100 - train_ratio

    return {
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
    }


def train_test_split_mapping(
    train_ratio: int,
    mappings: Dict[str, float],
) -> float:
    """
    Map train ratio to a configured score interval.

    Parameters
    ----------
    train_ratio : int
    mappings    : dict of "lower-upper" -> score

    Returns
    -------
    score : float
    """

    for key, value in mappings.items():
        bounds = re.findall(r"\d+-\d+", key)
        for b in bounds:
            low, high = map(int, b.split("-"))
            if low <= train_ratio < high:
                return float(value)

    return float("nan")


# =============================================================================
# Missing Data
# =============================================================================

def count_missing_values(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> int:
    """
    Count total missing values across train and test datasets.
    """

    return int(
        train_data.isna().sum().sum() +
        test_data.isna().sum().sum()
    )


# =============================================================================
# Normalization Analysis
# =============================================================================

def normalization_statistics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute mean and standard deviation statistics.
    """

    return {
        "train_mean": float(np.mean(X_train.values)),
        "train_std": float(np.std(X_train.values)),
        "test_mean": float(np.mean(X_test.values)),
        "test_std": float(np.std(X_test.values)),
    }


def detect_normalization_type(
    stats: Dict[str, float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> str:
    """
    Detect normalization strategy.

    Returns one of:
        - "training_and_test_standardize"
        - "training_standardized"
        - "training_and_test_normal"
        - "training_normal"
        - "None"
    """

    tm = stats["train_mean"]
    ts = stats["train_std"]
    tem = stats["test_mean"]
    tes = stats["test_std"]

    if (
        isclose(tm, 0, abs_tol=1e-3)
        and isclose(ts, 1, abs_tol=1e-3)
        and isclose(tem, 0, abs_tol=1e-3)
        and isclose(tes, 1, abs_tol=1e-3)
    ):
        return "training_and_test_standardize"

    if (
        isclose(tm, 0, abs_tol=1e-3)
        and isclose(ts, 1, abs_tol=1e-3)
    ):
        return "training_standardized"

    if (
        X_train.min().min() >= 0
        and X_train.max().max() <= 1
        and X_test.min().min() >= 0
        and X_test.max().max() <= 1
    ):
        return "training_and_test_normal"

    if (
        X_train.min().min() >= 0
        and X_train.max().max() <= 1
    ):
        return "training_normal"

    return "None"


# =============================================================================
# Regularization
# =============================================================================

def regularization_mapping(
    regularization_name: str | None,
) -> float:
    """
    Map regularization technique to accountability score.
    Regularization is the technique used to prevent overfitting and improving model generalization.

    Returns
    -------
    score in [0,5]
    """

    mapping: Dict[str | None, float] = {
        "elasticnet_regression": 5.0,
        "lasso_regression": 4.0,
        "ridge_regression": 4.0,
        "other": 3.0,
        None: 0.0,
    }

    return mapping.get(regularization_name, 1.0)


# =============================================================================
# Factsheet Completeness
# =============================================================================

def factsheet_completeness(
    factsheet: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Compute completeness ratio of a structured factsheet.

    A field is considered present if:
        - value is not None
        - strings/lists/dicts are non-empty
        - numeric/boolean values exist

    Returns
    -------
    {
        "ratio": float,
        "present": int,
        "total": int,
        "missing": List[str]
    }
    """

    total: int = 0
    present: int = 0
    missing: List[str] = []

    for section, fields in factsheet.items():
        for field_key, field_data in fields.items():
            total += 1
            value = field_data.get("value")

            filled: bool = False

            if value is not None:
                if isinstance(value, (str, list, dict)):
                    filled = len(value) > 0
                else:
                    filled = True

            if filled:
                present += 1
            else:
                missing.append(f"{section}.{field_key}")

    ratio: float = present / total if total > 0 else 0.0

    return {
        "ratio": float(ratio),
        "present": present,
        "total": total,
        "missing": missing,
    }