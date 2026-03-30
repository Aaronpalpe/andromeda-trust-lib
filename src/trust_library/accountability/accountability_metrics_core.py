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

from trust_library.factsheet import Factsheet

# =============================================================================
# Train / Test Split
# =============================================================================

def train_test_split_ratio(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Dict[str, int]:
    """
    Compute percentage ratio of training vs test data.

    Parameters
    ----------
    train_data : dataset used for training (pd.DataFrame)
    test_data  : dataset used for testing (pd.DataFrame)

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
    Map train ratio to a configured score interval. If train_ratio falls within a defined interval, return the corresponding score.

    Parameters
    ----------
    train_ratio : int percentage of training data (0-100)
    mappings    : dict of "lower-upper : score"

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

    raise ValueError(f"No mapping found for train_ratio={train_ratio}. Check your configuration.")


# =============================================================================
# Missing Data
# =============================================================================

def count_missing_values(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> int:
    """
    Count total missing values across train and test datasets.

    Parameters
    ----------
    train_data: pd.DataFrame
        dataset used for training
    test_data  : pd.DataFrame
        dataset used for testing

    Returns
    -------
    {
        "missing_train": int,
        "missing_test": int,
        "value": int,  # total missing
    }
    """

    missing_train = train_data.isna().sum().sum()
    missing_test = test_data.isna().sum().sum()

    return {
        "missing_train": int(missing_train),
        "missing_test": int(missing_test),
        "value": int(missing_train + missing_test),
    }


# =============================================================================
# Normalization Analysis
# =============================================================================

def normalization_statistics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute mean and standard deviation statistics.

    Parameters
    ----------
    X_train: pd.DataFrame
        training features
    X_test  : pd.DataFrame
        testing features

    Returns
    -------
    {
        "train_mean": float,
        "train_std": float,
        "test_mean": float,
        "test_std": float,
    }
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

    Parameters
    ----------
    stats:  Dict[str, float]
        output of normalization_statistics function
    X_train: pd.DataFrame
        training features 
    X_test: pd.DataFrame
        testing features (pd.DataFrame)

    Returns one of
    -------
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
    mappings: Dict[str | None, float],
) -> float:
    """
    Map regularization technique to accountability score.
    Regularization is the technique used to prevent overfitting and improving model generalization.

    Parameters
    ----------
    regularization_name: str | None
        name of the regularization technique used (e.g., "elasticnet_regression", "lasso_regression", "ridge_regression", "other", None)
    mappings: Dict[str | None, float]
        dict of "regularization_name : score"
    
    Returns
    -------
    int: score between 1 and 5, where higher values indicate stronger regularization techniques that contribute to better accountability. If the regularization technique is not recognized, a default score of 1.0 is returned.
    """

    return mappings.get(regularization_name, 1.0)


# =============================================================================
# Factsheet Completeness
# =============================================================================

def factsheet_completeness(
    factsheet: Factsheet,
) -> Dict[str, Any]:
    """
    Compute completeness ratio of a structured factsheet.

    A field is considered present if:
        - value is not None
        - strings/lists/dicts are non-empty
        - numeric/boolean values exist

    Parameters
    ----------
    factsheet: Factsheet or dict
        structured factsheet class or dict

    Returns
    -------
    {
        "ratio": float, # completeness ratio (0-1)
        "present": int, # number of present fields
        "total": int, # total number of fields
        "missing": List[str] # list of missing field paths (e.g., "training_data.size")
    }
    """
    if not isinstance(factsheet, dict):
        factsheet = factsheet.to_dict()

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