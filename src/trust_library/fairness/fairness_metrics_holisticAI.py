"""
fairness_metrics_holisticai.py
===============
Implementation of fairness metrics using HolisticAI as the calculation engine.

Functions maintain the original interface based on raw numpy arrays:
    y_true          : ground-truth labels (0/1)
    y_pred          : predicted labels (0/1)
    group_mask      : boolean array - True indicates membership in the *protected* (group_a) group
    X               : feature matrix (needed for individual consistency)
"""

import numpy as np
import pandas as pd

# Importamos las métricas individuales de HolisticAI
from holisticai.bias.metrics import (
    statistical_parity,
    disparate_impact as hai_disparate_impact,
    # four_fifths_rule,
    cohen_d,
    z_test_diff as z_test_diff_hai,
    equal_opportunity_diff,
    #false_positive_rate_diff,
    average_odds_diff,
    accuracy_diff,
    theil_index as hai_theil_index,
    generalized_entropy_index as hai_generalized_entropy_index,
    coefficient_of_variation as hai_coefficient_of_variation,
    consistency_score
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Extraction of base rates (since HolisticAI returns only the float)
# ─────────────────────────────────────────────────────────────────────────────

def _get_base_rates(y: np.ndarray, mask: np.ndarray):
    """Calculates base rates to fill the wrapper dictionaries."""
    prot_rate = float(y[mask].mean()) if mask.sum() > 0 else 0.0
    unprot_rate = float(y[~mask].mean()) if (~mask).sum() > 0 else 0.0
    return prot_rate, unprot_rate

def _get_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Calculates TPR and FPR manually for the wrapper (HolisticAI hides these intermediate steps)."""
    positives = y_true[mask] == 1
    negatives = y_true[mask] == 0
    tpr = float((y_pred[mask][positives] == 1).sum() / positives.sum()) if positives.sum() > 0 else 0.0
    fpr = float((y_pred[mask][negatives] == 1).sum() / negatives.sum()) if negatives.sum() > 0 else 0.0
    return tpr, fpr


# ─────────────────────────────────────────────────────────────────────────────
# Group Fairness Metrics (Via HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_parity_difference(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Statistical Parity Difference (SPD) via HolisticAI."""
    # HolisticAI convention: group_a = protected/unprivileged, group_b = unprotected
    group_a = group_mask
    group_b = ~group_mask
    
    val = statistical_parity(group_a, group_b, y_pred)
    prot_rate, unprot_rate = _get_base_rates(y_pred, group_mask)
    
    return {
        "value": float(val),
        "favored_ratio_protected": prot_rate,
        "favored_ratio_unprotected": unprot_rate,
        "n_protected": int(group_mask.sum()),
        "n_unprotected": int((~group_mask).sum()),
        "n_protected_favored": int((y_pred[group_mask] == 1).sum()),
        "n_unprotected_favored": int((y_pred[~group_mask] == 1).sum()),
    }


def disparate_impact(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Disparate Impact (DI) via HolisticAI."""
    group_a = group_mask
    group_b = ~group_mask
    
    val = hai_disparate_impact(group_a, group_b, y_pred)
    prot_rate, unprot_rate = _get_base_rates(y_pred, group_mask)
    
    return {
        "value": float(val),
        "favored_ratio_protected": prot_rate,
        "favored_ratio_unprotected": unprot_rate,
        "n_protected": int(group_mask.sum()),
        "n_unprotected": int((~group_mask).sum()),
        "n_protected_favored": int((y_pred[group_mask] == 1).sum()),
        "n_unprotected_favored": int((y_pred[~group_mask] == 1).sum()),
    }


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Equal Opportunity Difference (EOD) via HolisticAI."""
    group_a = group_mask
    group_b = ~group_mask
    
    val = equal_opportunity_diff(group_a, group_b, y_pred, y_true)
    tpr_prot, _ = _get_tpr_fpr(y_true, y_pred, group_mask)
    tpr_unprot, _ = _get_tpr_fpr(y_true, y_pred, ~group_mask)
    
    return {
        "value": float(val),
        "tpr_protected": tpr_prot,
        "tpr_unprotected": tpr_unprot,
    }


def average_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Average Odds Difference (AOD) via HolisticAI."""
    group_a = group_mask
    group_b = ~group_mask
    
    val = average_odds_diff(group_a, group_b, y_pred, y_true)
    tpr_p, fpr_p = _get_tpr_fpr(y_true, y_pred, group_mask)
    tpr_u, fpr_u = _get_tpr_fpr(y_true, y_pred, ~group_mask)
    
    return {
        "value": float(val),
        "tpr_protected": tpr_p,
        "tpr_unprotected": tpr_u,
        "fpr_protected": fpr_p,
        "fpr_unprotected": fpr_u,
    }


def accuracy_parity(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Accuracy Parity via HolisticAI."""
    group_a = group_mask
    group_b = ~group_mask
    
    val = accuracy_diff(group_a, group_b, y_pred, y_true)
    
    # Extraemos la accuracy de los grupos para el wrapper
    acc_prot = float((y_true[group_mask] == y_pred[group_mask]).mean()) if group_mask.sum() > 0 else 0.0
    acc_unprot = float((y_true[~group_mask] == y_pred[~group_mask]).mean()) if (~group_mask).sum() > 0 else 0.0
    
    return {
        "value": float(val),
        "accuracy_protected": acc_prot,
        "accuracy_unprotected": acc_unprot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual Fairness (Via HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def individual_consistency(X: np.ndarray, y_pred: np.ndarray, k: int = 5) -> dict:
    """
    Individual Consistency Score via HolisticAI.
    HolisticAI implements this natively by asking for X, y_pred and n_neighbors.
    """
    # Important: HolisticAI assumes n_neighbors, its default value is usually 5.
    val = consistency_score(X, y_pred, n_neighbors=k)

    return {
        "value": float(val),
        "k": k
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size (Via HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """
    Cohen's D effect size via HolisticAI.
    HolisticAI has this native metric in its bias module.
    """
    group_a = group_mask      # Protected
    group_b = ~group_mask     # Unprotected

    # HolisticAI returns directly the effect value
    val = cohen_d(group_a, group_b, y_pred)

    # Extract the means and standard deviation to fill your wrapper
    g1 = y_pred[group_mask].astype(float)
    g2 = y_pred[~group_mask].astype(float)
    sigma = float(np.sqrt((g1.var() + g2.var()) / 2))
    
    return {
        "value": float(val),
        "mean_protected": float(g1.mean()) if len(g1) > 0 else 0.0,
        "mean_unprotected": float(g2.mean()) if len(g2) > 0 else 0.0,
        "pooled_std": sigma,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inequality / Information-Theory Metrics (Via HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def generalized_entropy_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 2
) -> dict:
    """
    Generalized Entropy Index via HolisticAI.
    """
    # HolisticAI calculates the benefits internally by receiving y_pred and y_true
    val = hai_generalized_entropy_index(y_pred, y_true, alpha=alpha)

    return {
        "value": float(val)
    }

def theil_index(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Theil Index via HolisticAI.
    Equivalent to GEI with alpha=1.
    """
    val = hai_theil_index(y_pred, y_true)

    return {
        "value": float(val)
    }

def coefficient_of_variation( # NO IGUAL QUE AIF---------------
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Coefficient of Variation via HolisticAI.
    """
    val = hai_coefficient_of_variation(y_pred, y_true)

    return {
        "value": float(val)
    }


def z_test_diff(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    n_prot = int(group_mask.sum())
    n_unprot = int((~group_mask).sum())
    
    if n_prot == 0 or n_unprot == 0:
        return {"value": 0.0, "sr_protected": 0.0, "sr_unprotected": 0.0}

    sr_prot = float(y_pred[group_mask].mean())
    sr_unprot = float(y_pred[~group_mask].mean())
    
    val = z_test_diff_hai(group_mask, ~group_mask, y_pred)

    return {
        "value": float(val),
        "sr_protected": sr_prot,
        "sr_unprotected": sr_unprot,
    }