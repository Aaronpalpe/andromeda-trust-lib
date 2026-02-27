"""
fairness_metrics_holisticai.py
===============
Implementación de métricas de fairness utilizando HolisticAI como motor de cálculo.

Las funciones mantienen la interfaz original basada en arrays crudos de numpy:
    y_true          : ground-truth labels (0/1)
    y_pred          : predicted labels (0/1)
    group_mask      : boolean array - True indica pertenencia al grupo *protegido* (group_a)
    X               : matriz de características (necesaria para consistencia individual)
"""

import numpy as np
import pandas as pd

# Importamos las métricas individuales de HolisticAI
from holisticai.bias.metrics import (
    statistical_parity,
    disparate_impact as hai_disparate_impact,
    # four_fifths_rule,
    cohen_d,
    # sd_rule,
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
# Helper: Extracción de tasas base (ya que HolisticAI devuelve solo el float)
# ─────────────────────────────────────────────────────────────────────────────

def _get_base_rates(y: np.ndarray, mask: np.ndarray):
    """Calcula las tasas base para rellenar los diccionarios del wrapper."""
    prot_rate = float(y[mask].mean()) if mask.sum() > 0 else 0.0
    unprot_rate = float(y[~mask].mean()) if (~mask).sum() > 0 else 0.0
    return prot_rate, unprot_rate

def _get_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    """Calcula TPR y FPR manualmente para el wrapper (HolisticAI oculta estos pasos intermedios)."""
    positives = y_true[mask] == 1
    negatives = y_true[mask] == 0
    tpr = float((y_pred[mask][positives] == 1).sum() / positives.sum()) if positives.sum() > 0 else 0.0
    fpr = float((y_pred[mask][negatives] == 1).sum() / negatives.sum()) if negatives.sum() > 0 else 0.0
    return tpr, fpr


# ─────────────────────────────────────────────────────────────────────────────
# Group Fairness Metrics (Vía HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_parity_difference(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Statistical Parity Difference (SPD) via HolisticAI."""
    # HolisticAI convención: group_a = protegido/unprivileged, group_b = no protegido
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
# Individual Fairness (Vía HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def individual_consistency(X: np.ndarray, y_pred: np.ndarray, k: int = 5) -> dict:
    """
    Individual Consistency Score via HolisticAI.
    HolisticAI implementa esto nativamente pidiendo X, y_pred y n_neighbors.
    """
    # Importante: HolisticAI asume n_neighbors, su valor por defecto suele ser 5.
    val = consistency_score(X, y_pred, n_neighbors=k)
    
    return {
        "value": float(val), 
        "k": k
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size (Vía HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """
    Cohen's D effect size via HolisticAI.
    HolisticAI tiene esta métrica nativa en su módulo de bias.
    """
    group_a = group_mask      # Protegido
    group_b = ~group_mask     # No protegido
    
    # HolisticAI devuelve directamente el valor del efecto
    val = cohen_d(group_a, group_b, y_pred)
    
    # Extraemos las medias y la desviación para rellenar tu wrapper
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
# Inequality / Information-Theory Metrics (Vía HolisticAI)
# ─────────────────────────────────────────────────────────────────────────────

def generalized_entropy_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 2
) -> dict:
    """
    Generalized Entropy Index via HolisticAI.
    """
    # HolisticAI calcula los beneficios internamente recibiendo y_pred y y_true
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