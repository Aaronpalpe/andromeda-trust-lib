"""
fairness_metrics_aif360.py
===============
Implementación de métricas de fairness utilizando AIF360 como motor de cálculo.

Las funciones mantienen la interfaz original basada en arrays crudos de numpy:
    y_true          : ground-truth labels (0/1)
    y_pred          : predicted labels (0/1)
    group_mask      : boolean array - True indica pertenencia al grupo *protegido* (unprivileged)
    X               : matriz de características (necesaria para métricas individuales)
"""

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Puente entre Numpy y AIF360
# ─────────────────────────────────────────────────────────────────────────────

def _get_aif360_datasets(
    y_true: np.ndarray = None, 
    y_pred: np.ndarray = None, 
    group_mask: np.ndarray = None, 
    X: np.ndarray = None
):
    """
    Convierte arrays de numpy en objetos BinaryLabelDataset de AIF360.
    Asume:
      - Privileged group (no protegido): group_mask == False (0.0)
      - Unprivileged group (protegido): group_mask == True (1.0)
    """
    privileged_groups = [{'protected_attribute': 0.0}]
    unprivileged_groups = [{'protected_attribute': 1.0}]
    
    # AIF360 requiere al menos una feature. Si no se provee X, creamos una dummy.
    if X is None:
        n_samples = len(group_mask) if group_mask is not None else (len(y_true) if y_true is not None else len(y_pred))
        df_X = pd.DataFrame({'dummy_feature': np.zeros(n_samples)})
    else:
        df_X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])

    mask_float = group_mask.astype(float) if group_mask is not None else np.zeros(len(df_X))

    dataset_true = None
    dataset_pred = None

    if y_true is not None:
        df_true = pd.concat([pd.DataFrame({'label': y_true, 'protected_attribute': mask_float}), df_X], axis=1)
        dataset_true = BinaryLabelDataset(
            df=df_true,
            label_names=['label'],
            protected_attribute_names=['protected_attribute'],
            favorable_label=1.0,
            unfavorable_label=0.0
        )

    if y_pred is not None:
        df_pred = pd.concat([pd.DataFrame({'label': y_pred, 'protected_attribute': mask_float}), df_X], axis=1)
        dataset_pred = BinaryLabelDataset(
            df=df_pred,
            label_names=['label'],
            protected_attribute_names=['protected_attribute'],
            favorable_label=1.0,
            unfavorable_label=0.0
        )

    return dataset_true, dataset_pred, privileged_groups, unprivileged_groups


# ─────────────────────────────────────────────────────────────────────────────
# Group Fairness Metrics (Vía AIF360)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_parity_difference(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Statistical Parity Difference (SPD) via AIF360."""
    _, dataset_pred, priv, unpriv = _get_aif360_datasets(y_pred=y_pred, group_mask=group_mask)
    metric = BinaryLabelDatasetMetric(dataset_pred, privileged_groups=priv, unprivileged_groups=unpriv)
    
    return {
        "value": metric.statistical_parity_difference(),
        "favored_ratio_protected": metric.base_rate(privileged=False),
        "favored_ratio_unprotected": metric.base_rate(privileged=True),
        "n_protected": metric.num_instances(privileged=False),
        "n_unprotected": metric.num_instances(privileged=True),
    }


def disparate_impact(y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Disparate Impact (DI) via AIF360."""
    _, dataset_pred, priv, unpriv = _get_aif360_datasets(y_pred=y_pred, group_mask=group_mask)
    metric = BinaryLabelDatasetMetric(dataset_pred, privileged_groups=priv, unprivileged_groups=unpriv)
    
    return {
        "value": metric.disparate_impact(),
        "favored_ratio_protected": metric.base_rate(privileged=False),
        "favored_ratio_unprotected": metric.base_rate(privileged=True),
    }


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Equal Opportunity Difference (EOD) via AIF360."""
    dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, group_mask)
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
    return {
        "value": metric.equal_opportunity_difference(),
        "tpr_protected": metric.true_positive_rate(privileged=False),
        "tpr_unprotected": metric.true_positive_rate(privileged=True),
    }


def average_odds_difference(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
    """Average Odds Difference (AOD) via AIF360."""
    dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, group_mask)
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
    return {
        "value": metric.average_odds_difference(),
        "tpr_protected": metric.true_positive_rate(privileged=False),
        "tpr_unprotected": metric.true_positive_rate(privileged=True),
        "fpr_protected": metric.false_positive_rate(privileged=False),
        "fpr_unprotected": metric.false_positive_rate(privileged=True),
    }


# def accuracy_parity(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
#     """Accuracy Parity calculada mediante los primitivos de AIF360."""
#     dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, group_mask)
#     metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
#     acc_prot = metric.accuracy(privileged=False)
#     acc_unprot = metric.accuracy(privileged=True)
    
#     return {
#         "value": acc_prot - acc_unprot,
#         "accuracy_protected": acc_prot,
#         "accuracy_unprotected": acc_unprot,
#     }


# def predictive_parity(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
#     """Predictive Parity calculada mediante los primitivos de AIF360."""
#     dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, group_mask)
#     metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
#     ppv_prot = metric.positive_predictive_value(privileged=False)
#     ppv_unprot = metric.positive_predictive_value(privileged=True)
#     npv_prot = metric.negative_predictive_value(privileged=False)
#     npv_unprot = metric.negative_predictive_value(privileged=True)
    
#     val = 0.5 * ((ppv_prot - ppv_unprot) + (npv_prot - npv_unprot))
#     return {
#         "value": val,
#         "ppv_protected": ppv_prot,
#         "ppv_unprotected": ppv_unprot,
#         "npv_protected": npv_prot,
#         "npv_unprotected": npv_unprot,
#     }


# def treatment_equality(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict:
#     """Treatment Equality calculada extraíendo FNs y FPs de AIF360."""
#     dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, group_mask)
#     metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
#     fn_p = metric.false_negatives(privileged=False)
#     fp_p = metric.false_positives(privileged=False)
#     fn_u = metric.false_negatives(privileged=True)
#     fp_u = metric.false_positives(privileged=True)
    
#     ratio_p = (fn_p / fp_p) if fp_p != 0 else np.inf
#     ratio_u = (fn_u / fp_u) if fp_u != 0 else np.inf
    
#     return {
#         "value": ratio_p - ratio_u,
#         "fn_protected": int(fn_p),
#         "fp_protected": int(fp_p),
#         "fn_fp_ratio_protected": ratio_p,
#         "fn_unprotected": int(fn_u),
#         "fp_unprotected": int(fp_u),
#         "fn_fp_ratio_unprotected": ratio_u,
#     }


# ─────────────────────────────────────────────────────────────────────────────
# Inequality / Information-Theory Metrics (Vía AIF360)
# ─────────────────────────────────────────────────────────────────────────────

def generalized_entropy_index(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 2) -> dict:
    """Generalized Entropy Index via AIF360."""
    dummy_mask = np.zeros_like(y_true, dtype=bool)
    dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, dummy_mask)
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
    return {
        "value": metric.generalized_entropy_index(alpha=alpha), 
        "alpha": alpha
    }


def theil_index(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Theil Index via AIF360."""
    dummy_mask = np.zeros_like(y_true, dtype=bool)
    dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, dummy_mask)
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
    return {
        "value": metric.theil_index(), 
        "name": "Theil Index"
    }


def coefficient_of_variation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Coefficient of Variation via AIF360."""
    dummy_mask = np.zeros_like(y_true, dtype=bool)
    dataset_true, dataset_pred, priv, unpriv = _get_aif360_datasets(y_true, y_pred, dummy_mask)
    metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)
    
    return {
        "value": metric.coefficient_of_variation()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual Fairness (Vía AIF360)
# ─────────────────────────────────────────────────────────────────────────────

def individual_consistency(X: np.ndarray, y_pred: np.ndarray, k: int = 5) -> dict:
    """Individual Consistency via AIF360 (Requiere matriz X)."""
    dummy_mask = np.zeros(len(y_pred), dtype=bool)
    _, dataset_pred, priv, unpriv = _get_aif360_datasets(y_pred=y_pred, group_mask=dummy_mask, X=X)
    metric = BinaryLabelDatasetMetric(dataset_pred, privileged_groups=priv, unprivileged_groups=unpriv)
    
    # AIF360 devuelve un array con un único valor
    val = metric.consistency(n_neighbors=k)[0]
    
    return {
        "value": float(val), 
        "k": k
    }


# class_imbalance

# kl_divergence

# smoothed_edf

# bias_amplification