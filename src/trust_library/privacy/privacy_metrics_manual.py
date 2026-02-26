from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# =============================================================================
# Epsilon DP Leakage
# =============================================================================
def compute_epsilon_dp(epsilon: float) -> Dict[str, float]:
    """
    Compute a score for epsilon DP leakage based on predefined thresholds.
    """
    if epsilon is None:
        return {"value": np.nan}
    
    return {
        "value": epsilon,
    }

# =============================================================================
# Helper: Log-loss per instance
# =============================================================================

def _calculate_losses(model, X, y) -> np.ndarray:
    """
    Compute Log Loss (Cross-Entropy) for each instance.
    """
    probs = model.predict_proba(X)

    if hasattr(model, "classes_"):
        class_map = {c: i for i, c in enumerate(model.classes_)}
        y_idx = np.array([class_map[label] for label in y])
    else:
        y_idx = y.astype(int)

    true_probs = probs[np.arange(len(y)), y_idx]
    true_probs = np.clip(true_probs, 1e-15, 1 - 1e-15)

    return -np.log(true_probs)


# =============================================================================
# Epsilon Star
# =============================================================================

def compute_epsilon_star(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
) -> Dict[str, float]:
    """
    Compute empirical epsilon* based on Loss Distribution as Definition 2 of the paper.
    """

    loss_train = _calculate_losses(model, X_train, y_train)
    loss_test  = _calculate_losses(model, X_test, y_test)

    scores = np.concatenate([-loss_train, -loss_test])
    y_true = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])

    fpr, tpr, _ = roc_curve(y_true, scores)

    fpr = np.clip(fpr, 1e-10, 1 - 1e-10)
    tpr = np.clip(tpr, 1e-10, 1 - 1e-10)
    fnr = 1 - tpr

    delta = 1.0 / len(loss_train) if len(loss_train) > 0 else 1e-5

    m1 = (1 - delta - fnr) / fpr
    m2 = (1 - delta - fpr) / fnr
    m3 = (fnr - delta) / (1 - fpr)
    m4 = (fpr - delta) / (1 - fnr)

    epsilon_star_val = np.log(
        np.nanmax(np.maximum.reduce([m1, m2, m3, m4, np.ones_like(m1)]))
    )

    return {
        "value": float(epsilon_star_val),
        "delta": float(delta),
    }


# =============================================================================
# SHAPr
# =============================================================================

def compute_shapr(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42,
) -> Dict[str, float]:
    '''
    Compute an approximate SHAPr score for membership inference risk.
    '''
    return


# =============================================================================
# Attribute Inference
# =============================================================================

def compute_attribute_inference(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_attribute: str,
) -> Dict[str, float]:
    '''
    Compute attribute inference risk for a specified sensitive attribute.
    The function evaluates how well an attacker can predict the sensitive attribute
    '''
    return


# =============================================================================
# Privacy Risk
# =============================================================================

def compute_privacy_risk(
    y_prob_train: np.ndarray,
    y_train: np.ndarray,
    y_prob_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute membership inference privacy risk using HolisticAI.

    Parameters
    ----------
    y_prob_train : np.ndarray
        Predicted probabilities for training samples.
    y_train : np.ndarray
        True labels for training samples.
    y_prob_test : np.ndarray
        Predicted probabilities for test samples.
    y_test : np.ndarray
        True labels for test samples.
    """
    
    return

# =============================================================================
# Accuracy Ratio (Data Minimization)
# =============================================================================

def compute_accuracy_ratio(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    model,
    X_test,
) -> Dict[str, float]:
    '''
    Compute accuracy ratio for data minimization techniques.
    '''
    return


# =============================================================================
# k-Anonymity
# =============================================================================

def compute_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
) -> Dict[str, float]:
    """
    Compute k-anonymity for a dataset.
    """
    counts = df[quasi_identifiers].value_counts() # How many times each combination of quasi-identifiers appears.

    if isinstance(counts, pd.Series):
        k_value = counts.min() if not counts.empty else 0
    else:
        k_value = counts

    return {
        "value": float(k_value),
        "quasi_identifiers": quasi_identifiers,
    }


# =============================================================================
# l-Diversity
# =============================================================================

def compute_l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
) -> Dict[str, float]:
    """
    Compute l-diversity for sensitive attributes in a dataset.
    """
    df_grouped = df.groupby(quasi_identifiers, as_index=False)
    result = {
            s: sorted([len(row["unique"]) for _, row in df_grouped[s].agg(["unique"]).dropna().iterrows()])
            for s in sensitive_attributes
        }

    all_vals = []

    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                all_vals.extend(v)

    min_l = min(all_vals) if all_vals else 0

    return {
        "value": float(min_l),
        "quasi_identifiers": quasi_identifiers,
        "sensitive_attributes": sensitive_attributes,
    }


# =============================================================================
# t-Closeness
# =============================================================================

def compute_t_closeness(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
) -> Dict[str, float]:
    """
    Compute t-closeness for sensitive attributes using Earth Mover's Distance (EMD) between local and global distributions.
    """
    max_t = 0.0

    for s in sensitive_attributes:

        global_dist = df[s].value_counts(normalize=True)

        grouped = df.groupby(quasi_identifiers)

        for _, group in grouped:

            local_dist = group[s].value_counts(normalize=True)

            aligned = global_dist.index.union(local_dist.index)
            
            # tvd = 0.5 * np.sum(np.abs(g - l)) # Total Variation Distance.
            # distances.append(tvd)
            g = global_dist.reindex(aligned, fill_value=0)
            l = local_dist.reindex(aligned, fill_value=0)

            g_cdf = np.cumsum(g.values)
            l_cdf = np.cumsum(l.values)

            emd = float(np.sum(np.abs(g_cdf - l_cdf)))
            max_t = max(max_t, emd)

    return {   
        "value": max_t, 
        "quasi_identifiers": quasi_identifiers, 
        "sensitive_attributes": sensitive_attributes}

