from __future__ import annotations

import contextlib
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import shap

def _safe_import_shap():
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Explainability requires the optional dependency 'shap'. "
            "Install with: pip install shap"
        ) from exc
    return shap

from sklearn.base import clone
from typing import Dict, Any
from holisticai.utils.surrogate_models import get_features, get_number_of_rules

import random
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence

from holisticai.inspection import compute_partial_dependence, compute_permutation_importance, compute_conditional_permutation_importance
from holisticai.utils import BinaryClassificationProxy

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier, is_regressor


# ============================================================
# Silence SHAP
# ============================================================
logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_shap_noise():
    """Silence stdout/stderr + warnings emitted by SHAP/Numba/tqdm."""
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

def _ensure_dataframe(X):
    ''' Ensure X is a pandas DataFrame with column names. If it's already a DataFrame, return as is. If it's a numpy array, convert to DataFrame with generic column names. '''
    if hasattr(X, "columns"):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _predict_fn_for_permutation(model):
    """
    Assume sklearn-compatible estimator: fit, predict, and (optionally) predict_proba.
    Prefer predict_proba when available so SHAP explains probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba
    return model.predict


def shap_based_metrics(
    *,
    model,
    X,
    n_samples: int = 50,
    shap_threshold: float = 1e-3,
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    """
    Compute SHAP-derived explainability metrics on a subsample of X using ONLY
    PermutationExplainer (algorithm="permutation") and assuming sklearn compatibility.

    Parameters
    ----------
    model: object
        sklearn-compatible model already fitted.
    X: pd.DataFrame or np.ndarray
        The input data for which to compute explainability metrics.
    n_samples: int, default=50
        The number of samples to use for computing explainability metrics.
    shap_threshold: float, default=1e-3
        The threshold for considering a feature as important.
    top_k: int, default=5
        The number of top features to consider for concentration metrics.
    seed: int, default=42
        The random seed for reproducibility.


    Returns
    -------
    Dict with the following keys:
    - n_features: number of features in the dataset
    - explainer: the SHAP explainer used (PermutationExplainer)
    - sample_size: the number of samples used for SHAP computations
    - shap_threshold: the threshold used for SHAP importance
    - sparsity: the fraction of features with importance above the threshold
    - feature_entropy: the normalized entropy of the global feature importance distribution. Higher means more uniform importance, lower means more concentrated.
    - top_k: the number of top features considered for concentration
    - topk_concentration: the fraction of total importance concentrated in the top_k features
    - interaction_strength: the estimated strength of feature interactions (based on SHAP interaction values). Higher means more interactions, lower means more additive.
    - base_values: the base values (mean of each feature) used for SHAP computations (shape: n_features). For faithfulness/monotonicity metrics, this can be used as the "baseline" input.
    - local_importances: the local SHAP importances for each sample and feature (shape: n_samples x n_features)     
    """

    shap = _safe_import_shap()
    X_full = _ensure_dataframe(X)

    # --- Subsample (row-wise) ---
    if n_samples and len(X_full) > n_samples:
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(X_full), size=int(n_samples), replace=False)
        X_eval = X_full.iloc[idx].copy()
    else:
        X_eval = X_full.copy()

    predict_fn = _predict_fn_for_permutation(model)
    explainer = shap.Explainer(predict_fn, X_eval, algorithm="permutation", seed=seed)

    with suppress_shap_noise():
        #shap_output = explainer(X_eval)
        safe_max_evals = min(100, X_eval.shape[1] * 10)
        shap_output = explainer(X_eval, max_evals=safe_max_evals)

    shap_values = np.asarray(shap_output.values)

    # Ensure 2D shape (n_samples, n_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    # If classifier returns probabilities, SHAP may be (n, d, n_outputs).
    # For binary proba (n, 2), SHAP often becomes (n, d, 2). Use positive class.
    if shap_values.ndim == 3:
        # Prefer class-1 contributions when available; otherwise take last output.
        cls_idx = 1 if shap_values.shape[2] > 1 else 0
        shap_values = shap_values[:, :, cls_idx]

    abs_vals = np.abs(shap_values)
    #n_features = int(abs_vals.shape[1]) if abs_vals.ndim == 2 else int(X_eval.shape[1])
    n_features = int(abs_vals.shape[1])

    # 1) Sparsity (fraction of features above threshold)
    active = abs_vals > float(shap_threshold) # boolean mask of shape (n_samples, n_features)
    sparsity = float(active.sum(axis=1).mean() / max(n_features, 1)) # sum over features, then average over samples, then divide by total features

    if abs_vals.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {abs_vals.shape}")

    # 2) Global feature entropy (normalized)
    global_importance = abs_vals.mean(axis=0)
    total = float(global_importance.sum())
    d = int(len(global_importance))

    if total == 0.0 or d <= 1:
        entropy_norm = 0.0
    else:
        p = global_importance / total
        entropy = float(-(p * np.log(p + 1e-12)).sum()) # Shannon entropy with small epsilon to avoid log(0)
        entropy_norm = float(entropy / float(np.log(d))) # Highest when all features are equally important, lowest when one dominates

    # 3) Top-K concentration
    k = int(min(max(int(top_k), 1), d))
    sorted_imp = np.sort(global_importance)[::-1]
    topk_concentration = 0.0 if total == 0.0 else float(sorted_imp[:k].sum() / total)

    # --- Interaction strength calculation ---
    # interaction_strength_value = np.nan
    # try:
    #     # Convert to 3D tensor if not (PermutationExplainer returns 2D)
    #     # Create a diagonalized tensor to simulate interactions
    #     # This is approximate; real SHAP interaction requires TreeExplainer
    #     if shap_values.ndim == 2:
    #         # Create a tensor (n_samples, n_features, n_features) with diagonal = main effect
    #         n_samples, n_features = shap_values.shape
    #         shap_int = np.zeros((n_samples, n_features, n_features))
    #         for i in range(n_features):
    #             shap_int[:, i, i] = shap_values[:, i]
    #     else:
    #         shap_int = shap_values  # if already 3D


    #     # explainer = shap.TreeExplainer(model)
    #     # shap_int = explainer.shap_interaction_values(X)
    #     # shap_int = np.abs(shap_int)
    #     # if isinstance(shap_int, list):
    #     #     shap_int = shap_int[1] if len(shap_int) > 1 else shap_int[0]


    #     total = np.abs(shap_int).sum()
    #     main_effect = np.sum(np.abs(np.diagonal(shap_int, axis1=1, axis2=2)))
    #     interaction_strength_value = float((total - main_effect) / total) if total != 0 else 0.0
    # except Exception:
    #     raise ValueError("Error computing interaction strength. Ensure SHAP values are in expected format.")

    # --- Also returns base and local values ---
    base_values = np.mean(X_eval.values, axis=0)
    local_importances = shap_values

    return {
        "n_features": float(n_features),
        "explainer": "PermutationExplainer",
        "sample_size": float(len(X_eval)),
        "shap_threshold": float(shap_threshold),

        "sparsity": sparsity,
        "feature_entropy": entropy_norm,
        "top_k": float(k),
        "topk_concentration": topk_concentration,
        #"interaction_strength": float(interaction_strength_value),
        "base_values": base_values,
        "local_importances": local_importances,
        "global_imps_array": global_importance,
    }


# ============================================================
# Structural Explainability Metrics
# ============================================================

def algorithm_class(model, model_type: str | None = None) -> dict:
    '''
    Identify the class of the model (e.g., "RandomForest", "SVM", "NeuralNetwork")
    
    Parameters
    ----------
    model: object
        The model for which to identify the class.
    model_type: str, optional
        If provided, use this string as the model class instead of inferring from the model object.

    Returns
    -------
    Dict with the following keys:
    - model_type: the identified class of the model (e.g., "RandomForest", "SVM", "NeuralNetwork")
    '''
    model_name = model_type if model_type is not None else type(model).__name__
    return {
        "model_type": model_name,
    }


def correlated_features(X_train : pd.DataFrame | np.ndarray, X_test: pd.DataFrame | np.ndarray, high_cor=0.95):
    '''
    Calculate the percentage of features that are highly correlated with at least one other feature, based on a combined correlation matrix of X_train and X_test.
    
    Parameters
    ----------
    X_train: pd.DataFrame or np.ndarray
        The training data.
    X_test: pd.DataFrame or np.ndarray
        The test data.
    high_cor: float, default=0.95
        The correlation threshold above which features are considered highly correlated.

    Returns
    -------
    Dict with the following keys:
    - value: the percentage of features that are highly correlated with at least one other feature.
    '''

    X_train = _ensure_dataframe(X_train)
    X_test = _ensure_dataframe(X_test)

    X_comb = pd.concat([X_train, X_test])
    corr_matrix = X_comb.corr().abs() # shape (n_features, n_features) with absolute correlation values

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [c for c in upper.columns if any(upper[c] > high_cor)]
    pct_corr = len(to_drop) / max(len(X_comb.columns), 1)

    return {
        "value": float(pct_corr),
        "highly_correlated_features": to_drop,
        "threshold": float(high_cor),
    }


def model_size(X_train: pd.DataFrame | np.ndarray):
    '''
    Calculate the number of features in the dataset.

    Parameters
    ----------
    X_train: pd.DataFrame or np.ndarray
        The training data.

    Returns
    -------
    Dict with the following keys:
    - value: the number of features in the dataset.
    '''
    X_train = _ensure_dataframe(X_train)
    n_features = X_train.shape[1]

    return {
        "value": int(n_features),
    }


# def feature_relevance(
#     model,
#     X_train,
#     y_train,
#     threshold_outlier=0.03,
# ):
#     # if hasattr(model, "coef_"):
#     #     model.fit(X_train, y_train)
#     #     importance = np.abs(model.coef_[0])

#     # elif hasattr(model, "feature_importances_"):
#     #     importance = np.abs(model.feature_importances_)

#     # else:
#     #     return {"value": np.nan}

#     if hasattr(model, "feature_importances_"):
#         importance = np.abs(model.feature_importances_)
#     elif hasattr(model, "coef_"):
#         importance = np.abs(model.coef_).flatten()
#     else:
#         raise ValueError("Model does not provide feature importances.")
    
#     q1, q3 = np.percentile(importance, [25, 75])
#     iqr = q3 - q1
#     lower = q1 - 1.5 * iqr
#     upper = q3 + 1.5 * iqr

#     n_outliers = np.sum((importance < lower) | (importance > upper))

#     cumulative = np.cumsum(np.sort(importance)[::-1])
#     total = importance.sum()

#     # pct_dominant = np.sum(cumulative < 0.6 * total) / len(importance)

#     # pct_dominant = (np.searchsorted(cumulative, 0.6 * total) + 1) / len(importance)

#     # Number of features needed to reach 60% cumulative importance
#     n_dominant = np.searchsorted(cumulative, 0.6 * total) + 1
    
#     # Percentage of features that do NOT reach 60%
#     pct_marginal = 1 - n_dominant / len(importance)

#     return {
#         "value": float(pct_marginal),
#         "n_outliers": int(n_outliers),
#         "importances": importance.tolist(),
#     }
#     # dist_score = np.digitize(pct_dist, thresholds, right=False) + 1 
    
#     # if n_outliers/len(importance) >= threshold_outlier (0.03):
#     #     dist_score -= penalty_outlier (0.5)

#     # NECESITA FIT

def feature_relevance(
    model,
    global_imps_array : np.ndarray | None,
    threshold_outlier=0.03, 
):
    '''
    Calculate the percentage of features that are considered irrelevant based on their importance being below a specified threshold. 
    If SHAP global importances are available, use them; otherwise, fall back to model-provided feature importances.

    Parameters
    ----------
    model: object
        The model for which to calculate feature relevance. Must have either feature_importances_ or coef_ attribute if SHAP importances are not provided.
    global_imps_array: np.ndarray or None
        An array of global feature importances (e.g., from SHAP). If None, the function will attempt to use model-provided importances.
    threshold_outlier: float, default=0.03
        The importance threshold below which features are considered irrelevant.

    Returns
    -------
    Dict with the following keys:
    - value: the percentage of features that are considered irrelevant.
    - threshold: the importance threshold used to determine irrelevance.
    - n_outliers: the number of features considered irrelevant.
    - importances: the array of feature importances used for the calculation.
    '''
    if global_imps_array is not None:
        importance = np.abs(global_imps_array)
    else:
        if hasattr(model, "feature_importances_"):
            importance = np.abs(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not provide feature importances.")

    # Identify irrelevant features according to threshold
    irrelevant_features = np.sum(importance <= threshold_outlier)

    # Percentage of irrelevant features
    pct_irrelevant = irrelevant_features / len(importance)

    return {
        "value": float(pct_irrelevant),
        "threshold": float(threshold_outlier),
        "n_outliers": int(irrelevant_features),
        "importances": importance.tolist(),
    }


# def performance_difference(f, g, X_test, y_test):

#     y_f = f.predict(X_test)
#     y_g = g.predict(X_test)

#     acc_f = accuracy_score(y_test, y_f)
#     acc_g = accuracy_score(y_test, y_g)

#     D = acc_f - acc_g

#     return {
#         "value": float(D),
#         "perf_original": float(acc_f),
#         "perf_explainer": float(acc_g),
#     }

    
##########
## HOLISTICAI
#########

# =============================================================================
# Alpha Score
# =============================================================================

def alpha_score(feature_importances: list, alpha: float = 0.8) -> float:
    '''
    Calculate the alpha score, which measures the concentration of feature importance in the top features.
    The alpha score is defined as the fraction of features that need to be included to reach a cumulative 
    importance of alpha (e.g., 0.8). A lower alpha score indicates that fewer features are needed to reach 
    the threshold, suggesting a more concentrated importance distribution.
    (e. g. if feature importance values are [0.4, 0.1, 0.01, 0.01, 0] and alpha=0.8, the cumulative importance is 0.52, so the threshold is 0.8*0.52=0.416, which is reached by the first feature alone, so the alpha score would be 1/5=0.2)
    
    Parameters
    ----------
    feature_importances: list
        A list of feature importance values (e.g., from SHAP or model coefficients).
    alpha: float, default=0.8
        The cumulative importance threshold to reach (e.g., 0.8 for 80%).

    Returns
    -------
    Dict with the following keys:
    - value: the computed alpha score, where lower values indicate that fewer features are needed to reach the cumulative importance threshold.
    - feature_importances: the input list of feature importance values.
    - alpha: the cumulative importance threshold.
    - n_features: the total number of features.
    - n_top_features: the number of top features needed to reach the threshold.
    ''' 
    if not feature_importances or feature_importances is None:
        raise ValueError("Feature importances list is empty, cannot compute alpha score.")
    
    vals = np.abs(np.array(feature_importances, dtype=float))
    n_total_features = len(feature_importances) # BEFORE was set to len(vals), but we want the original number of features, not the length of the array after filtering out zeros.
    
    if n_total_features == 0 or np.sum(vals) == 0.0:
        raise ValueError("Feature importances array is empty or all values are zero, cannot compute alpha score.")

    vals_sorted = np.sort(vals)[::-1]  # Sort in descending order
    cum_sum = np.cumsum(vals_sorted)  # Cumulative sum of sorted importances
    threshold = alpha * np.sum(vals_sorted) # Total importance multiplied by alpha to get the threshold for cumulative importance
    idx = np.searchsorted(cum_sum, threshold)  # Find the index where cumulative importance reaches or exceeds the threshold
    
    value = (idx + 1) / n_total_features
    return {"value": value, "feature_importances": feature_importances, "alpha": float(alpha), "n_features": n_total_features, "n_top_features": int(idx + 1)}


# =============================================================================
# Spread Ratio & Spread Divergence
# =============================================================================

def _spread_base(feature_importances: list, divergence: bool = True) -> float:
    '''
    Calculate the base spread metric (either ratio or divergence). 
    It calculates the distribution of feature importance values and compares it to a uniform distribution.
    If divergence is True, it calculates the Jensen-Shannon divergence; if False, it calculates the ratio of entropies.
    
    Parameters
    ----------
    feature_importances: list
        A list of feature importance values.
    divergence: bool, default=True
        If True, calculate spread divergence; otherwise, calculate spread ratio.

    Returns
    -------
    Dict with the following keys:
    - value: the computed spread metric (divergence or ratio).
    - feature_importances: the input list of feature importance values.

    '''
    if not feature_importances or feature_importances is None:
        raise ValueError("Feature importances list is empty, cannot compute spread metric.")
    tol = 1e-8
    vals = np.abs(np.array(feature_importances, dtype=float)) # BEFORE without abs
    
    if len(vals) == 0 or np.sum(vals) < tol:
        raise ValueError("Feature importances array is empty or all values are zero, cannot compute spread metric.")
    if len(vals) == 1:
        raise ValueError("Feature importances array has only one feature, cannot compute spread metric.")

    weights = vals / np.sum(vals)
    equal_weights = np.ones(len(vals)) / len(vals)

    if divergence:
        metric = jensenshannon(weights, equal_weights, base=2)
    else:
        metric = entropy(weights) / entropy(equal_weights)
    return {"value": float(metric), "feature_importances": feature_importances}

def spread_ratio(feature_importances: list) -> float:
    '''
    Calculate the spread ratio, which measures the evenness of feature importance distribution.
    
    Parameters
    ----------
    feature_importances: list
        A list of feature importance values.

    Returns
    -------
    dict with the following keys:
    - value: the spread ratio, where higher values indicate a more even distribution of feature importance
    - feature_importances: the input list of feature importance values.
    '''
    return _spread_base(feature_importances, divergence=False)

def spread_divergence(feature_importances: list) -> float:
    '''
    Calculate the spread divergence, which measures the dissimilarity of feature importance distribution from an even distribution.
    
    Parameters
    ----------
    feature_importances: list
        A list of feature importance values.

    Returns
    -------
    dict with the following keys:
    - value: the spread divergence, where higher values indicate a more uneven distribution.
    - feature_importances: the input list of feature importance values.
    '''
    return _spread_base(feature_importances, divergence=True)


# =============================================================================
# Position Parity
# =============================================================================

def position_parity(conditional_rankings: dict, global_ranking: list) -> float:
    '''
    Calculate the position parity, which measures how well the conditional feature rankings align with the global feature ranking.
    For each group in the conditional rankings, it calculates the cumulative match of the conditional ranking with
    the global ranking and averages this across groups. A higher position parity indicates better alignment between conditional and global rankings.
    Take into account is cumulative, so that matches at higher ranks contribute more to the score than matches at lower ranks.

    Parameters
    ----------
    conditional_rankings: dict
        A dictionary where keys are group names and values are lists of feature names ranked by importance for that group.
    global_ranking: list
        A list of feature names ranked by importance globally.

    Returns
    -------
    dict with the following keys:
    - value: the computed position parity score, where higher values indicate better alignment between conditional and
      global rankings.
    - conditional_rankings: the input dictionary of conditional rankings.
    - global_ranking: the input list of global ranking.
    - conditional_position_parity: a dictionary with the average cumulative match for each group.
    '''
    if not global_ranking or global_ranking is None:
        raise ValueError("Global ranking list is empty, cannot compute position parity.")
    
    conditional_position_parity = {}
    for group_name, cond_features in conditional_rankings.items():
        match_order = [c == r for c, r in zip(cond_features, global_ranking)] # boolean list indicating if the conditional feature at each position matches the global feature at that position
        if not match_order: continue
        m_order_cum = np.cumsum(match_order) / np.arange(1, len(match_order) + 1) # cumulative average of matches up to each position
        conditional_position_parity[group_name] = np.mean(m_order_cum) # average cumulative match across all positions for this group
        
    if not conditional_position_parity or len(conditional_position_parity) == 0:
        raise ValueError("No valid conditional rankings provided to compute position parity.")
    
    value = np.mean(list(conditional_position_parity.values()))

    return {"value": float(value), "conditional_rankings": conditional_rankings, "global_ranking": global_ranking, "conditional_position_parity": conditional_position_parity}


# =============================================================================
# Rank Alignment
# =============================================================================

def _get_top_alpha_features(importances_dict: dict, alpha: float) -> set:
    sorted_items = sorted(importances_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    total = sum(abs(v) for k, v in sorted_items)
    cum = 0
    top = []
    for k, v in sorted_items:
        top.append(k)
        cum += abs(v)
        if cum >= alpha * total:
            break
    return set(top)

def rank_alignment(conditional_importances: dict, global_importances: dict, alpha: float = 0.8, aggregation: bool = True):
    '''
    Calculate the rank alignment, which measures the similarity of the top alpha features between conditional and global feature importance distributions.
    For each group in the conditional importances, it identifies the top alpha features and compares them to the top alpha features of the global importances using Jaccard similarity. 
    The final score is either the average similarity across groups or the list of similarities for each group. 
    Note that is a set-based comparison, so it does not take into account the order of features within the top alpha, only whether they are included or not.
    
    Parameters
    ----------
    conditional_importances: dict
        A dictionary where keys are group names and values are dictionaries of feature importances for that group.
    global_importances: dict
        A dictionary of global feature importances.
    alpha: float, default=0.8
        The cumulative importance threshold to identify the top features (e.g., 0.8 for
        80% of total importance).
    aggregation: bool, default=True 
        If True, return the average similarity across groups; if False, return the list of similarities for each group.
    
    Returns
    -------
    If aggregation is True:
    - dict with the following keys
        - value: the average Jaccard similarity of top alpha features between conditional and global importances across groups.
        - top_global_features: the list of top alpha features from global importances.
        - top_conditional_features: a dictionary where keys are group names and values are lists of
        - top alpha features for each group from conditional importances.
        - conditional_importances: the input dictionary of conditional importances.
        - global_importances: the input dictionary of global importances.
    If aggregation is False:
    - list of dictionaries, each with the following keys:
        - group: the group name.
        - value: the Jaccard similarity of top alpha features between conditional and global importances for this group.
        - top_global_features: the list of top alpha features from global importances.
        - top_conditional_features: the list of top alpha features for this group from conditional importances.
        - conditional_importances: the input dictionary of conditional importances.
        - global_importances: the input dictionary of global importances.
    '''
    if not global_importances or global_importances is None:
        raise ValueError("Global importances dictionary is empty, cannot compute rank alignment.")
    
    top_global = _get_top_alpha_features(global_importances, alpha)
    if not top_global:
        raise ValueError("No features with non-zero importance found in global importances.")

    top_global_list = list(top_global)
    
    similarities = []
    top_conditionals = {} # Dictionary for storing the top alpha features of each group
    detailed_results = [] # List for storing detailed results for each group when aggregation is False
    
    for group, cond_imps in conditional_importances.items():
        if not cond_imps or cond_imps is None:
            raise ValueError(f"Conditional importances for group '{group}' is empty or None, cannot compute rank alignment.")
        top_cond = _get_top_alpha_features(cond_imps, alpha)
        top_conditionals[group] = list(top_cond)
        
        intersection = len(top_global.intersection(top_cond))
        union = len(top_global.union(top_cond))
        sim = intersection / union if union > 0 else 0.0
        similarities.append(sim)
        
        detailed_results.append({
            "group": group,
            "value": sim, 
            "top_global_features": top_global_list,
            "top_conditional_features": list(top_cond),
            "conditional_importances": conditional_importances, 
            "global_importances": global_importances
        })

    if aggregation:
        return {
            "value": float(np.mean(similarities)) if similarities else 1.0, 
            "top_global_features": top_global_list,
            "top_conditional_features": top_conditionals,
            "conditional_importances": conditional_importances,  
            "global_importances": global_importances
        }
    
    return detailed_results


# =============================================================================
# XAI Ease Score
# =============================================================================

def calculate_discrete_derivative(y_values):
    dy = np.diff(y_values)
    dx = np.ones_like(dy)
    return dy / dx

def cosine_similarity(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def compare_tangents(points):
    num_sections = 3
    n = len(points)
    if n < num_sections:
        return (1, 1), True

    cut1 = n // 3
    cut2 = 2 * n // 3
    sections = [points[:cut1 + 1], points[cut1:cut2 + 1], points[cut2:]]
    slopes = []
    
    for section in sections:
        if len(section) > 1:
            avg_slope = np.mean(calculate_discrete_derivative(section))
        else:
            avg_slope = 0
        slopes.append(avg_slope + 1e-5)

    similarities = [cosine_similarity([slopes[i]], [slopes[i + 1]]) for i in range(len(slopes) - 1)]
    return similarities, False

def xai_ease_score(pdp_averages: dict, global_ranked_features: list) -> float:
    '''
    Calculate the XAI Ease Score, which evaluates the ease of understanding the relationship between features and
    model predictions based on the shape of partial dependence plots (PDPs) for the top globally ranked features.
    For each top feature, it analyzes the average PDP values and compares the tangents of the PDP curve across three sections (beginning, middle, end).
    The score is based on the similarity of the tangents across sections, with the intuition that more similar tangents indicate a simpler relationship that is easier to understand.
    
    Parameters
    ----------
    pdp_averages: dict
        A dictionary where keys are feature names and values are lists of average PDP values for those features.
    global_ranked_features: list   
        A list of feature names ranked by global importance, with the most important features first.

    Returns
    -------
    float: the XAI Ease Score, where higher values indicate that the relationships between features and predictions are easier to understand based on the shape of the PDPs.
    '''
    if not global_ranked_features or global_ranked_features is None:
        raise ValueError("Global ranked features list is empty, cannot compute XAI Ease Score.")
    
    threshold = 0.0
    levels = ["Hard", "Medium", "Easy"]
    scores_list = []

    for feat in global_ranked_features:
        if feat not in pdp_averages: continue
        r, few_points = compare_tangents(pdp_averages[feat])
        score_val = sum([1 for rr in r if rr > threshold])
        scores_list.append({"feature": feat, "scores": levels[score_val]})

    if not scores_list: raise ValueError("No features with PDP averages found in global ranked features.")

    df = pd.DataFrame(scores_list)
    counts = df.groupby("scores")["feature"].count()
    total = counts.sum()

    values = []
    for level in levels: # levels are ordered from hardest to easiest, so that higher levels contribute more to the score
        cnt = counts.get(level, 0)
        prop = cnt / total
        values.append(levels.index(level) * prop)

    value = sum(values)/2
    return {"value": value, "pdp_averages": pdp_averages, "global_ranked_features": global_ranked_features} # max_score is 2


###########################
## AIX 360
###########################

def faithfulness_metric(model, x: np.ndarray, coefs: np.ndarray, base: np.ndarray) -> dict:
    """ This metric evaluates the correlation between the importance assigned by the interpretability algorithm
    to attributes and the effect of each of the attributes on the performance of the predictive model.
    The higher the importance, the higher should be the effect, and vice versa. The metric evaluates this by
    incrementally removing each of the attributes deemed important by the interpretability metric, and
    evaluating the effect on the performance, and then calculating the correlation between the weights (importance)
    of the attributes and corresponding model performance. [#]_

    Parameters:
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        x (numpy.ndarray): row of data.
        coefs (numpy.ndarray): coefficients (weights) corresponding to attribute importance.
        base ((numpy.ndarray): base (default) values of attributes

    Returns:
        float: correlation between attribute importance weights and corresponding effect on classifier.

    References:
    .. [#] `David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining
        neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors,
        Advances in Neural Information Processing Systems 31, pages 7775-7784. 2018.
        <https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf>`_
    """
    if coefs is None or base is None:
        raise ValueError("Coefficients and base values cannot be None.")
    
    # Ensure that all have the same length
    assert len(x) == len(coefs) == len(base)

    #find predicted class
    pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)), axis=1)[0]

    #find indexs of coefficients in decreasing order of value
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]

    if np.std(coefs) == 0 or np.std(pred_probs) == 0:
        corr = 0.0  # If there is no variation, there is no correlation
    else:
        corr = -np.corrcoef(coefs, pred_probs)[0, 1]

    if np.isnan(corr):
        raise ValueError("Correlation is NaN. Check if coefs and pred_probs have sufficient variation.")

    return {
        "value": float(corr),
        "pred_class": int(pred_class),
        "coefs": coefs,
        "pred_probs": pred_probs,
    }


def monotonicity_metric(model, x: np.ndarray, coefs: np.ndarray, base: np.ndarray) -> dict:
    """ This metric measures the effect of individual features on model performance by evaluating the effect on
    model performance of incrementally adding each attribute in order of increasing importance. As each feature
    is added, the performance of the model should correspondingly increase, thereby resulting in monotonically
    increasing model performance. [#]_

    Parameters:
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        x (numpy.ndarray): row of data.
        coefs (numpy.ndarray): coefficients (weights) corresponding to attribute importance.
        base ((numpy.ndarray): base (default) values of attributes

    Returns:
        bool: True if the relationship is monotonic.

    References:
        .. [#] `Ronny Luss, Pin-Yu Chen, Amit Dhurandhar, Prasanna Sattigeri, Karthikeyan Shanmugam, and
           Chun-Chen Tu. Generating Contrastive Explanations with Monotonic Attribute Functions. CoRR abs/1905.13565. 2019.
           <https://arxiv.org/pdf/1905.12698.pdf>`_
    """
    if coefs is None or base is None:
        raise ValueError("Coefficients and base values cannot be None.")
    
    #find predicted class
    pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)), axis=1)[0]

    x_copy = base.copy()

    #find indexs of coefficients in increasing order of value
    ar = np.argsort(coefs)
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy[ind] = x[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]

    monotone = bool(np.all(np.diff(pred_probs[ar]) >= 0))

    return {
        "value": int(monotone),
        "pred_class": int(pred_class),
        "coefs": coefs,
        "pred_probs": pred_probs,
    }


def infidelity(model, X_test: np.ndarray, feature_weights: np.ndarray) -> Dict[str, float]:
    """
    Computes the infidelity by evaluating whether perturbations to important features lead to proportional changes in model output.
    Implementation of https://arxiv.org/pdf/1901.09392.pdf, based on https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py

    Parameters
    ----------
    model: object
        The predictive model to evaluate. Must have a predict() method.
    X_test: pd.DataFrame or np.ndarray
        The test data for which to compute infidelity.
    feature_weights: np.ndarray
        The importance weights for each feature, typically obtained from an interpretability method. Should have the same number of rows as X_test and the same number of columns as features in X_test.

    Returns
    -------
    Dict with the following keys:
    - value: the computed infidelity score, where lower values indicate better fidelity of the feature importance weights to the model's behavior.

    """
    if feature_weights is None:
        raise ValueError("Feature weights cannot be None.")
    
    X_np = np.asarray(X_test, dtype=float)
    num_datapoints, num_features = X_np.shape
    infids = []
    
    def get_exp(ind, exp):
        return exp[ind.astype(int)]

    def set_zero_infid(array, size, point):
        arr_copy = array.copy()
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + arr_copy[ind]
        randd = np.minimum(arr_copy[ind], randd)
        randd = np.maximum(arr_copy[ind] - 1.0, randd)
        arr_copy[ind] -= randd
        return np.concatenate((arr_copy, ind, randd))
    
    num_datapoints = min(len(X_np), len(feature_weights))
    for i in range(num_datapoints):
        num_reps = 100
        x_orig = np.tile(X_np[i], [num_reps, 1])
        x = X_np[i]
        expl_copy = np.copy(feature_weights[i])
        
        val = np.apply_along_axis(set_zero_infid, 1, x_orig, num_features, num_features)
        x_ptb = np.apply_along_axis(set_zero_infid, 1, x_orig, num_features, num_features)
        x_ptb = val[:, :num_features]
        ind = val[:, num_features: 2*num_features]
        rand = val[:, 2*num_features: 3*num_features]
        
        exp_sum = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, expl_copy), axis=1)
        ks = np.ones(num_reps)
        
        pdt = model.predict(np.array([x]))[0]
        pdt_ptb = model.predict(x_ptb)
        pdt_diff = pdt - pdt_ptb

        # Avoid division by zero in beta computation.
        denominator = np.mean(ks * exp_sum * exp_sum)
        beta = np.mean(ks * pdt_diff * exp_sum) / (denominator if denominator != 0 else 1e-10)
        exp_sum *= beta
        
        infid = np.mean(ks * np.square(pdt_diff - exp_sum)) / np.mean(ks)
        infids.append(infid)

    mean_infid = float(np.mean(infids))
    if np.isnan(mean_infid):
        raise ValueError("Infidelity is NaN. Check if model predictions and feature weights are valid.")
    return {"value": mean_infid}

def number_of_rules(tree_model) -> dict:
    '''
    Calculate the number of rules in a tree-based model. For Random Forests, it calculates the mean number of rules across all trees.
    
    Parameters
    ----------
    tree_model: object
        The tree-based (or rule-based) model for which to calculate the number of rules. Must have either get_n_leaves() method or be a Random Forest with estimators_ attribute.
    
    Returns
    ------- 
    Dict with the following keys:
    - value: the number of rules in the model (or mean number of rules across trees for Random Forests).
    '''
    if hasattr(tree_model, "rules_"):
        return {"value": float(len(tree_model.rules_))}
    
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [number_of_rules(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals))}
        
    if hasattr(tree_model, "get_n_leaves"):
        n_rules = tree_model.get_n_leaves()
        return {"value": float(n_rules)}
    else:
        raise ValueError("Model does not provide number of rules.")


def average_rule_length(tree_model) -> dict:
    '''
    Calculate the average rule length (depth) in a tree-based model. For Random Forests, it calculates the mean average rule length across all trees.
    
    Parameters
    ----------
    tree_model: object
        The tree-based (or rule-based) model for which to calculate the average rule length. Must have a tree_ attribute with children_left and children_right arrays, or be a Random Forest with estimators_ attribute.
    
    Returns
    -------
    Dict with the following keys:
    - value: the average rule length in the model (or mean average rule length across trees for Random Forests).

    '''

    if hasattr(tree_model, "rules_"):
        rules = tree_model.rules_
        n_rules = len(rules)
        lengths = [len(rule.conditions) for rule in rules] if n_rules > 0 else [0]
        return {"value": float(np.mean(lengths))}

    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [average_rule_length(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals))}
        
    if not hasattr(tree_model, "tree_"):
        raise ValueError("Model does not provide tree structure.")
        
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right

    def node_depth(node, depth=0):
        if children_left[node] == children_right[node]:
            return [depth]
        depths = []
        if children_left[node] != -1:
            depths += node_depth(children_left[node], depth + 1)
        if children_right[node] != -1:
            depths += node_depth(children_right[node], depth + 1)
        return depths

    depths = node_depth(0)
    avg_depth = np.mean(depths) if depths else 0.0

    return {"value": float(avg_depth)}

def tree_depth(tree_model) -> dict:
    '''
    Calculate the maximum depth of a tree-based model. For Random Forests, it calculates the mean depth across all trees.

    Parameters
    ----------
    tree_model: object
        The tree-based model for which to calculate the tree depth. Must have a get_depth() method, or be a Random Forest with estimators_ attribute.
    
    Returns
    -------
    Dict with the following keys:
    - value: the maximum depth of the tree model (or mean depth across trees for Random Forests).
    '''
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [tree_depth(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals))}
        
    if hasattr(tree_model, "get_depth"):
        depth = tree_model.get_depth()
        return {"value": float(depth)}
    else:
        raise ValueError("Model does not provide tree depth.")
    
# =============================================================================
# Helper Functions for Trees
# =============================================================================
'''
Examples
--------
>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from holisticai.explainability.metrics import tree_number_of_features
>>> X, y = load_iris(return_X_y=True)
>>> clf = DecisionTreeClassifier()
>>> clf.fit(X, y)
>>> tree_number_of_features(clf.tree_)
'''

def is_leaf(node_index: int, tree) -> bool:
    """
    Check if a node is a leaf.

    Parameters
    ----------
    node_index : int
        The index of the node to check.
    tree : Tree
        The tree to check the node in.

    Returns
    -------
    bool
        Whether the node is a leaf or not.
    """
    return tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1

def get_cuts_counts(node_index: int, tree, cuts: list, counts: list, cur_set: set):
    """
    Get the cuts and counts of a tree.

    Parameters
    ----------
    node_index : int
        The index of the node to start from.
    tree : Tree
        The tree to get the cuts and counts from.
    cuts : list
        The list to store the cuts.
    counts : list
        The list to store the counts.
    cur_set : set
        The set to store the current cuts.

    Returns
    -------
    list
        The list of cuts.
    list
        The list of counts.
    """
    if is_leaf(node_index, tree):
        cuts.append(len(cur_set))
        counts.append(tree.n_node_samples[node_index])
    else:
        children_left_set = cur_set.copy()
        children_left_set.add((tree.feature[node_index], -1))
        children_right_set = cur_set.copy()
        children_right_set.add((tree.feature[node_index], 1))

        if tree.children_left[node_index] != -1:
            get_cuts_counts(tree.children_left[node_index], tree, cuts, counts, children_left_set)
        if tree.children_right[node_index] != -1:
            get_cuts_counts(tree.children_right[node_index], tree, cuts, counts, children_right_set)
    return cuts, counts

def get_depths_counts(node_index: int, tree, depths: list, counts: list, h: int = 0):
    """
    Get the depths and counts of a tree.

    Parameters
    ----------
    node_index : int
        The index of the node to start from.
    tree : Tree
        The tree to get the depths and counts from.
    depths : list
        The list to store the depths.
    counts : list
        The list to store the counts.
    h : int, default=0
        The current depth.

    Returns
    -------
    list
        The list of depths.
    list
        The list of counts.
    """
    if is_leaf(node_index, tree):
        depths.append(h)
        counts.append(tree.n_node_samples[node_index])

    if tree.children_left[node_index] != -1:
        get_depths_counts(tree.children_left[node_index], tree, depths, counts, h + 1)
    if tree.children_right[node_index] != -1:
        get_depths_counts(tree.children_right[node_index], tree, depths, counts, h + 1)

    return depths, counts

# =============================================================================
# Tree / Structural Surrogate Metrics
# =============================================================================

def weighted_average_depth(tree_model) -> Dict[str, float]:
    """
    Weighted Average Depth calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree_model: Tree
        The tree to calculate the weighted average depth of.

    Returns
    -------
    dict
        A dictionary containing the weighted average depth of the tree.

    Reference
    ----------
        Laber, E., Murtinho, L., & Oliveira, F. (2023).
        Shallow decision trees for explainable k-means clustering.
        Pattern Recognition, 137, 109239.
    """
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [weighted_average_depth(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals)), "list": "Only with decision trees for legibility"}

    tree_obj = getattr(tree_model, "tree_", tree_model)
    if tree_obj is None or not hasattr(tree_obj, 'children_left'): 
        raise ValueError("Model must have attribute 'tree_'.")
        
    depths, counts = get_depths_counts(0, tree_obj, [], [])
    n_samples = sum(counts)
    if n_samples == 0 : raise ValueError("No samples to calculate weighted average depth.")
    all_list = [f"{d}*({c}/{n_samples})" for d, c in zip(depths, counts)]
    return {"value": float((np.array(depths) * (np.array(counts) / n_samples)).sum()), "list": all_list}

def weighted_average_explainability_score(tree_model) -> Dict[str, float]:
    """
    Weighted Average Explainability Score calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree_model: Tree
        The tree to calculate the weighted average depth of.

    Returns
    -------
    dict
        A dictionary containing the weighted average explainability score of the tree.

    Reference
    ----------
        Laber, E., Murtinho, L., & Oliveira, F. (2023).
        Shallow decision trees for explainable k-means clustering.
        Pattern Recognition, 137, 109239.
    """
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [weighted_average_explainability_score(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals)), "list": "Only with decision trees for legibility"}

    tree_obj = getattr(tree_model, "tree_", tree_model)
    if tree_obj is None or not hasattr(tree_obj, 'children_left'): 
        raise ValueError("Model must have attribute 'tree_'.")
        
    depths, counts = get_cuts_counts(0, tree_obj, [], [], set()) # cuts instead of depths (eliminates duplicates), but the same logic applies for weighting
    n_samples = sum(counts)
    if n_samples == 0: raise ValueError("No samples to calculate weighted average explainability score.")
    all_list = [f"{d}*({c}/{n_samples})" for d, c in zip(depths, counts)]
    return {"value": float((np.array(depths) * (np.array(counts) / n_samples)).sum()), "list": all_list }

def weighted_tree_gini(tree_model) -> Dict[str, float]: # ALGO DIFERENTE
    """
    Compute the weighted Gini index for the tree (WGNI).
    Reference value: 0.0

    Parameters
    ----------
    tree_model : Tree
        The tree to compute the weighted Gini index of.

    Returns
    -------
    dict
        A dictionary containing the weighted Gini index of the tree.
    """ 
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [weighted_tree_gini(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals))}

    tree_obj = getattr(tree_model, "tree_", tree_model)
    if tree_obj is None or not hasattr(tree_obj, 'children_left'): 
        raise ValueError("Model must have attribute 'tree_'.")
        
    is_classification = tree_obj.n_classes[0] > 1
    weighted_impurity = 0.0
    total_samples = tree_obj.n_node_samples[0]

    def accumulate_impurity(node_index):
        nonlocal weighted_impurity
        if is_leaf(node_index, tree_obj):
            node_samples = tree_obj.n_node_samples[node_index]
            if node_samples > 0:
                node_value = tree_obj.value[node_index, 0, :]
                if is_classification:
                    #impurity = 1.0 - np.sum((node_value / node_samples)**2)
                    total_node_weight = np.sum(node_value)
                    if total_node_weight > 0:
                        impurity = 1.0 - np.sum((node_value / total_node_weight)**2)
                    else:
                        impurity = 0.0
                else:
                    impurity = np.sum((node_value - np.mean(node_value)) ** 2) / node_samples
                weighted_impurity += (node_samples / total_samples) * impurity
        else:
            accumulate_impurity(tree_obj.children_left[node_index])
            accumulate_impurity(tree_obj.children_right[node_index])

    accumulate_impurity(0)
    return {"value": float(weighted_impurity)}

def tree_depth_variance(tree_model) -> Dict[str, float]:
    """
    Compute the variance of the depths of the leaves in the tree (TDV).
    Reference value: 0.0

    Parameters
    ----------
    tree_model : Tree
        The tree to compute the depth variance of.

    Returns
    -------
    dict
        A dictionary containing the variance of the leaf depths.

    """
    # If it's a Random Forest, calculate the mean across all trees
    if hasattr(tree_model, "estimators_"):
        vals = [tree_depth_variance(est)["value"] for est in tree_model.estimators_]
        return {"value": float(np.mean(vals)), "leaf_depths": "Only with decision trees for legibility"}

    tree_obj = getattr(tree_model, "tree_", tree_model)
    if tree_obj is None or not hasattr(tree_obj, 'children_left'): 
        raise ValueError("Model must have attribute 'tree_'.")
        
    depths, _ = get_depths_counts(0, tree_obj, [], [])
    if not depths: raise ValueError("No leaf nodes to calculate depth variance.")
    depths_arr = np.array(depths)
    return {"value": float(np.mean((depths_arr - np.mean(depths_arr)) ** 2)), "leaf_depths": depths_arr.tolist()}


def tree_number_of_features(surrogate) -> Dict[str, float]:
    """
    Calculates the number of features used in a decision tree model.

    Parameters
    ----------
        surrogate: A surrogate model, typically a decision tree, for which the number of features is to be calculated.

    Returns
    -------
        int: The number of features used in the surrogate model.
    """
# If it's a Random Forest, calculate the mean across all trees
    if hasattr(surrogate, "estimators_"):
        vals = [tree_number_of_features(est)["value"] for est in surrogate.estimators_]
        return {"value": float(np.mean(vals))}
        
    if surrogate is None: raise ValueError("Surrogate cannot be None.")
    features = get_features(surrogate.tree_)
    return {"value": float(len(np.unique(features[features >= 0])))}


# =============================================================================
# Ensemble XAI Consistency Metrics (Custom)
# =============================================================================

def to_scalar(val):
    """
    Converts numpy arrays or lists with 1 element to float.
    If it has multiple elements, returns the mean.
    """
    if isinstance(val, (np.ndarray, list)):
        val = np.array(val).flatten()
        if len(val) == 1:
            return float(val[0])
        return float(np.mean(val))
    return float(val)

def get_top_k(importance_dict, k, return_values=False):
    """
    Returns the k most important features according to the magnitude of their value.
    If return_values=True, returns a list of tuples (feature, importance).
    If return_values=False, returns a set with names (for Jaccard).
    """
    cleaned_dict = {feat: to_scalar(val) for feat, val in importance_dict.items()}
    sorted_feats = sorted(cleaned_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    if return_values:
        return sorted_feats[:k]
    return set([feat for feat, val in sorted_feats[:k]])

# ----------------------------
# XAI methods
# ----------------------------

def compute_lime(model, X, mode='classification', num_samples=20, seed=42):
    feature_names = X.columns.tolist()

    def predict_fn_wrapper(x_numpy):
        df_temp = pd.DataFrame(x_numpy, columns=feature_names)
        if mode == 'classification':
            return model.predict_proba(df_temp)
        else:
            return model.predict(df_temp)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=feature_names,
        mode=mode,
        verbose=False,
        random_state=np.random.RandomState(seed)
    )

    importances = {f: 0.0 for f in feature_names}

    for i in range(num_samples):
        exp = explainer.explain_instance(X.values[i], predict_fn_wrapper, num_samples=1000)
        local_list = exp.local_exp[1] if mode=='classification' else exp.local_exp[0]
        for feat_idx, weight in local_list:
            importances[feature_names[feat_idx]] += abs(weight)

    importances = {k: v/num_samples for k, v in importances.items()}
    return importances

# def compute_shap_custom(model, X, mode='classification'):
#     feature_names = X.columns.tolist()
#     background = X.iloc[:20, :]
#     test_samples = X.iloc[:20, :]

#     if hasattr(model, 'predict_proba') and mode == 'classification':
#         explainer = shap.KernelExplainer(model.predict_proba, background)
#         shap_values = explainer.shap_values(test_samples, silent=True, nsamples=100)
#         vals = np.abs(shap_values[1]).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
#     else:
#         explainer = shap.KernelExplainer(model.predict, background)
#         shap_values = explainer.shap_values(test_samples, silent=True, nsamples=100)
#         vals = np.abs(shap_values).mean(axis=0)

#     importances = dict(zip(feature_names, vals))
#     return importances

# def shap_importance_from_local(local_importances, feature_names):
#     abs_vals = np.abs(local_importances)
#     global_importance = abs_vals.mean(axis=0)
#     return dict(zip(feature_names, global_importance))

# def compute_pdp_importance(model, X, random_state=42):
#     # Check that the model is a valid estimator
#     if not (is_classifier(model) or is_regressor(model)):
#         if not (hasattr(model, 'predict') and hasattr(model, 'fit')):
#             raise ValueError("Model must be a fitted sklearn-compatible estimator for PDP computation.")

#     # Convertir a float para evitar errores de dtype
#     X_float = X.astype(float)
#     X_float = X_float.sample(n=100, random_state=random_state) if len(X_float) > 100 else X_float

#     importances = {}
#     for col in X_float.columns:
#         try:
#             pdp_result = partial_dependence(model, X_float, [col], kind='average', grid_resolution=20)
#             val = np.std(pdp_result['average'])
#             importances[col] = val
#         except Exception as e:
#             raise ValueError(f"Failed to compute PDP for feature '{col}': {e}")
#     return importances

# def compute_pfi(model, X, y, random_state=42):
#     r = permutation_importance(model, X, y, n_repeats=5, random_state=random_state)
#     importances = dict(zip(X.columns, r.importances_mean))
#     return importances

# def compute_lofo(model, X, y):
#     base_score = model.score(X, y)
#     importances = {}
#     for col in X.columns:
#         X_temp = X.copy()
#         X_temp[col] = np.random.permutation(X_temp[col].values)
#         score_drop = base_score - model.score(X_temp, y)
#         importances[col] = score_drop
#     return importances

# def compute_surrogate(model, X, mode='classification'):
#     targets = model.predict(X)
#     if mode == 'classification':
#         surrogate = DecisionTreeClassifier(max_depth=3)
#     else:
#         surrogate = DecisionTreeRegressor(max_depth=3)
#     surrogate.fit(X, targets)
#     importances = dict(zip(X.columns, surrogate.feature_importances_))
#     return importances

# ----------------------------
# XAI Consistency metric
# ----------------------------

def calculate_consistency_matrix(rankings, k=5):
    methods = list(rankings.keys())
    n = len(methods)
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for i in range(n):
        for j in range(n):
            # Here we call with return_values=False to calculate Jaccard
            set1 = get_top_k(rankings[methods[i]], k, return_values=False)
            set2 = get_top_k(rankings[methods[j]], k, return_values=False)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            matrix.loc[methods[i], methods[j]] = intersection / union if union > 0 else 0
    return matrix

def get_aggregated_score(matrix):
    """
    Computes the aggregated consistency score from the consistency matrix.
    """
    values = matrix.values[np.triu_indices(len(matrix), k=1)]
    return np.mean(values)

def xai_consistency(model, global_importances, pdp_std, X, k=5, mode='classification', random_state=42) -> Dict[str, Any]:
    '''
    Calculate the XAI Consistency Score, which evaluates the consistency of feature importance rankings across different 
    interpretability methods (LIME, SHAP, PDP) for the top K features. The score is based on the Jaccard similarity of the 
    top K features identified by each method, averaged across all pairs of methods. Higher scores indicate greater consistency 
    in identifying important features across methods.

    Parameters
    ----------
    model: object
        The predictive model for which to evaluate XAI consistency. Must be a fitted sklearn-compatible estimator.
    global_importances: dict
        A dictionary of global feature importances from SHAP, where keys are feature names and values
        are the importance scores. This is required for the consistency calculation.
    pdp_std: dict   
        A dictionary of feature importance scores derived from the standard deviation of partial dependence plots (PDPs), where keys are feature names and values are the standard deviation of the PDP values for those features.
    X: pd.DataFrame or np.ndarray
        The input data used for computing LIME explanations. Must have the same features as those in
        global_importances and pdp_std.
    k: int, optional    
        The number of top features to consider for the consistency calculation (default is 5).
    mode: str, optional
        The mode for LIME explanations, either 'classification' or 'regression' (default
        is 'classification').
    random_state: int, optional 
        The random state for reproducibility of LIME explanations (default is 42).

    Returns
    -------
    dict: A dictionary containing the following keys:
    - value: the computed XAI Consistency Score, where higher values indicate greater consistency in feature importance rankings across methods.
    - consistency_matrix: a matrix showing the pairwise Jaccard similarity of top K features
        between each pair of methods.
    - top_k_details: a string detailing the top K features and their importance values for each method, for interpretability.
    - rankings: a dictionary containing the feature importance rankings for each method (LIME, SHAP, PDP) used in the consistency calculation.
    '''
    np.random.seed(random_state)
    random.seed(random_state)

    # Initial validations
    if global_importances is None:
        raise ValueError("Global importances (SHAP) are required for XAI consistency computation.")

    if X is None or (hasattr(X, 'shape') and X.shape[0] == 0):
        raise ValueError("X data is required and cannot be empty.")

    # Ensure X is a DataFrame so they have column names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Convert to float to avoid dtype errors with int64
    X = X.astype(float)

    rankings = {}

    # LIME
    try:
        rankings['LIME'] = compute_lime(model, X, mode=mode, num_samples=20, seed=random_state)
    except Exception as e:
        raise ValueError(f"Failed to compute LIME importances: {e}")

    rankings['SHAP']= global_importances
    
    if pdp_std is not None:
        rankings['PDP'] = pdp_std

    matrix = calculate_consistency_matrix(rankings, k=k)
    score = get_aggregated_score(matrix)

    top_k_info = []
    for method, importances in rankings.items():
        # Extract the top K with their values
        top_k_list = get_top_k(importances, k, return_values=True)
        features_str = ", ".join([feat for feat, val in top_k_list])
        vals_str = ", ".join([f"{val:.3f}" for feat, val in top_k_list])

        info_string = f"Top-{k} features {method}: {features_str}, with importances: {vals_str}"
        top_k_info.append(info_string)

    return {
        "value": float(score),
        "consistency_matrix": matrix.to_dict(),
        "top_k_details": "\n".join(top_k_info),
        "rankings": rankings
    }