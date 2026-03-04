from __future__ import annotations

import contextlib
import logging
import os
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score

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
from holisticai.explainability.metrics import classification_explainability_metrics
from holisticai.explainability.metrics.local_feature_importance import classification_local_feature_importance_explainability_metrics
from holisticai.explainability.metrics.surrogate import regression_surrogate_explainability_metrics
from holisticai.explainability.metrics.surrogate import classification_surrogate_explainability_metrics

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


def compute_shap_based_metrics(
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

    Returns a dict with:
      - sparsity
      - feature_entropy
      - topk_concentration
      - n_features
      - explainer
      - sample_size
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
    explainer = shap.Explainer(predict_fn, X_eval, algorithm="permutation")

    with suppress_shap_noise():
        shap_output = explainer(X_eval)

    shap_values = np.asarray(shap_output.values)

    # If classifier returns probabilities, SHAP may be (n, d, n_outputs).
    # For binary proba (n, 2), SHAP often becomes (n, d, 2). Use positive class.
    if shap_values.ndim == 3:
        # Prefer class-1 contributions when available; otherwise take last output.
        cls_idx = 1 if shap_values.shape[2] > 1 else 0
        shap_values = shap_values[:, :, cls_idx]

    abs_vals = np.abs(shap_values)
    n_features = int(abs_vals.shape[1]) if abs_vals.ndim == 2 else int(X_eval.shape[1])

    # 1) Sparsity (fraction of features above threshold)
    active = abs_vals > float(shap_threshold)
    sparsity = float(active.sum(axis=1).mean() / max(n_features, 1))

    # 2) Global feature entropy (normalized)
    global_importance = abs_vals.mean(axis=0)
    total = float(global_importance.sum())
    d = int(len(global_importance))

    if total == 0.0 or d <= 1:
        entropy_norm = 0.0
    else:
        p = global_importance / total
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        entropy_norm = float(entropy / float(np.log(d)))

    # 3) Top-K concentration
    k = int(min(max(int(top_k), 1), d))
    sorted_imp = np.sort(global_importance)[::-1]
    topk_concentration = 0.0 if total == 0.0 else float(sorted_imp[:k].sum() / total)

    return {
        "sparsity": sparsity,
        "feature_entropy": entropy_norm,
        "topk_concentration": topk_concentration,
        "n_features": float(n_features),
        "explainer": "PermutationExplainer",
        "sample_size": float(len(X_eval)),
        "shap_threshold": float(shap_threshold),
        "top_k": float(k),
    }


# def compute_structural_explainability_metrics(
#     *,
#     model,
#     X_train,
#     X_test,
#     y_train,
#     clf_type_score: dict,
#     correlated_thresholds,
#     high_cor: float,
#     model_size_thresholds,
#     feature_relevance_thresholds,
#     threshold_outlier: float,
#     penalty_outlier: float,
# ) -> dict:

#     # ============================================================
#     # 1) Algorithm Class Score
#     # ============================================================

#     model_name = type(model).__name__
#     algorithm_score = float(clf_type_score.get(model_name, np.nan))

#     # ============================================================
#     # 2) Correlated Features Score
#     # ============================================================

#     X_comb = pd.concat([X_train, X_test])
#     corr_matrix = X_comb.corr().abs()

#     upper = corr_matrix.where(
#         np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
#     )

#     to_drop = [col for col in upper.columns if any(upper[col] > high_cor)]
#     pct_corr = float(len(to_drop) / max(len(X_comb.columns), 1))

#     corr_score = float(
#         5 - np.digitize(pct_corr, correlated_thresholds, right=True)
#     )

#     # ============================================================
#     # 3) Model Size Score
#     # ============================================================

#     n_features = int(X_train.shape[1])
#     model_size_score = float(
#         5 - np.digitize(n_features, model_size_thresholds, right=True)
#     )

#     # ============================================================
#     # 4) Feature Relevance Score
#     # ============================================================

#     importance = None

#     if hasattr(model, "coef_"):
#         model.fit(X_train, y_train)
#         importance = np.abs(model.coef_[0])

#     elif hasattr(model, "feature_importances_"):
#         importance = np.abs(model.feature_importances_)

#     if importance is None:
#         feature_relevance_score = np.nan
#         n_outliers = 0
#         pct_dominant = 0.0
#     else:
#         importance = np.asarray(importance)

#         q1, q3 = np.percentile(importance, [25, 75])
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper_t = q3 + 1.5 * iqr

#         n_outliers = int(
#             np.sum((importance < lower) | (importance > upper_t))
#         )

#         cumulative = np.cumsum(np.sort(importance)[::-1])
#         total = float(importance.sum())
#         pct_dominant = float(
#             np.sum(cumulative < 0.6 * total) / max(len(importance), 1)
#         )

#         base_score = float(
#             np.digitize(pct_dominant, feature_relevance_thresholds, right=False) + 1
#         )

#         if n_outliers / max(len(importance), 1) >= threshold_outlier:
#             base_score -= penalty_outlier

#         feature_relevance_score = float(max(base_score, 1))

#     # ============================================================
#     # Output
#     # ============================================================

#     return {
#         "algorithm_class": algorithm_score,
#         "correlated_features": corr_score,
#         "model_size": model_size_score,
#         "feature_relevance": feature_relevance_score,
#         "pct_correlated": pct_corr,
#         "n_features": n_features,
#         "model_type": model_name,
#         "n_outliers": n_outliers,
#         "pct_dominant": pct_dominant,
#     }


# ============================================================
# Structural Explainability Metrics
# ============================================================

def compute_algorithm_class(model):
    model_name = type(model).__name__

    # Ejemplo simple (puedes externalizarlo a config)
    mapping = {
        "LogisticRegression": 5,
        "LinearRegression": 5,
        "DecisionTreeClassifier": 3,
        "RandomForestClassifier": 2,
    }

    return {
        "value": mapping.get(model_name, np.nan),
        "model_type": model_name,
    }


def compute_correlated_features(X_train, X_test, high_cor=0.9):
    X_comb = pd.concat([X_train, X_test])
    corr_matrix = X_comb.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [c for c in upper.columns if any(upper[c] > high_cor)]
    pct_corr = len(to_drop) / max(len(X_comb.columns), 1)

    return {
        "value": float(pct_corr),
    }


def compute_model_size(X_train):
    n_features = X_train.shape[1]

    return {
        "value": int(n_features),
    }


def compute_feature_relevance(
    model,
    X_train,
    y_train,
    threshold_outlier=0.03,
):
    if hasattr(model, "coef_"):
        model.fit(X_train, y_train)
        importance = np.abs(model.coef_[0])

    elif hasattr(model, "feature_importances_"):
        importance = np.abs(model.feature_importances_)

    else:
        return {"value": np.nan}

    q1, q3 = np.percentile(importance, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    n_outliers = np.sum((importance < lower) | (importance > upper))

    cumulative = np.cumsum(np.sort(importance)[::-1])
    total = importance.sum()
    pct_dominant = np.sum(cumulative < 0.6 * total) / len(importance)

    return {
        "value": float(pct_dominant),
        "n_outliers": int(n_outliers),
        "importances": importance.tolist(),
    }
    # dist_score = np.digitize(pct_dist, thresholds, right=False) + 1 
    
    # if n_outliers/len(importance) >= threshold_outlier (0.03):
    #     dist_score -= penalty_outlier (0.5)

    # NECESITA FIT



def compute_performance_difference(f, g, X_test, y_test):

    y_f = f.predict(X_test)
    y_g = g.predict(X_test)

    acc_f = accuracy_score(y_test, y_f)
    acc_g = accuracy_score(y_test, y_g)

    D = acc_f - acc_g

    return {
        "value": float(D),
        "perf_original": float(acc_f),
        "perf_explainer": float(acc_g),
    }

def compute_number_of_rules(tree_model) -> dict:
    if hasattr(tree_model, "get_n_leaves"):
        n_rules = tree_model.get_n_leaves()
    else:
        n_rules = float('nan')
    return {"value": float(n_rules), "n_rules": float(n_rules)}


def compute_average_rule_length(tree_model) -> dict:
    if not hasattr(tree_model, "tree_"):
        return {"value": float('nan'), "avg_rule_length": float('nan')}
        
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

    return {"value": float(avg_depth), "avg_rule_length": float(avg_depth)}

def compute_rule_stats(rule_model) -> dict:
    if not hasattr(rule_model, "rules_"):
        return {"value": float('nan'), "n_rules": 0, "avg_rule_length": float('nan')}
        
    rules = rule_model.rules_
    n_rules = len(rules)
    lengths = [len(rule.conditions) for rule in rules] if n_rules > 0 else [0]

    return {
        "value": float(np.mean(lengths)),
        "n_rules": int(n_rules),
        "avg_rule_length": float(np.mean(lengths)),
    }

def compute_tree_depth(tree_model) -> dict:
    if hasattr(tree_model, "get_depth"):
        depth = tree_model.get_depth()
    else:
        depth = float('nan')
    return {"value": float(depth), "max_depth": float(depth)}

def compute_interaction_strength(model, X) -> dict:
    try:
        explainer = shap.TreeExplainer(model)
        shap_int = explainer.shap_interaction_values(X)
        shap_int = np.abs(shap_int)

        total = shap_int.sum()
        # shap_interaction_values devuelve tensores (n_samples, n_features, n_features)
        # o una lista de ellos si es multiclase. Asumimos el formato estándar:
        if isinstance(shap_int, list):
            shap_int = shap_int[1] if len(shap_int) > 1 else shap_int[0]
            
        main_effect = np.sum(np.diagonal(shap_int, axis1=1, axis2=2))
        interaction_strength = (total - main_effect) / total if total != 0 else 0.0
        
        return {"value": float(interaction_strength)}
    except Exception:
        return {"value": float('nan')}

###########################
## AIX 360 ¿FIT? XPLIQUE todo el dataset
###########################

def compute_faithfulness_metric(model, x: np.ndarray, coefs: np.ndarray, base: np.ndarray) -> dict:
    """ This metric evaluates the correlation between the importance assigned by the interpretability algorithm
    to attributes and the effect of each of the attributes on the performance of the predictive model.
    The higher the importance, the higher should be the effect, and vice versa, The metric evaluates this by
    incrementally removing each of the attributes deemed important by the interpretability metric, and
    evaluating the effect on the performance, and then calculating the correlation between the weights (importance)
    of the attributes and corresponding model performance. [#]_

    References:
        .. [#] `David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining
           neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors,
           Advances in Neural Information Processing Systems 31, pages 7775-7784. 2018.
           <https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf>`_

    Args:
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        x (numpy.ndarray): row of data.
        coefs (numpy.ndarray): coefficients (weights) corresponding to attribute importance.
        base ((numpy.ndarray): base (default) values of attributes

    Returns:
        float: correlation between attribute importance weights and corresponding effect on classifier.
    """

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

    corr = -np.corrcoef(coefs, pred_probs)[0, 1]

    return {
        "value": float(corr),
        "pred_class": int(pred_class),
        "coefs": coefs,
        "pred_probs": pred_probs,
    }


def compute_monotonicity_metric(model, x: np.ndarray, coefs: np.ndarray, base: np.ndarray) -> dict:
    """ This metric measures the effect of individual features on model performance by evaluating the effect on
    model performance of incrementally adding each attribute in order of increasing importance. As each feature
    is added, the performance of the model should correspondingly increase, thereby resulting in monotonically
    increasing model performance. [#]_

    References:
        .. [#] `Ronny Luss, Pin-Yu Chen, Amit Dhurandhar, Prasanna Sattigeri, Karthikeyan Shanmugam, and
           Chun-Chen Tu. Generating Contrastive Explanations with Monotonic Attribute Functions. CoRR abs/1905.13565. 2019.
           <https://arxiv.org/pdf/1905.12698.pdf>`_

    Args:
        model: Trained classifier, such as a ScikitClassifier that implements
            a predict() and a predict_proba() methods.
        x (numpy.ndarray): row of data.
        coefs (numpy.ndarray): coefficients (weights) corresponding to attribute importance.
        base ((numpy.ndarray): base (default) values of attributes

    Returns:
        bool: True if the relationship is monotonic.
    """
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

# =============================================================================
# Perturbation & Correlation Metrics (Xplique-style) 
# ============================================================================= 

def compute_shapley_corr(feature_weights: np.ndarray, ground_truth_weights: np.ndarray) -> Dict[str, float]:
    """
    Calcula la correlación de Pearson entre los pesos de las características explicadas 
    y unos pesos que se consideran el 'ground truth'.
    """
    feature_weights = np.asarray(feature_weights)
    ground_truth_weights = np.asarray(ground_truth_weights)
    
    corr = [np.corrcoef(a, b)[0, 1] for a, b in zip(feature_weights, ground_truth_weights)]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {"value": float(np.mean(corr))}

def compute_roar(model, X_train, y_train, X_test, y_test, train_feature_weights, test_feature_weights) -> Dict[str, float]: # DIFIERE
    """
    Implementation of ROAR, remove and retrain - https://arxiv.org/abs/1806.10758
    Remove the features deemed most important, then remove them from the data and retrain.
    Evaluate the performance degradation using AUC.
    A higher roar score is better.
    """
    
    X_train_np = np.asarray(X_train).copy()
    X_test_np = np.asarray(X_test).copy()
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)
    
    # Combinamos para obtener las medias globales de las características (Interventional conditional)
    X_all = np.vstack([X_train_np, X_test_np])
    avg_feature_values = X_all.mean(axis=0)
    num_features = X_all.shape[1]
    
    cutoffs = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    losses = []
    
    for cutoff_percent in cutoffs:
        cutoff = int(cutoff_percent * num_features)
        X_train_new = X_train_np.copy()
        X_test_new = X_test_np.copy()
        
        # Enmascaramos características en el conjunto de entrenamiento
        for i in range(len(X_train_new)):
            sorted_indices = np.argsort(np.abs(train_feature_weights[i]))[::-1]
            indices_to_remove = sorted_indices[:cutoff]
            X_train_new[i, indices_to_remove] = avg_feature_values[indices_to_remove]
            
        # Enmascaramos características en el conjunto de prueba
        for i in range(len(X_test_new)):
            sorted_indices = np.argsort(np.abs(test_feature_weights[i]))[::-1]
            indices_to_remove = sorted_indices[:cutoff]
            X_test_new[i, indices_to_remove] = avg_feature_values[indices_to_remove]
            
        # Clonamos el modelo original sin entrenar, lo ajustamos y predecimos
        model_new = clone(model)
        model_new.fit(X_train_new, y_train_np.ravel())
        preds = model_new.predict(X_test_new)
        
        # Pérdida usando MAE (acorde al script original evaluate_model)
        loss = np.mean(np.abs(y_test_np - preds))
        losses.append(loss)
        
    # Calculamos el AUC de la curva de pérdidas
    area = 0.0
    for i in range(1, len(cutoffs)):
        length = (losses[i] + losses[i - 1]) / 2
        width = cutoffs[i] - cutoffs[i - 1]
        area += length * width
        
    return {"value": float(area)}


def compute_infidelity(model, X_test, feature_weights) -> Dict[str, float]:
    """
    Computes the infidelity by evaluating whether perturbations to important features lead to proportional changes in model output.
    Implementation of https://arxiv.org/pdf/1901.09392.pdf, based on https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py
    """
    X_np = np.asarray(X_test)
    num_datapoints, num_features = X_np.shape
    infids = []
    
    def get_exp(ind, exp):
        return exp[ind.astype(int)]

    def set_zero_infid(array, size, point):
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1.0, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])

    for i in range(num_datapoints):
        num_reps = 1000
        x_orig = np.tile(X_np[i], [num_reps, 1])
        x = X_np[i]
        expl_copy = np.copy(feature_weights[i])
        
        val = np.apply_along_axis(set_zero_infid, 1, x_orig, num_features, num_features)
        x_ptb = val[:, :num_features]
        ind = val[:, num_features: 2*num_features]
        rand = val[:, 2*num_features: 3*num_features]
        
        exp_sum = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, expl_copy), axis=1)
        ks = np.ones(num_reps)
        
        pdt = model.predict([x])[0]
        pdt_ptb = model.predict(x_ptb)
        pdt_diff = pdt - pdt_ptb

        # Evitamos divisiones por cero en el cálculo de beta
        denominator = np.mean(ks * exp_sum * exp_sum)
        beta = np.mean(ks * pdt_diff * exp_sum) / (denominator if denominator != 0 else 1e-10)
        exp_sum *= beta
        
        infid = np.mean(ks * np.square(pdt_diff - exp_sum)) / np.mean(ks)
        infids.append(infid)
        
    return {"value": float(np.mean(infids))}

##########
## HOLISTICAI
#########

def _extract_metric_from_df(df: pd.DataFrame, metric_name: str) -> float:
    """Helper para extraer un valor específico de los DataFrames de holisticai."""
    if df is None or df.empty:
        return float('nan')
    
    if 'value' in df.columns:
        if 'metric' in df.columns:
            row = df[df['metric'].str.contains(metric_name, case=False, na=False)]
            if not row.empty:
                return float(row['value'].iloc[0])
        else:
            for idx in df.index:
                if metric_name.lower() in str(idx).lower():
                    return float(df.loc[idx, 'value'])
    return float('nan')

# =============================================================================
# Global Explainability Metrics
# =============================================================================

def compute_global_explainability_metrics(importances, partial_dependencies, conditional_importances) -> Dict[str, float]:
    df_metrics = classification_explainability_metrics(importances, partial_dependencies, conditional_importances)
    
    return {
        "alpha_score": _extract_metric_from_df(df_metrics, "Alpha Importance Score"),
        "xai_ease_score": _extract_metric_from_df(df_metrics, "XAI Ease Score"),
        "position_parity": _extract_metric_from_df(df_metrics, "Position Parity"),
        "rank_alignment": _extract_metric_from_df(df_metrics, "Rank Alignment"),
        "spread_ratio": _extract_metric_from_df(df_metrics, "Spread Ratio"),
        "spread_divergence": _extract_metric_from_df(df_metrics, "Spread Divergence"),
        "fluctuation_ratio": _extract_metric_from_df(df_metrics, "Fluctuation Ratio") # Si aplica globalmente
    }

# =============================================================================
# Local Explainability Metrics
# =============================================================================

def compute_local_explainability_metrics(local_importances) -> Dict[str, float]:
    df_metrics = classification_local_feature_importance_explainability_metrics(local_importances)
    
    return {
        "rank_consistency": _extract_metric_from_df(df_metrics, "Rank Consistency"),
        "importance_stability": _extract_metric_from_df(df_metrics, "Importance Stability")
    }

# =============================================================================
# Surrogate Accuracy/Fidelity Metrics
# =============================================================================

def compute_surrogate_explainability_metrics(X_test, y_test, y_pred, surrogate, is_regression: bool = False) -> Dict[str, float]:
    if is_regression:
        df_metrics = regression_surrogate_explainability_metrics(X_test, y_test, y_pred, surrogate)
    else:
        df_metrics = classification_surrogate_explainability_metrics(X_test, y_test, y_pred, surrogate)
        
    return {
        "mse_degradation": _extract_metric_from_df(df_metrics, "MSE Degradation"),
        "surrogate_fidelity": _extract_metric_from_df(df_metrics, "Surrogate Fidelity"),
        "surrogate_feature_stability": _extract_metric_from_df(df_metrics, "Surrogate Feature Stability"),
        "spread_divergence": _extract_metric_from_df(df_metrics, "Spread Divergence"), 
        "fluctuation_ratio": _extract_metric_from_df(df_metrics, "Fluctuation Ratio"), 
        "rank_alignment": _extract_metric_from_df(df_metrics, "Rank Alignment"), 
        "alpha_score": _extract_metric_from_df(df_metrics, "Alpha Score")
    }

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

def compute_weighted_average_depth(tree) -> Dict[str, float]:
    """
    Weighted Average Depth calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree: Tree
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
    if tree is None: return {"value": np.nan}
    depths, counts = get_depths_counts(0, tree, [], [])
    n_samples = sum(counts)
    if n_samples == 0: return {"value": 0.0}
    return {"value": float((np.array(depths) * (np.array(counts) / n_samples)).sum())}

def compute_weighted_average_explainability_score(tree) -> Dict[str, float]:
    """
    Weighted Average Explainability Score calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree: Tree
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
    if tree is None: return {"value": np.nan}
    depths, counts = get_cuts_counts(0, tree, [], [], set())
    n_samples = sum(counts)
    if n_samples == 0: return {"value": 0.0}
    return {"value": float((np.array(depths) * (np.array(counts) / n_samples)).sum())}

def compute_weighted_tree_gini(tree) -> Dict[str, float]: # ALGO DIFERENTE
    """
    Compute the weighted Gini index for the tree (WGNI).
    Reference value: 0.0

    Parameters
    ----------
    tree : Tree
        The tree to compute the weighted Gini index of.

    Returns
    -------
    dict
        A dictionary containing the weighted Gini index of the tree.
    """ 
    if tree is None: return {"value": np.nan}
    
    is_classification = tree.n_classes[0] > 1
    weighted_impurity = 0.0
    total_samples = tree.n_node_samples[0]

    def accumulate_impurity(node_index):
        nonlocal weighted_impurity
        if is_leaf(node_index, tree):
            node_samples = tree.n_node_samples[node_index]
            if node_samples > 0:
                node_value = tree.value[node_index, 0, :]
                if is_classification:
                    impurity = 1.0 - np.sum((node_value / node_samples)**2)
                else:
                    impurity = np.sum((node_value - np.mean(node_value)) ** 2) / node_samples
                weighted_impurity += (node_samples / total_samples) * impurity
        else:
            accumulate_impurity(tree.children_left[node_index])
            accumulate_impurity(tree.children_right[node_index])

    accumulate_impurity(0)
    return {"value": float(weighted_impurity)}

def compute_tree_depth_variance(tree) -> Dict[str, float]:
    """
    Compute the variance of the depths of the leaves in the tree (TDV).
    Reference value: 0.0

    Parameters
    ----------
    tree : Tree
        The tree to compute the depth variance of.

    Returns
    -------
    dict
        A dictionary containing the variance of the leaf depths.

    """
    if tree is None: return {"value": np.nan}
    depths, _ = get_depths_counts(0, tree, [], [])
    if not depths: return {"value": 0.0}
    return {"value": float(np.mean((depths - np.mean(depths)) ** 2))}

def compute_tree_number_of_rules(surrogate) -> Dict[str, float]:
    """
    Calculates the number of rules in a decision tree surrogate model.

    Parameters
    ----------
        surrogate: A surrogate model, typically a decision tree, for which the number of rules is to be calculated.

    Returns
    -------
        int: The number of rules present in the surrogate model.
    """
    if surrogate is None: return {"value": np.nan}
    return {"value": float(get_number_of_rules(surrogate))}

def compute_tree_number_of_features(surrogate) -> Dict[str, float]:
    """
    Calculates the number of features used in a decision tree surrogate model.

    Parameters
    ----------
        surrogate: A surrogate model, typically a decision tree, for which the number of features is to be calculated.

    Returns
    -------
        int: The number of features used in the surrogate model.
    """
    if surrogate is None: return {"value": np.nan}
    features = get_features(surrogate)
    return {"value": float(len(np.unique(features[features >= 0])))}


# =============================================================================
# Ensemble XAI Consistency Metrics (Custom)
# =============================================================================

import random
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# ----------------------------
# Funciones auxiliares
# ----------------------------

def to_scalar(val):
    """
    Convierte arrays de numpy o listas de 1 elemento a float.
    Si tiene varios elementos, devuelve la media.
    """
    if isinstance(val, (np.ndarray, list)):
        val = np.array(val).flatten()
        if len(val) == 1:
            return float(val[0])
        return float(np.mean(val))
    return float(val)

def get_top_k(importance_dict, k, return_values=False):
    """
    Devuelve las k características más importantes según la magnitud de su valor.
    Si return_values=True, devuelve una lista de tuplas (feature, importance).
    Si return_values=False, devuelve un set con los nombres (para Jaccard).
    """
    cleaned_dict = {feat: to_scalar(val) for feat, val in importance_dict.items()}
    sorted_feats = sorted(cleaned_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    if return_values:
        return sorted_feats[:k]
    return set([feat for feat, val in sorted_feats[:k]])

# ----------------------------
# Métodos XAI
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
        exp = explainer.explain_instance(X.values[i], predict_fn_wrapper)
        local_list = exp.local_exp[1] if mode=='classification' else exp.local_exp[0]
        for feat_idx, weight in local_list:
            importances[feature_names[feat_idx]] += abs(weight)

    importances = {k: v/num_samples for k, v in importances.items()}
    return importances

def compute_shap_custom(model, X, mode='classification'):
    feature_names = X.columns.tolist()
    background = X.iloc[:20, :]
    test_samples = X.iloc[:20, :]

    if hasattr(model, 'predict_proba') and mode == 'classification':
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(test_samples, silent=True)
        vals = np.abs(shap_values[1]).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
    else:
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(test_samples, silent=True)
        vals = np.abs(shap_values).mean(axis=0)

    importances = dict(zip(feature_names, vals))
    return importances

def compute_pdp_importance(model, X):
    importances = {}
    for col in X.columns:
        pdp_result = partial_dependence(model, X, [col], kind='average', grid_resolution=20)
        val = np.std(pdp_result['average'])
        importances[col] = val
    return importances

def compute_pfi(model, X, y, seed=42):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=seed)
    importances = dict(zip(X.columns, r.importances_mean))
    return importances

# def compute_lofo(model, X, y):
#     base_score = model.score(X, y)
#     importances = {}
#     for col in X.columns:
#         X_temp = X.copy()
#         X_temp[col] = np.random.permutation(X_temp[col].values)
#         score_drop = base_score - model.score(X_temp, y)
#         importances[col] = score_drop
#     return importances

def compute_surrogate(model, X, mode='classification'):
    targets = model.predict(X)
    if mode == 'classification':
        surrogate = DecisionTreeClassifier(max_depth=3)
    else:
        surrogate = DecisionTreeRegressor(max_depth=3)
    surrogate.fit(X, targets)
    importances = dict(zip(X.columns, surrogate.feature_importances_))
    return importances

# ----------------------------
# Métrica de consistencia XAI
# ----------------------------

def calculate_consistency_matrix(rankings, k=5):
    methods = list(rankings.keys())
    n = len(methods)
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for i in range(n):
        for j in range(n):
            # Aquí llamamos con return_values=False para calcular Jaccard
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

def compute_xai_consistency(model, X, y, k=5, mode='classification', seed=42) -> Dict[str, Any]:
    np.random.seed(seed)
    random.seed(seed)
    
    # Aseguramos que X es un DataFrame para que tengan nombres de columnas
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
    rankings = {}
    rankings['LIME'] = compute_lime(model, X, mode=mode, num_samples=20, seed=seed)
    rankings['SHAP'] = compute_shap_custom(model, X, mode=mode)
    rankings['PDP'] = compute_pdp_importance(model, X)
    rankings['PFI'] = compute_pfi(model, X, y, seed=seed)
    #rankings['LOFO'] = compute_lofo(model, X, y)
    rankings['Surrogate'] = compute_surrogate(model, X, mode=mode)

    matrix = calculate_consistency_matrix(rankings, k=k)
    score = get_aggregated_score(matrix)
    
    top_k_info = []
    for method, importances in rankings.items():
        # Extraemos las top K con sus valores
        top_k_list = get_top_k(importances, k, return_values=True)
        features_str = ", ".join([feat for feat, val in top_k_list])
        vals_str = ", ".join([f"{val:.3f}" for feat, val in top_k_list])
        
        info_string = f"Top-{k} features {method}: {features_str}, con importancias: {vals_str}"
        top_k_info.append(info_string)
    
    return {
        "value": float(score),
        "consistency_matrix": matrix.to_dict(),
        "top_k_details": "\n".join(top_k_info), # Aquí pasamos el texto formateado
        "rankings": rankings
    }