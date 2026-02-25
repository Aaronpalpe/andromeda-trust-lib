from __future__ import annotations

import contextlib
import logging
import os
import warnings

import numpy as np
import pandas as pd


def _safe_import_shap():
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Explainability requires the optional dependency 'shap'. "
            "Install with: pip install shap"
        ) from exc
    return shap


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