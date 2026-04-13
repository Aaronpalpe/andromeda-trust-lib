from __future__ import annotations

import contextlib
import logging
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from art.metrics import loss_sensitivity
from art.attacks.evasion import CarliniL2Method, FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DeepFool
from scipy.stats import kendalltau, spearmanr


def _safe_import_art_blackbox():
    """Import ART components for black-box HSJ."""
    try:
        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import BlackBoxClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Robustness requires 'adversarial-robustness-toolbox'. "
            "Install with: pip install adversarial-robustness-toolbox"
        ) from exc
    return HopSkipJump, BlackBoxClassifier


def _safe_import_art_metrics():
    """Import ART metrics lazily."""
    try:
        from art.metrics import RobustnessVerificationTreeModelsCliqueMethod, clever_u
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Robustness metrics require 'adversarial-robustness-toolbox'. "
            "Install with: pip install adversarial-robustness-toolbox"
        ) from exc
    return RobustnessVerificationTreeModelsCliqueMethod, clever_u


# def _safe_import_art_sklearn_wrappers():
#     """ART wrappers for scikit-learn models (needed for Clique Method)."""
#     try:
#         from art.estimators.classification.scikitlearn import ScikitlearnClassifier
#     except Exception as exc:  # pragma: no cover
#         raise ImportError(
#             "Clique Method requires ART scikit-learn wrappers. "
#             "Install with: pip install adversarial-robustness-toolbox"
#         ) from exc
#     return ScikitlearnClassifier


logging.getLogger("art").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_attack_noise():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


def _ensure_dataframe(X, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X_arr = np.asarray(X)
    if columns is None:
        columns = [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=columns)


def _to_numpy_float(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    X = np.asarray(X)
    return X.astype(np.float32, copy=False)


def _infer_classes(model, y_ref: np.ndarray) -> np.ndarray:
    if hasattr(model, "classes_"):
        try:
            cls = np.asarray(model.classes_)
            if cls.ndim == 1 and cls.size > 0:
                return cls
        except Exception:
            pass # raise ValueError("Model's 'classes_' attribute is not a valid 1D array of class labels.")
    return np.unique(np.asarray(y_ref))


def _one_hot_from_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    idx = {c: i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([idx.get(v, -1) for v in labels], dtype=int)
    if (y_idx < 0).any():
        missing = labels[y_idx < 0]
        raise ValueError(f"Predicted labels contain unseen classes: {np.unique(missing).tolist()}")
    oh = np.zeros((labels.shape[0], classes.shape[0]), dtype=np.float32)
    oh[np.arange(labels.shape[0]), y_idx] = 1.0
    return oh


def _make_predict_fn(model, *, columns: list[str], classes: np.ndarray):
    def predict_fn(x: np.ndarray) -> np.ndarray:
        X_df = _ensure_dataframe(x, columns=columns)
        y_pred = np.asarray(model.predict(X_df)).reshape(-1)
        return _one_hot_from_labels(y_pred, classes)
    return predict_fn


def _featurewise_clip_values(X_train: pd.DataFrame | np.ndarray):
    X_df = _ensure_dataframe(X_train)
    mins = np.asarray(X_df.min(axis=0), dtype=np.float32)
    maxs = np.asarray(X_df.max(axis=0), dtype=np.float32)
    same = mins == maxs
    if same.any():
        mins = mins.copy()
        maxs = maxs.copy()
        mins[same] -= 1e-6
        maxs[same] += 1e-6
    return mins, maxs


def _attack_metrics_core(
    *,
    model,
    X_test,
    y_test,
    attack,
    n_samples=50,
    seed=42,
    attack_name="attack",
):
    X_df = _ensure_dataframe(X_test)
    y = np.asarray(y_test).reshape(-1)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_df), size=min(n_samples, len(X_df)), replace=False)

    X_eval = X_df.iloc[idx]
    y_eval = y[idx]

    # Clean predictions
    y_pred_clean = np.asarray(model.predict(X_eval)).reshape(-1) #BEFORE model
    clean_acc = float((y_pred_clean == y_eval).mean())

    correct_mask = (y_pred_clean == y_eval)
    n_total = len(X_eval)
    n_correct = int(np.sum(correct_mask))

    if n_correct == 0:
        return {
            "clean_accuracy": clean_acc * 100.0,
            "adv_accuracy": clean_acc * 100.0,
            "accuracy_drop_pct (effective_robustness)": 0.0,
            "robustness_ratio (adv/clean)": 1.0,
            "adv_accuracy_correct_only": 0.0,
            "attack_success_rate_pct (correct_only)": 0.0,
            "sample_size (n_eval)": float(n_total),
            "n_attacked": float(n_correct),
            "attack": attack_name,
            "note": "No correctly classified samples.",
        }

    # Attack only correct samples 
    X_correct = _to_numpy_float(X_eval.iloc[correct_mask]) # BEFORE X_eval
    y_correct = y_eval[correct_mask]

    X_adv = attack.generate(X_correct)

    X_adv_df = _ensure_dataframe(X_adv, columns=X_eval.columns) 
    y_pred_adv_correct = np.asarray(model.predict(X_adv_df)).reshape(-1) # BEFORE model

    # Metrics
    adv_acc_correct = float((y_pred_adv_correct == y_correct).mean())
    asr = float((y_pred_adv_correct != y_correct).mean())

    # Reconstruct full predictions
    y_pred_adv_full = y_pred_clean.copy()
    y_pred_adv_full[correct_mask] = y_pred_adv_correct

    adv_acc_full = float((y_pred_adv_full == y_eval).mean())
    drop_pct = max(0.0, clean_acc - adv_acc_full) * 100.0

    result = {
        "clean_accuracy": clean_acc * 100.0,
        "adv_accuracy": adv_acc_full * 100.0,
        "accuracy_drop_pct (effective_robustness)": drop_pct,
        "robustness_ratio (adv/clean)": adv_acc_full / clean_acc if clean_acc > 0 else 0.0,
        "adv_accuracy_correct_only": adv_acc_correct * 100.0,        
        "attack_success_rate_pct (correct_only)": asr * 100.0,
        "sample_size (n_eval)": float(n_total),
        "n_attacked": float(n_correct),
        "attack": attack_name,
    }

    success_mask = (y_pred_adv_correct != y_correct)
    if np.sum(success_mask) > 0:
        l2 = np.linalg.norm((X_adv[success_mask] - X_correct[success_mask]).reshape(np.sum(success_mask), -1), ord=2, axis=1)
        linf = np.max(np.abs((X_adv[success_mask] - X_correct[success_mask]).reshape(np.sum(success_mask), -1)), axis=1)
        result.update({
            "er_l2_success": float(np.mean(l2)),
            "er_linf_success": float(np.mean(linf)),
        })
    else:
        result.update({
            "er_l2_success": 0.0,
            "er_linf_success": 0.0,
        })

    delta = X_adv - X_correct
    flat = delta.reshape(delta.shape[0], -1)

    l2 = np.linalg.norm(flat, ord=2, axis=1)
    linf = np.max(np.abs(flat), axis=1)

    result.update({
        "mean_l2": float(np.mean(l2)),
        "mean_linf": float(np.mean(linf)),
    })
    
    return result

def hopskipjump_metrics(
    *,
    model,
    X_test,
    y_test,
    X_train=None,
    n_samples: int = 30,
    seed: int = 42,
    max_iter: int = 10,
    max_eval: int = 1000,
    init_eval: int = 10,
    init_size: int = 10,
    norm: int | float | str = 2,
    beta: float = 1.0,
) -> dict:
    '''
    For a given model this function calculates the HopSkipJump attack metrics.
    First from the test data selects a random test subset.
    Then measures the accuracy of the model on this subset.
    Next creates HopSkipJump attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns a dictionary with the metrics.
    Parameters
    ----------
    model: object
        ML-model (compatible with ART).
    X_test: pd.DataFrame or np.ndarray
        Test features.
    y_test: pd.Series or np.ndarray
        Test labels.
    X_train: pd.DataFrame or np.ndarray, optional
        Optional train features (for feature-wise clipping).
    n_samples: int
        Number of test samples to use for evaluation (random subset).
    seed: int
        Random seed for reproducibility.
    max_iter: int
        Maximum number of iterations for HSJ attack.
    max_eval: int
        Maximum number of model evaluations for HSJ attack.
    init_eval: int
        Number of evaluations for initial HSJ attack.
    init_size: int
        Number of samples for initial HSJ attack.
    norm: int | float | str
        Norm to use for HSJ attack (e.g., 1, 2, np.inf).
    beta: float
        Scaling factor for accuracy drop calculation (default 1.0, set <1.0 to reduce drop).

    Returns
    -------
    Dictionary with HSJ attack metrics: 
        clean accuracy: accuracy on the clean test subset, 
        adversarial accuracy: accuracy on the adversarial test subset, 
        accuracy drop percentage: percentage decrease in accuracy due to adversarial attacks,
        attack success rate percentage: percentage of successful attacks,
        adversarial accuracy on correct-only subset: accuracy on correctly classified samples under adversarial attacks,
        error rate L2 on successful attacks: average L2 perturbation magnitude on successful attacks,
        error rate Linf on successful attacks: average Linf perturbation magnitude on successful attacks,
        mean L2 perturbation magnitudes on successful attacks: average perturbation sizes for successful attacks,
        mean Linf perturbation magnitudes on successful attacks: average perturbation sizes for successful attacks,
        number of samples evaluated: total number of samples evaluated,
        number of samples attacked: total number of samples subjected to adversarial attacks,
        attack name: name of the attack method,
        sample size: size of the evaluation subset,
        note: additional notes or observations,
        parameters used: parameters used for the HSJ attack.
    '''
    # Measure the execution time of this function to evaluate its efficiency
    HopSkipJump, BlackBoxClassifier = _safe_import_art_blackbox()

    X_test_df = _ensure_dataframe(X_test)
    y_test_arr = np.asarray(y_test).reshape(-1)

    # Subsample
    if n_samples and len(X_test_df) > int(n_samples):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(X_test_df), size=int(n_samples), replace=False)
        X_eval_df = X_test_df.iloc[idx].copy()
        y_eval = y_test_arr[idx]
    else:
        X_eval_df = X_test_df.copy()
        y_eval = y_test_arr

    # Clean preds on eval subset
    y_pred_clean_full = np.asarray(model.predict(X_eval_df)).reshape(-1)
    clean_acc_full = float((y_pred_clean_full == y_eval).mean())

    correct_mask = (y_pred_clean_full == y_eval)
    n_total = int(len(X_eval_df))
    n_correct = int(np.sum(correct_mask))

    if n_correct == 0:
        return {
            "clean_accuracy": clean_acc_full,
            "adv_accuracy": clean_acc_full,
            "accuracy_drop_pct (effective_robustness)": 0.0,
            "robustness_ratio (adv/clean)": 1.0,
            "adv_accuracy_correct_only": 0.0,
            "attack_success_rate_pct (correct_only)": 0.0,
            "er_l2_success": 0.0,
            "er_linf_success": 0.0,
            "mean_l2": 0.0,
            "mean_linf": 0.0,
            "sample_size (n_eval)": float(n_total),
            "n_attacked": 0.0,
            "attack": "HopSkipJump",
            "note": "No correctly classified samples in evaluation subset.",
            "params": {
                "max_iter": int(max_iter),
                "max_eval": int(max_eval),
                "init_eval": int(init_eval),
                "init_size": int(init_size),
                "norm": norm,
                "seed": int(seed),
                "beta": float(beta),
            },
        }

    # Correct-only subset (attack protocol)
    X_correct_df = X_eval_df.loc[correct_mask].copy()
    y_correct = y_eval[correct_mask]
    y_pred_clean_correct = y_pred_clean_full[correct_mask]

    classes = _infer_classes(model, np.concatenate([y_eval, y_pred_clean_full]))
    nb_classes = int(classes.shape[0])
    columns = list(X_test_df.columns)
    predict_fn = _make_predict_fn(model, columns=columns, classes=classes)

    if X_train is None:
        X_train = X_test_df
    clip_min, clip_max = _featurewise_clip_values(X_train)

    bb = BlackBoxClassifier(
        predict_fn=predict_fn,
        input_shape=(X_test_df.shape[1],),
        nb_classes=nb_classes,
        clip_values=(clip_min, clip_max),
    )

    attack = HopSkipJump(
        classifier=bb,
        targeted=False,
        norm=norm,
        max_iter=int(max_iter),
        max_eval=int(max_eval),
        init_eval=int(init_eval),
        init_size=int(init_size),
        verbose=False,
    )

    X_correct = _to_numpy_float(X_correct_df)

    with suppress_attack_noise():
        X_adv = attack.generate(x=X_correct)

    X_adv_df = _ensure_dataframe(X_adv, columns=columns)
    y_pred_adv_correct = np.asarray(model.predict(X_adv_df)).reshape(-1)

    # Correct-only metrics
    adv_acc_correct_only = float((y_pred_adv_correct == y_correct).mean())
    asr = float((y_pred_adv_correct != y_pred_clean_correct).mean())
    asr_pct = asr * 100.0

    # Full-subset adversarial accuracy (replace attacked preds only)
    y_pred_adv_full = y_pred_clean_full.copy()
    y_pred_adv_full[correct_mask] = y_pred_adv_correct
    adv_acc_full = float((y_pred_adv_full == y_eval).mean())
    accuracy_drop_pct = float(max(0.0, clean_acc_full - adv_acc_full*beta) * 100.0)

    # Perturbation magnitudes
    delta = np.asarray(X_adv, dtype=np.float32) - np.asarray(X_correct, dtype=np.float32)
    flat = delta.reshape(delta.shape[0], -1)
    l2 = np.linalg.norm(flat, ord=2, axis=1)
    linf = np.max(np.abs(flat), axis=1)

    mean_l2 = float(np.mean(l2))
    mean_linf = float(np.mean(linf))

    success_mask = (y_pred_adv_correct != y_correct)
    if np.any(success_mask):
        er_l2_success = float(np.mean(l2[success_mask]))
        er_linf_success = float(np.mean(linf[success_mask]))
    else:
        er_l2_success = 0.0
        er_linf_success = 0.0
    return {
        "clean_accuracy": clean_acc_full,
        "adv_accuracy": adv_acc_full,
        "accuracy_drop_pct (effective_robustness)": accuracy_drop_pct,
        "robustness_ratio (adv/clean)": adv_acc_full / clean_acc_full if clean_acc_full > 0 else 0.0,
        "adv_accuracy_correct_only": adv_acc_correct_only * 100.0, # predictions that remain correct after attack        
        "attack_success_rate_pct (correct_only)": asr_pct, # predictions that changed after attack 
        "er_l2_success": er_l2_success, # average L2 perturbation on successful attacks
        "er_linf_success": er_linf_success, # average Linf perturbation on successful attacks
        "mean_l2": mean_l2, # average L2 perturbation on all attacked samples
        "mean_linf": mean_linf, # average Linf perturbation on all attacked samples
        "sample_size (n_eval)": float(n_total), # total samples evaluated
        "n_attacked": float(n_correct), # total samples attacked (correctly classified subset)
        "attack": "HopSkipJump",
        "note": "HSJ attack metrics on correctly classified subset. Full subset adv accuracy also reported.",
        "params": {
            "max_iter": int(max_iter),
            "max_eval": int(max_eval),
            "init_eval": int(init_eval),
            "init_size": int(init_size),
            "norm": norm,
            "seed": int(seed),
        },
    }

# Only for NN, LG, SVM models (GRADIENT-BASED - REQUIRES NN)
def fgm_attack_metrics(*, art_clf, X_test, y_test, eps=0.2, n_samples=50, seed=42):
    """For a given model this function calculates the fast gradient attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates FSG attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns the difference in accuracy as the score. Higher drop in accuracy means lower robustness.
    NOTE: FastGradientMethod requires gradient computation.
    For sklearn models, use HopSkipJump (black-box) attack instead.
    
    Parameters
    ----------
    art_clf: object
        ART classifier with gradient support (PyTorchClassifier/TensorFlowV2Classifier)
    X_test: pd.DataFrame or np.ndarray
        Test features
    y_test: pd.Series or np.ndarray
        Test labels
    eps: float
        Perturbation magnitude
    n_samples: int
        Number of samples to evaluate
    seed: int
        Random seed

    Returns
    -------
    FSG attack metrics (clean accuracy, adversarial accuracy, accuracy drop percentage, metric name)
    """
    # Check whether this is a ScikitlearnClassifier (it does not support gradients).
    # ScikitlearnClassifier = _safe_import_art_sklearn_wrappers()
    # art_clf = ScikitlearnClassifier(model=model)

    # Validate gradient support before running the attack.
    if not hasattr(art_clf, 'loss_gradient') or not callable(getattr(art_clf, 'loss_gradient', None)):
        raise ValueError(
            "FastGradientMethod requires an estimator derived from BaseEstimator and LossGradientsMixin.\n"
            "sklearn models do NOT support gradient-based attacks.\n"
            "Options:\n"
            "  1. Use 'er_hopskipjump_attack' (black-box) which works with any model\n"
            "  2. For gradient attacks, use PyTorchClassifier or TensorFlowV2Classifier"
        )

    attack = FastGradientMethod(estimator=art_clf, eps=eps)

    return _attack_metrics_core(
        model=art_clf,
        X_test=X_test,
        y_test=y_test,
        attack=attack,
        n_samples=n_samples,
        seed=seed,
        attack_name="FGM",
    )

# Only for NN models (GRADIENT-BASED). # Only for NN, LG, SVM models
def carlini_wagner_metrics(*, art_clf, X_test, y_test, n_samples=10, seed=42):
    """For a given model this function calculates the CW attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates CW attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns the difference in accuracy as the score. Higher drop in accuracy means lower robustness.
    NOTE: CarliniL2Method requires gradient computation.
    For sklearn models, use HopSkipJump (black-box) attack instead.
    Parameters
    ----------
    art_clf: object
        ART classifier with gradient support (PyTorchClassifier/TensorFlowV2Classifier)
    X_test: pd.DataFrame or np.ndarray
        Test features
    y_test: pd.Series or np.ndarray
        Test labels
    n_samples: int
        Number of samples to evaluate
    seed: int
        Random seed

    Returns:
        CW attack metrics (clean accuracy, adversarial accuracy, accuracy drop percentage, metric name)
    """
    # Validate gradient support.
    # ScikitlearnClassifier = _safe_import_art_sklearn_wrappers()
    # art_clf = ScikitlearnClassifier(model=model)

    if not hasattr(art_clf, 'class_gradient') or not callable(getattr(art_clf, 'class_gradient', None)):
        raise ValueError(
            "CarliniL2Method requires an estimator derived from BaseEstimator and ClassGradientsMixin.\n"
            "sklearn models do NOT support gradient-based attacks.\n"
            "Options:\n"
            "  1. Use 'er_hopskipjump_attack' (black-box) which works with any model\n"
            "  2. For gradient attacks, use PyTorchClassifier or TensorFlowV2Classifier"
        )

    attack = CarliniL2Method(art_clf)

    return _attack_metrics_core(
        model=art_clf,
        X_test=X_test,
        y_test=y_test,
        attack=attack,
        n_samples=n_samples,
        seed=seed,
        attack_name="CarliniWagner",
    )

# Only for NN models (GRADIENT-BASED). Only for NN, LG, SVM models
def deepfool_metrics(*, art_clf, X_test, y_test, n_samples=10, seed=42):
    """For a given model this function calculates the deepfool attack score.
    First from the test data selects a random small test subset.
    Then measures the accuracy of the model on this subset.
    Next creates deepfool attacks on this test set and measures the model's
    accuracy on the attacks. Compares the before attack and after attack accuracies.
    Returns  the difference in accuracy as the score. Higher drop in accuracy means lower robustness.
    NOTE: DeepFool requires gradient computation.
    For sklearn models, use HopSkipJump (black-box) attack instead.

    Parameters
    ----------
    art_clf: object
        ART classifier with gradient support (PyTorchClassifier/TensorFlowV2Classifier)
    X_test: pd.DataFrame or np.ndarray
        Test features
    y_test: pd.Series or np.ndarray
        Test labels
    n_samples: int
        Number of samples to evaluate
    seed: int
        Random seed

    Returns
    -------
    Deepfool attack metrics (clean accuracy, adversarial accuracy, accuracy drop percentage, metric name)
    """
    # # Validate gradient support.
    # ScikitlearnClassifier = _safe_import_art_sklearn_wrappers()
    # art_clf = ScikitlearnClassifier(model=model)

    if not hasattr(art_clf, 'class_gradient') or not callable(getattr(art_clf, 'class_gradient', None)):
        raise ValueError(
            "DeepFool requires an estimator derived from BaseEstimator and ClassGradientsMixin.\n"
            "sklearn models do NOT support gradient-based attacks.\n"
            "Options:\n"
            "  1. Use 'er_hopskipjump_attack' (black-box) which works with any model\n"
            "  2. For gradient attacks, use PyTorchClassifier or TensorFlowV2Classifier"
        )

    attack = DeepFool(art_clf)

    return _attack_metrics_core(
        model=art_clf,
        X_test=X_test,
        y_test=y_test,
        attack=attack,
        n_samples=n_samples,
        seed=seed,
        attack_name="DeepFool",
    )

# Only for DT, RF, GBDT models
def clique_method_metrics(
    *,
    model,
    X_test,
    y_test,
    X_train=None,
    n_samples: int = 200,
    seed: int = 42,
    eps_init: float = 0.5,
    norm: float = 1,
    nb_search_steps: int = 5,
    max_clique: int = 2,
    max_level: int = 2,
) -> dict:
    """For a given ensemble tree-based model this function calculates the Clique score.
    It uses RobustnessVerificationTreeModelsCliqueMethod function from
    IBM art library to calculate the score. Date must be normalized.

    Parameters
    ----------
    model: object
        ML-model (Tree-based).
    X_test: pd.DataFrame or np.ndarray
        Test features.
    y_test: pd.Series or np.ndarray
        Test labels.
    X_train: pd.DataFrame or np.ndarray
        Train features (for feature-wise clipping).
    n_samples: int
        Number of test samples to use for evaluation (random subset).
    seed: int
        Random seed for reproducibility.
    eps_init: float
        Initial perturbation size for the attack.
    norm: float
        Norm to use for measuring perturbations (e.g., 1, 2, np.inf).
    nb_search_steps: int
        Number of search steps for the attack.
    max_clique: int
        Maximum clique size to consider.
    max_level: int
        Maximum tree level to consider.

    Returns
    -------
    Dictionary with Clique Method metrics:
        robustness_bound: estimated robustness bound. No alteration below this distance can change the prediction. Higher is better,
        verification_error: error rate of the robustness verification of the algorithm (lower is better),
        sample_size: number of samples evaluated,
        metric: name of the metric,
        params: parameters used for the Clique Method attack.
    """
    X_df = _ensure_dataframe(X_test)
    X_np = _to_numpy_float(X_df)
    if np.min(X_np) < 0.0 or np.max(X_np) > 1.0:
        raise ValueError("Clique Method requires input features to be normalized to [0, 1].")
    RobustnessVerificationTreeModelsCliqueMethod, _ = _safe_import_art_metrics()
    # ScikitlearnClassifier = _safe_import_art_sklearn_wrappers()

    X_df = _ensure_dataframe(X_test)
    y = np.asarray(y_test).reshape(-1)

    if n_samples and len(X_df) > int(n_samples):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(X_df), size=int(n_samples), replace=False)
        X_eval = X_df.iloc[idx].copy()
        y_eval = y[idx]
    else:
        X_eval = X_df.copy()
        y_eval = y

    if X_train is None:
        X_train = X_df
    clip_min, clip_max = _featurewise_clip_values(X_train)

    art_clf = SklearnClassifier(model=model, clip_values=(clip_min, clip_max))
    verifier = RobustnessVerificationTreeModelsCliqueMethod(classifier=art_clf, verbose=False)

    X_np = _to_numpy_float(X_eval)

    with suppress_attack_noise():
        bound, err = verifier.verify(
            x=X_np,
            y=y_eval,
            eps_init=float(eps_init),
            norm=float(norm),
            nb_search_steps=int(nb_search_steps),
            max_clique=int(max_clique),
            max_level=int(max_level),
        )

    return {
        "robustness_bound": float(bound),
        "verification_error": float(err),
        "sample_size": float(len(X_eval)),
        "metric": "CliqueMethod",
        "params": {
            "eps_init": float(eps_init),
            "norm": float(norm),
            "nb_search_steps": int(nb_search_steps),
            "max_clique": int(max_clique),
            "max_level": int(max_level),
            "seed": int(seed),
            "n_samples": int(n_samples),
        },
    }

# Only for NN models
def clever_score_metrics(
    *,
    classifier,
    x,
    n_samples: int = 10,
    seed: int = 42,
    nb_batches: int = 1,
    batch_size: int = 1,
    radius: float = 500,
    norm: int = 1,
) -> dict:
    """For a given Keras-NN model this function calculates the Untargeted-Clever score.
    It uses clever_u function from IBM art library.
    Returns a score according to the thresholds.
    
    Parameters
    ----------
    classifier: object
        ML-model (Keras).
    x: pd.DataFrame or np.ndarray
        Test features.
    n_samples: int
        Number of test samples to use for evaluation (random subset).
    seed: int
        Random seed for reproducibility.
    nb_batches: int
        Number of batches to use for CLEVER estimation.
    batch_size: int
        Batch size to use for CLEVER estimation.
    radius: float
        Radius to use for CLEVER estimation.
    norm: int
        Norm to use for CLEVER estimation (e.g., 1, 2, np.inf).

    Returns
    -------
    Dictionary with CLEVER metrics:
        clever_score_mean: Mean CLEVER score. It is an estimate of the minimum perturbation required to change the model's prediction. Higher is better,
        clever_score_std: Standard deviation of CLEVER scores,
        n_eval: Number of evaluations,
        metric: name of the metric,
        params: parameters used for the CLEVER estimation.
    """
    _, clever_u = _safe_import_art_metrics()

    if not (hasattr(classifier, "class_gradient") and hasattr(classifier, "loss_gradient")):
        raise RuntimeError(
            "CLEVER requires an ART classifier with class_gradient and loss_gradient "
            "(e.g., PyTorchClassifier / TensorFlowV2Classifier)."
        )

    X_df = _ensure_dataframe(x)
    X_np = _to_numpy_float(X_df)

    if n_samples and X_np.shape[0] > int(n_samples):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(X_np.shape[0], size=int(n_samples), replace=False)
        X_np = X_np[idx]

    scores: list[float] = []
    with suppress_attack_noise():
        for i in range(X_np.shape[0]):
            s = clever_u(
                classifier=classifier,
                x=X_np[i],
                nb_batches=int(nb_batches),
                batch_size=int(batch_size),
                radius=float(radius),
                norm=int(norm),
                verbose=False,
            )
            scores.append(float(s))
    # ANTES SE DEVOLVIA EL MINIMO
    return {
        "clever_score_mean": float(np.mean(scores)) if scores else 0.0,
        "clever_score_std": float(np.std(scores)) if scores else 0.0,
        "n_eval": float(len(scores)),
        "metric": "CLEVER",
        "params": {
            "n_samples": int(n_samples),
            "seed": int(seed),
            "nb_batches": int(nb_batches),
            "batch_size": int(batch_size),
            "radius": float(radius),
            "norm": int(norm),
        },
    }



# def confidence_score_metrics(*, model, X_test, y_test):
#     """For a given model this function calculates the Confidence score.
#     It takes the average over confusion_matrix. Then returns a score according to the thresholds.

#     Parameters
#     ----------
#     model: object
#         ML-model.
#     X_test: pd.DataFrame
#         Training data.
#     y_test: pd.DataFrame
#         Test labels.

#     Returns
#     -------
#     Confidence score
#     """
#     X_df = _ensure_dataframe(X_test)
#     y = np.asarray(y_test).reshape(-1)

#     y_pred = np.asarray(model.predict(X_df)).reshape(-1)

#     cm = confusion_matrix(y, y_pred)
#     cm_norm = cm / cm.sum(axis=1, keepdims=True)

#     confidence = float(np.mean(np.diag(cm_norm)) * 100.0)

#     return {
#         "confidence_score": confidence,
#         "metric": "ConfidenceScore",
#     }

def confidence_score_metrics(*, model, X_test, y_test, thresholds=None):
    """
    Compute a confidence score based on the Jaccard index (Intersection over Union) across multiple confidence thresholds.
    For each threshold T, we compute a binary prediction (positive if predicted probability >= T)
    and calculate the Jaccard index: TP / (TP + FP + FN).
    The final confidence score is the average Jaccard index across all specified thresholds.
    
    Parameters
    ----------
    model: object
        ML-model (must implement predict_proba()).
    X_test: pd.DataFrame or np.ndarray
        Test features.
    y_test: pd.Series or np.ndarray
        Test labels.
    thresholds: list of float, optional
        List of confidence thresholds to evaluate.  
    
    Returns
    -------
    Dictionary with confidence score and parameters used.

    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
    y = np.asarray(y_test).reshape(-1)
    
    if not hasattr(model, "predict_proba"):
        raise ValueError("The model must implement predict_proba() method to compute confidence scores.")
        
    probas = model.predict_proba(X_test)

    if probas is None:
        raise ValueError("Model's predict_proba() returned None. Cannot compute confidence scores.")
    
    #We assume binary classification and take the probability of the positive class. For multi-class, this can be adapted to compute per-class confidence scores.
    pos_probas = probas[:, 1] if probas.shape[1] > 1 else probas
    
    jaccard_scores = []
    
    for t in thresholds:
        y_pred_t = (pos_probas >= t).astype(int)
        
        cm = confusion_matrix(y, y_pred_t, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        
        denominator = TP + FP + FN
        if denominator == 0:
            score_t = 0.0
        else:
            score_t = TP / denominator
            
        jaccard_scores.append(score_t)
        
    confidence = float(np.mean(jaccard_scores) * 100.0)

    return {
        "confidence_score": confidence,
        "metric": "ConfidenceScore",
        "thresholds": thresholds
    }

# Only for NN models
def loss_sensitivity_metrics(*, classifier, X_test):
    """For a given Keras-NN model this function calculates the Loss Sensitivity score.
    It uses loss_sensitivity function from IBM art library. 
    Local loss sensitivity estimated through the gradients of the prediction at points in x.
    Returns a score according to the thresholds.

    Parameters
    ----------
    classifier: object
        ML-model (Keras).
    X_test: pd.DataFrame or np.ndarray
        Test features.

    Returns
    -------
    Loss Sensitivity score
    """
    try:
        X_df = _ensure_dataframe(X_test)
        X_np = _to_numpy_float(X_df)

        y_pred = classifier.predict(X_np)

        value = float(loss_sensitivity(classifier, X_np, y_pred))

        return {
            "loss_sensitivity": value,
            "metric": "LossSensitivity",
        }
    except Exception as e:
        raise RuntimeError(f"Loss Sensitivity metric requires an ART classifier with predict method (e.g., PyTorchClassifier / TensorFlowV2Classifier). Error: {str(e)}")


# def _check_gradient_support(model, attack_name: str) -> None:
#     """
#     Check whether the model supports gradients for white-box attacks.

#     Ataques white-box (requieren gradientes):
#         - FastGradientMethod, CarliniL2Method, DeepFool, CLEVER, Loss Sensitivity

#     Para sklearn: Solo ataques black-box (HopSkipJump) funcionan.
#     Para PyTorch/TensorFlow: Usar PyTorchClassifier o TensorFlowV2Classifier de ART.
#     """
#     # Check whether it is an ART estimator with gradient support.
#     has_loss_gradient = hasattr(model, 'loss_gradient')
#     has_class_gradient = hasattr(model, 'class_gradient')

#     if not (has_loss_gradient or has_class_gradient):
#         raise ValueError(
#             f"{attack_name} requires gradient computation which is NOT supported by sklearn models.\n"
#             f"Options:\n"
#             f"  1. Use black-box attacks (HopSkipJump) which work with any model\n"
#             f"  2. For gradient-based attacks, wrap your model with:\n"
#             f"     - PyTorchClassifier (for PyTorch models)\n"
#             f"     - TensorFlowV2Classifier (for TensorFlow models)\n"
#             f"  3. Skip this metric for sklearn models"
#         )


def ece_metrics(
    *,
    model,
    X_test,
    y_test,
    n_bins: int = 10,
) -> dict:
    """
    Expected Calibration Error (ECE)
    ECE measures the calibration of a model: how well the predicted probabilities reflect the true likelihood of correct predictions (it does not measure classification accuracy itself).

    - Predictions are grouped into intervals (bins) based on their confidence scores.
    - For each bin, we compute:
    - the average confidence (conf_b)
    - the average accuracy (acc_b)
    - ECE is the weighted average of the absolute difference between these two quantities:

    [
    ECE = \sum_{b=1}^{B} \frac{|B_b|}{n} \cdot |acc_b - conf_b|
    ]

    where |B_b| is the number of samples in bin b, and n is the total number of samples.

    Note: In binary classification, when using the maximum predicted probability as confidence, values lie in ([0.5, 1]), so bins in the interval ((0, 0.5]) remain empty.
    
    Parameters
    ----------
    model: object
    X_test: pd.DataFrame or np.ndarray
        Test features.  
    y_test: pd.Series or np.ndarray
        Test labels.
    n_bins: int
        Number of bins to use for ECE calculation.

    Returns
    -------
    Dictionary with ECE metrics:
        ece: Expected Calibration Error (lower is better),
        n_bins: number of bins used for calculation,
        bins: list of dictionaries with bin statistics (bin index, lower bound, upper bound, count, accuracy, confidence),
        metric: name of the metric.
    """

    X_df = _ensure_dataframe(X_test)
    y_true = np.asarray(y_test).reshape(-1)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("ECE requires model.predict_proba().")

    probs = np.asarray(model.predict_proba(X_df))

    if probs is None:
        raise RuntimeError("Model's predict_proba() returned None. Cannot compute ECE.")
    
    confidences = np.max(probs, axis=1)
    # predictions = np.argmax(probs, axis=1)
    indices = np.argmax(probs, axis=1)
    predictions = model.classes_[indices] # Mapeo a etiquetas reales

    correct = (predictions == y_true).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    bin_stats = []

    for b in range(n_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]

        mask = (confidences > lower) & (confidences <= upper)

        if not np.any(mask):
            continue

        bin_size = np.sum(mask)
        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(confidences[mask])

        ece += (bin_size / n) * abs(acc_bin - conf_bin)

        bin_stats.append({
            "bin": int(b),
            "lower": float(lower),
            "upper": float(upper),
            "count": int(bin_size),
            "accuracy": float(acc_bin),
            "confidence": float(conf_bin),
        })
        
    if np.isnan(ece):
        raise ValueError("ECE computation resulted in NaN. Check if model.predict_proba() returns valid probabilities and if n_bins is appropriate.")
    return {
        "ece": float(ece),
        "n_bins": int(n_bins),
        "bins": bin_stats,
        "metric": "ECE",
    }

# LO QUE SE HAN DESPLAZADO LOS DATOS DE TRAIN, NOSOTROS SOLO UN MOMENTO
# def psi_metrics(
#     *,
#     train_values,
#     current_values,
#     n_bins: int = 10,
#     eps: float = 1e-8,
# ) -> dict:
#     """
#     Population Stability Index between two 1D distributions.
#     """

#     train = np.asarray(train_values).reshape(-1)
#     current = np.asarray(current_values).reshape(-1)

#     # Bins defined on train split (standard practice)
#     breakpoints = np.linspace(
#         np.min(train),
#         np.max(train),
#         n_bins + 1
#     )

#     p_counts, _ = np.histogram(train, bins=breakpoints)
#     q_counts, _ = np.histogram(current, bins=breakpoints)

#     p = p_counts / max(len(train), 1)
#     q = q_counts / max(len(current), 1)

#     # evitar log(0)
#     p = np.clip(p, eps, None)
#     q = np.clip(q, eps, None)

#     psi = float(np.sum((p - q) * np.log(p / q)))

#     return {
#         "psi": psi,
#         "n_bins": int(n_bins),
#         "metric": "PSI",
#     }

# NECESITO TORCH PARA PROBARLO Y NO SON ROBUSTEZ
# def neuron_coverage(
#     *,
#     model,
#     X_test,
#     threshold: float = 0.0,
# ) -> dict:

#     import torch

#     model.eval()
#     covered = set()
#     total_neurons = 0

#     activations = []

#     def hook_fn(module, inp, out):
#         if isinstance(out, torch.Tensor):
#             activations.append(out.detach())

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, torch.nn.ReLU):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     with torch.no_grad():
#         _ = model(torch.tensor(X_test).float())

#     for act in activations:
#         total_neurons += act.shape[1]
#         mask = (act >= threshold).any(dim=0)
#         covered.update(torch.where(mask)[0].tolist())

#     for h in hooks:
#         h.remove()

#     nc = len(covered) / max(total_neurons, 1)

#     return {
#         "neuron_coverage": float(nc),
#         "total_neurons": int(total_neurons),
#         "covered_neurons": len(covered),
#         "metric": "NeuronCoverage",
#     }
# def tknc_bknc(
#     *,
#     model,
#     X_test,
#     k: int = 3,
# ) -> dict:

#     import torch

#     model.eval()
#     top_neurons = set()
#     bottom_neurons = set()
#     total_neurons = 0

#     activations = []

#     def hook_fn(module, inp, out):
#         if isinstance(out, torch.Tensor):
#             activations.append(out.detach())

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, torch.nn.ReLU):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     with torch.no_grad():
#         _ = model(torch.tensor(X_test).float())

#     for act in activations:
#         n = act.shape[1]
#         total_neurons += n

#         mean_act = act.mean(dim=0)

#         topk = torch.topk(mean_act, min(k, n)).indices
#         bottomk = torch.topk(-mean_act, min(k, n)).indices

#         top_neurons.update(topk.tolist())
#         bottom_neurons.update(bottomk.tolist())

#     for h in hooks:
#         h.remove()

#     return {
#         "tknc": len(top_neurons) / max(total_neurons, 1),
#         "bknc": len(bottom_neurons) / max(total_neurons, 1),
#         "metric": "TKNC_BKNC",
#     }

