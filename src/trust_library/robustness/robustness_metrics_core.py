from __future__ import annotations

import contextlib
import logging
import os
import warnings

import numpy as np
import pandas as pd


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


def _safe_import_art_sklearn_wrappers():
    """ART wrappers for scikit-learn models (needed for Clique Method)."""
    try:
        from art.estimators.classification.scikitlearn import ScikitlearnClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Clique Method requires ART scikit-learn wrappers. "
            "Install with: pip install adversarial-robustness-toolbox"
        ) from exc
    return ScikitlearnClassifier


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
            pass
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


def compute_hopskipjump_metrics(
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
) -> dict:
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
            "accuracy_drop_pct": 0.0,
            "asr_pct": 0.0,
            "attack_success_rate_pct": 0.0,
            "adv_accuracy_correct_only": 0.0,
            "er_l2_success": 0.0,
            "er_linf_success": 0.0,
            "mean_l2": 0.0,
            "mean_linf": 0.0,
            "n_eval": float(n_total),
            "n_attacked": 0.0,
            "attack": "HopSkipJump",
            "sample_size": float(n_total),
            "note": "No correctly classified samples in evaluation subset.",
            "params": {
                "max_iter": int(max_iter),
                "max_eval": int(max_eval),
                "init_eval": int(init_eval),
                "init_size": int(init_size),
                "norm": norm,
                "seed": int(seed),
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
    accuracy_drop_pct = float(max(0.0, clean_acc_full - adv_acc_full) * 100.0)

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
        "accuracy_drop_pct": accuracy_drop_pct,
        "asr_pct": asr_pct,
        "attack_success_rate_pct": asr_pct,  # alias for compatibility
        "adv_accuracy_correct_only": adv_acc_correct_only * 100.0,
        "er_l2_success": er_l2_success,
        "er_linf_success": er_linf_success,
        "mean_l2": mean_l2,
        "mean_linf": mean_linf,
        "n_eval": float(n_total),
        "n_attacked": float(n_correct),
        "attack": "HopSkipJump",
        "sample_size": float(n_total),
        "params": {
            "max_iter": int(max_iter),
            "max_eval": int(max_eval),
            "init_eval": int(init_eval),
            "init_size": int(init_size),
            "norm": norm,
            "seed": int(seed),
        },
    }


def compute_clique_method_metrics(
    *,
    model,
    X_test,
    y_test,
    X_train=None,
    n_samples: int = 200,
    seed: int = 42,
    eps_init: float = 0.1,
    norm: float = np.inf,
    nb_search_steps: int = 10,
    max_clique: int = 2,
    max_level: int = 2,
) -> dict:
    RobustnessVerificationTreeModelsCliqueMethod, _ = _safe_import_art_metrics()
    ScikitlearnClassifier = _safe_import_art_sklearn_wrappers()

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

    art_clf = ScikitlearnClassifier(model=model, clip_values=(clip_min, clip_max))
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


def compute_clever_score_metrics(
    *,
    classifier,
    x,
    n_samples: int = 5,
    seed: int = 42,
    nb_batches: int = 10,
    batch_size: int = 32,
    radius: float = 0.5,
    norm: int = 2,
) -> dict:
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


# def compute_psi(expected, actual, bins=10):
#     # crea contenedores de bins
#     breakpoints = np.linspace(0, 1, bins+1)
#     exp_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
#     act_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

#     psi_value = np.sum((exp_percents - act_percents) * np.log(exp_percents / act_percents))
#     return psi_value

# # uso
# train_dist = np.random.rand(1000)
# prod_dist  = np.random.rand(1000) * 1.1

# psi_score = compute_psi(train_dist, prod_dist)
# print("PSI:", psi_score)
