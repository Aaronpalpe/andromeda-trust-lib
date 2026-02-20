"""
fairness_metrics_core.py
===============
Pure mathematical implementations of all fairness metrics.
Zero dependency on AIF360, HolisticAI, or any fairness-specific library.

All functions operate on raw numpy arrays:
    y_true          : ground-truth labels (0/1)
    y_pred          : predicted labels (0/1)
    y_prob          : predicted probabilities for the positive class (float [0,1])
    group_mask      : boolean array — True where sample belongs to the *protected* group
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy, chisquare
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = np.inf) -> float:
    """Division with a safe fallback when denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def _tpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """True Positive Rate for a subgroup defined by *mask*."""
    positives = y_true[mask] == 1
    if positives.sum() == 0:
        return 0.0
    return float((y_pred[mask][positives] == 1).sum() / positives.sum())


def _fpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """False Positive Rate for a subgroup defined by *mask*."""
    negatives = y_true[mask] == 0
    if negatives.sum() == 0:
        return 0.0
    return float((y_pred[mask][negatives] == 1).sum() / negatives.sum())


def _ppv(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Positive Predictive Value (Precision) for a subgroup."""
    pred_pos = y_pred[mask] == 1
    if pred_pos.sum() == 0:
        return 0.0
    return float((y_true[mask][pred_pos] == 1).sum() / pred_pos.sum())


def _npv(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Negative Predictive Value for a subgroup."""
    pred_neg = y_pred[mask] == 0
    if pred_neg.sum() == 0:
        return 0.0
    return float((y_true[mask][pred_neg] == 0).sum() / pred_neg.sum())


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def _favored_ratio(y_pred: np.ndarray, mask: np.ndarray) -> float:
    """P(y_hat = 1 | group)."""
    if mask.sum() == 0:
        return 0.0
    return float(y_pred[mask].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Group Fairness Metrics
# ─────────────────────────────────────────────────────────────────────────────

def statistical_parity_difference(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Statistical Parity Difference (SPD).

    SPD = P(ŷ=1 | protected) − P(ŷ=1 | unprotected)

    Ideal value: 0  (negative → protected group is disadvantaged)
    """
    prot  = _favored_ratio(y_pred, group_mask)
    unprot = _favored_ratio(y_pred, ~group_mask)
    val = prot - unprot
    return {
        "value": val,
        "favored_ratio_protected": prot,
        "favored_ratio_unprotected": unprot,
        "n_protected": int(group_mask.sum()),
        "n_unprotected": int((~group_mask).sum()),
    }


def disparate_impact(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Disparate Impact (DI).

    DI = P(ŷ=1 | protected) / P(ŷ=1 | unprotected)

    Ideal value: 1  (< 0.8 is the common legal threshold)
    """
    prot  = _favored_ratio(y_pred, group_mask)
    unprot = _favored_ratio(y_pred, ~group_mask)
    val = _safe_div(prot, unprot, default=np.inf)
    return {
        "value": val,
        "favored_ratio_protected": prot,
        "favored_ratio_unprotected": unprot,
    }


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Equal Opportunity Difference (EOD).

    EOD = TPR(protected) − TPR(unprotected)

    Ideal value: 0
    """
    tpr_prot   = _tpr(y_true, y_pred, group_mask)
    tpr_unprot = _tpr(y_true, y_pred, ~group_mask)
    val = tpr_prot - tpr_unprot
    return {
        "value": val,
        "tpr_protected": tpr_prot,
        "tpr_unprotected": tpr_unprot,
    }


def average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Average Odds Difference (AOD).

    AOD = 0.5 × [(TPR_prot − TPR_unprot) + (FPR_prot − FPR_unprot)]

    Ideal value: 0
    """
    tpr_prot   = _tpr(y_true, y_pred, group_mask)
    tpr_unprot = _tpr(y_true, y_pred, ~group_mask)
    fpr_prot   = _fpr(y_true, y_pred, group_mask)
    fpr_unprot = _fpr(y_true, y_pred, ~group_mask)
    val = 0.5 * ((tpr_prot - tpr_unprot) + (fpr_prot - fpr_unprot))
    return {
        "value": val,
        "tpr_protected": tpr_prot,
        "tpr_unprotected": tpr_unprot,
        "fpr_protected": fpr_prot,
        "fpr_unprotected": fpr_unprot,
    }


def accuracy_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Accuracy Parity.

    AP = Accuracy(protected) − Accuracy(unprotected)

    Ideal value: 0
    """
    acc_prot   = _accuracy(y_true, y_pred, group_mask)
    acc_unprot = _accuracy(y_true, y_pred, ~group_mask)
    return {
        "value": acc_prot - acc_unprot,
        "accuracy_protected": acc_prot,
        "accuracy_unprotected": acc_unprot,
    }


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Predictive Parity (average of PPV and NPV gap).

    PP = 0.5 × [(PPV_prot − PPV_unprot) + (NPV_prot − NPV_unprot)]

    Ideal value: 0
    """
    ppv_prot   = _ppv(y_true, y_pred, group_mask)
    ppv_unprot = _ppv(y_true, y_pred, ~group_mask)
    npv_prot   = _npv(y_true, y_pred, group_mask)
    npv_unprot = _npv(y_true, y_pred, ~group_mask)
    val = 0.5 * ((ppv_prot - ppv_unprot) + (npv_prot - npv_unprot))
    return {
        "value": val,
        "ppv_protected": ppv_prot,
        "ppv_unprotected": ppv_unprot,
        "npv_protected": npv_prot,
        "npv_unprotected": npv_unprot,
    }


def treatment_equality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Treatment Equality.

    TE = (FN/FP)_protected − (FN/FP)_unprotected

    Ideal value: 0
    """
    def _fn_fp_ratio(mask):
        fn = int(((y_true[mask] == 1) & (y_pred[mask] == 0)).sum())
        fp = int(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum())
        return fn, fp, _safe_div(fn, fp)

    fn_p, fp_p, ratio_p   = _fn_fp_ratio(group_mask)
    fn_u, fp_u, ratio_u   = _fn_fp_ratio(~group_mask)
    return {
        "value": ratio_p - ratio_u,
        "fn_protected": fn_p,
        "fp_protected": fp_p,
        "fn_fp_ratio_protected": ratio_p,
        "fn_unprotected": fn_u,
        "fp_unprotected": fp_u,
        "fn_fp_ratio_unprotected": ratio_u,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calibration_gap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_mask: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Calibration Gap across groups.

    Bins individuals by their predicted score, then measures how much
    the actual outcome frequency differs between protected and unprotected
    groups within the same bin.

    Ideal value: 0
    """

    df = pd.DataFrame({
        "y": y_true,
        "score": y_prob,
        "group": group_mask.astype(int),
    })
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop")

    cal = (
        df.groupby(["group", "bin"])["y"]
        .mean()
        .unstack(level=0)
    )
    gap = (cal.iloc[:, 0] - cal.iloc[:, 1]).abs().mean()
    return {
        "value": float(gap),
        "n_bins": n_bins,
    }


def well_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Well-Calibration Error (Expected Calibration Error variant).

    Measures how far average predicted probability deviates from
    observed event frequency within each bin.

    Ideal value: 0
    """

    df = pd.DataFrame({"y": y_true, "score": y_prob})
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop")
    err = df.groupby("bin").apply(
        lambda g: abs(g["y"].mean() - g["score"].mean())
    ).mean()
    return {
        "value": float(err),
        "n_bins": n_bins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inequality / Information-Theory Metrics
# ─────────────────────────────────────────────────────────────────────────────

def generalized_entropy_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 2,
) -> dict:
    """
    Generalized Entropy Index (GEI).

    Measures inequality in benefit distribution (b_i = 1 if correctly
    predicted, 0 otherwise).

    alpha=1 → Theil T Index
    alpha=2 → Half squared coefficient of variation
    Ideal value: 0  (perfect equality)
    """
    b = (y_true == y_pred).astype(float)
    mu = b.mean()
    n = len(b)

    if mu == 0:
        return {"value": 0.0, "alpha": alpha}

    if alpha == 1:
        # Theil index: avoid log(0)
        ratio = b / mu
        val = float(np.mean(np.where(ratio > 0, ratio * np.log(ratio), 0)))
    elif alpha == 2:
        val = float(np.mean((b / mu - 1) ** 2) / 2)
    else:
        val = float(np.mean((b / mu) ** alpha - 1) / (alpha * (alpha - 1)))

    return {"value": val, "alpha": alpha, "mean_benefit": float(mu)}


def theil_index(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper for GEI with alpha=1."""
    result = generalized_entropy_index(y_true, y_pred, alpha=1)
    result["name"] = "Theil Index"
    return result


def coefficient_of_variation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Coefficient of Variation = sqrt(2 * GEI(alpha=2)).

    Measures relative spread of prediction benefits.
    Ideal value: 0
    """
    gei = generalized_entropy_index(y_true, y_pred, alpha=2)["value"]
    val = float(np.sqrt(2 * gei))
    return {"value": val, "gei_alpha2": gei}


def kl_divergence(
    y_true: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    KL Divergence between label distributions of protected vs unprotected group.

    KL(P_privileged || P_protected)
    Ideal value: 0  (identical distributions)
    """

    Pp = pd.Series(y_true[~group_mask]).value_counts(normalize=True).sort_index()
    Pu = pd.Series(y_true[group_mask]).value_counts(normalize=True).sort_index()
    Pp, Pu = Pp.align(Pu, fill_value=1e-9)

    val = float(scipy_entropy(Pp.values, Pu.values))
    return {"value": val}


# ─────────────────────────────────────────────────────────────────────────────
# Individual Fairness
# ─────────────────────────────────────────────────────────────────────────────

def individual_consistency(
    X: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Individual Consistency Score.

    Consistency = 1 − mean(|ŷ_i − mean(ŷ_neighbours)|)

    A score of 1 means every individual gets the same prediction as
    their k nearest neighbours.  Ideal value: 1
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)

    diffs = [
        abs(y_pred[i] - np.mean(y_pred[idx[i][1:]]))
        for i in range(len(X))
    ]
    val = 1.0 - float(np.mean(diffs))
    return {"value": val, "k": k}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / Distribution Metrics
# ─────────────────────────────────────────────────────────────────────────────

def class_balance(y_true: np.ndarray) -> dict:
    """
    Chi-square test for class balance in label distribution.

    p >= 0.05  → classes are balanced
    Ideal p-value: 1.0  (uniform distribution)
    """
    values, counts = np.unique(y_true, return_counts=True)
    stat, p_val = chisquare(counts)
    return {
        "p_value": float(p_val),
        "chi2_stat": float(stat),
        "balanced": bool(p_val >= 0.05),
        "class_counts": dict(zip(values.tolist(), counts.tolist())),
    }


def class_imbalance(group_mask: np.ndarray) -> dict:
    """
    Class Imbalance between privileged and protected groups.

    CI = (N_unprot − N_prot) / (N_unprot + N_prot)

    Ideal value: 0  (equal group sizes)
    |CI| < 0.1 is considered balanced
    """
    n_prot   = int(group_mask.sum())
    n_unprot = int((~group_mask).sum())
    val = _safe_div(n_unprot - n_prot, n_unprot + n_prot, default=0.0)
    return {
        "value": float(val),
        "n_protected": n_prot,
        "n_unprotected": n_unprot,
        "balanced": bool(abs(val) < 0.1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Performance & Fit Metrics (not fairness-specific, but included for context)
# ─────────────────────────────────────────────────────────────────────────────

def underfitting(y_test: np.ndarray, y_pred_test: np.ndarray) -> dict:
    """Test accuracy as a proxy for underfitting."""
    acc = float(accuracy_score(y_test, y_pred_test))
    return {"value": acc, "test_accuracy": acc}


def overfitting(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> dict:
    """
    Train–Test accuracy gap as a proxy for overfitting.

    Difference = Train Acc − Test Acc
    Ideal value: 0  (> 0.05 indicates overfitting)
    """
    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc  = float(accuracy_score(y_test, y_pred_test))
    diff = train_acc - test_acc
    return {
        "value": diff,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "overfitting": bool(diff > 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bias Amplification & Effect Size
# ─────────────────────────────────────────────────────────────────────────────

def bias_amplification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Bias Amplification.

    Measures whether the model *amplifies* the bias already present in labels.

    BA = |bias(ŷ)| − |bias(y)|
    where bias = mean(group_prot) − mean(group_unprot)

    Ideal value: 0 or negative (model should not amplify bias)
    """
    bias_y    = abs(float(y_true[group_mask].mean()) - float(y_true[~group_mask].mean()))
    bias_yhat = abs(float(y_pred[group_mask].mean()) - float(y_pred[~group_mask].mean()))
    val = bias_yhat - bias_y
    return {
        "value": val,
        "bias_in_labels": bias_y,
        "bias_in_predictions": bias_yhat,
        "amplified": bool(val > 0),
    }


def cohens_d(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Cohen's D effect size between protected and unprotected groups.

    d = (μ_prot − μ_unprot) / σ_pooled

    |d| < 0.2 → negligible, 0.2–0.5 → small, 0.5–0.8 → medium, > 0.8 → large
    """
    g1 = y_pred[group_mask].astype(float)
    g2 = y_pred[~group_mask].astype(float)
    sigma = float(np.sqrt((g1.var() + g2.var()) / 2))
    val = float((g1.mean() - g2.mean()) / sigma) if sigma > 0 else 0.0
    return {
        "value": val,
        "mean_protected": float(g1.mean()),
        "mean_unprotected": float(g2.mean()),
        "pooled_std": sigma,
    }


def smoothed_edf(
    y_prob: np.ndarray,
    group_values: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """
    Smoothed Empirical Differential Fairness (EDF).

    Based on log-ratio of (mean_prob + alpha) across groups.
    Ideal value: 0
    """
    unique_groups = np.unique(group_values)
    group_means = {g: float(y_prob[group_values == g].mean()) + alpha for g in unique_groups}
    vals = list(group_means.values())
    ratio = max(vals) / min(vals)
    val = float(abs(np.log(ratio)))
    return {
        "value": val,
        "group_smoothed_means": {str(k): v for k, v in group_means.items()},
        "alpha": alpha,
    }