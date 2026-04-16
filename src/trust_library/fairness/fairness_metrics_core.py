"""
fairness_metrics_core.py
===============
Pure mathematical implementations of all fairness metrics.
Zero dependency on AIF360, HolisticAI, or any fairness-specific library.

All functions operate on raw numpy arrays:
    y_true          : ground-truth labels (0/1)
    y_pred          : predicted labels (0/1)
    y_prob          : predicted probabilities for the positive class (float [0,1])
    group_mask      : boolean array - True where sample belongs to the *protected* group
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy, chisquare
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import NearestNeighbors



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
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def _favored_ratio(y_pred: np.ndarray, mask: np.ndarray) -> float:
    """P(y_hat = 1 | group)."""
    if mask.sum() == 0:
        return 0.0
    return float(y_pred[mask].mean())


def _positive_gap_with_favored_group(value: float) -> tuple[float, str] | tuple[float, None]:
    """Return absolute gap and favored group inferred from the original sign.

    Assumes the original metric follows the convention:
    metric = protected - unprotected.
    """
    if value < 0:
        return float(-value), "unprotected"
    if value > 0:
        return float(value), "protected"
    return 0.0, None


# ─────────────────────────────────────────────────────────────────────────────
# Model Performance & Fit Metrics (not fairness-specific, but included for context)
# ─────────────────────────────────────────────────────────────────────────────

def underfitting(y_test: np.ndarray, y_pred_test: np.ndarray) -> dict:
    """
    Test accuracy as a proxy for underfitting.
    
    Parameters
    ----------
    y_test : np.ndarray
        Ground-truth labels for the test set.
    y_pred_test : np.ndarray
        Predicted labels for the test set.

    Returns
    -------
    dict
        Dictionary containing the test accuracy.
    """
    acc = float(accuracy_score(y_test, y_pred_test))
    return {"value": acc}


def overfitting(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> dict:
    """
    Train-Test accuracy gap as a proxy for overfitting.

    Difference = Train Acc - Test Acc. Ideal value: 0 

    Parameters
    ----------
    y_train : np.ndarray
        Ground-truth labels for the training set.
    y_pred_train : np.ndarray
        Predicted labels for the training set.
    y_test : np.ndarray
        Ground-truth labels for the test set.
    y_pred_test : np.ndarray
        Predicted labels for the test set.
    
    Returns
    -------
    dict       
        Dictionary containing the accuracy gap, train accuracy and test accuracy.

    """
    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc  = float(accuracy_score(y_test, y_pred_test))
    diff = train_acc - test_acc
    return {
        "value": diff,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Group Fairness Metrics
# ─────────────────────────────────────────────────────────────────────────────

def statistical_parity_difference(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Statistical Parity Difference (SPD).

    SPD = P(ŷ=1 | protected) - P(ŷ=1 | unprotected)

    Ideal value: 0  (negative -> protected group is disadvantaged)

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The statistical parity difference.
        - favored_ratio_protected: P(ŷ=1 | protected)
        - favored_ratio_unprotected: P(ŷ=1 | unprotected)
        - n_protected: Number of samples in the protected group.
        - n_unprotected: Number of samples in the unprotected group.
        - n_protected_favored: Number of samples in the protected group predicted as 1.
        - n_unprotected_favored: Number of samples in the unprotected group predicted as 1.
    """
    prot  = _favored_ratio(y_pred, group_mask)
    unprot = _favored_ratio(y_pred, ~group_mask)
    
    raw_val = prot - unprot
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "favored_ratio_protected": prot,
        "favored_ratio_unprotected": unprot,
        "n_protected": int(group_mask.sum()),
        "n_unprotected": int((~group_mask).sum()),
        "n_protected_favored": int((y_pred[group_mask] == 1).sum()),
        "n_unprotected_favored": int((y_pred[~group_mask] == 1).sum()),
        "favored_group": favored_group,
    }


def disparate_impact(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Disparate Impact (DI).

    DI = P(ŷ=1 | protected) / P(ŷ=1 | unprotected) or its inverse if >1 to always be in [0,1] where 1 is ideal and lower is worse.

    Ideal value: 1  (< 0.8 is the common legal threshold)

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The disparate impact.
        - favored_ratio_protected: P(ŷ=1 | protected)
        - favored_ratio_unprotected: P(ŷ=1 | unprotected)
        - n_protected: Number of samples in the protected group.
        - n_unprotected: Number of samples in the unprotected group.
        - n_protected_favored: Number of samples in the protected group predicted as 1.
        - n_unprotected_favored: Number of samples in the unprotected group predicted as 1.
    """

    prot  = _favored_ratio(y_pred, group_mask)
    unprot = _favored_ratio(y_pred, ~group_mask)
    val = _safe_div(prot, unprot, default=np.inf)
    favored_group = "unprotected"
    if val > 1:
        val = 1/val  # flip ratio if >1 to always be in [0,1] where 1 is ideal and lower is worse
        favored_group= "protected"
    return {
        "value": val,
        "favored_ratio_protected": prot,
        "favored_ratio_unprotected": unprot,
        "n_protected": int(group_mask.sum()),
        "n_unprotected": int((~group_mask).sum()),
        "n_protected_favored": int((y_pred[group_mask] == 1).sum()),
        "n_unprotected_favored": int((y_pred[~group_mask] == 1).sum()),
        "favored_group": favored_group,
    }


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Equal Opportunity Difference (EOD).

    EOD = TPR(protected) - TPR(unprotected)

    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The equal opportunity difference.
        - tpr_protected: TPR for the protected group.
        - tpr_unprotected: TPR for the unprotected group.

    """
    tpr_prot   = _tpr(y_true, y_pred, group_mask)
    tpr_unprot = _tpr(y_true, y_pred, ~group_mask)
    raw_val = tpr_prot - tpr_unprot
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "tpr_protected": tpr_prot,
        "tpr_unprotected": tpr_unprot,
        "favored_group": favored_group,
    }


def average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Average Odds Difference (AOD).

    AOD = 0.5 x [(TPR_prot - TPR_unprot) + (FPR_prot - FPR_unprot)]

    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The average odds difference.
        - tpr_protected: TPR for the protected group.
        - tpr_unprotected: TPR for the unprotected group.
        - fpr_protected: FPR for the protected group.
        - fpr_unprotected: FPR for the unprotected group.

    """
    tpr_prot   = _tpr(y_true, y_pred, group_mask)
    tpr_unprot = _tpr(y_true, y_pred, ~group_mask)
    fpr_prot   = _fpr(y_true, y_pred, group_mask)
    fpr_unprot = _fpr(y_true, y_pred, ~group_mask)
    raw_val = 0.5 * ((tpr_prot - tpr_unprot) + (fpr_prot - fpr_unprot))
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "tpr_protected": tpr_prot,
        "tpr_unprotected": tpr_unprot,
        "fpr_protected": fpr_prot,
        "fpr_unprotected": fpr_unprot,
        "favored_group": favored_group,
    }


def accuracy_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Accuracy Parity.

    AP = Accuracy(protected) - Accuracy(unprotected)

    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The accuracy parity difference.
        - accuracy_protected: Accuracy for the protected group.
        - accuracy_unprotected: Accuracy for the unprotected group.
    """
    acc_prot   = _accuracy(y_true, y_pred, group_mask)
    acc_unprot = _accuracy(y_true, y_pred, ~group_mask)
    raw_val = acc_prot - acc_unprot
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "accuracy_protected": acc_prot,
        "accuracy_unprotected": acc_unprot,
        "favored_group": favored_group,
    }


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Predictive Parity (average of PPV and NPV gap).

    PP = 0.5 x [(PPV_prot - PPV_unprot) + (NPV_prot - NPV_unprot)]

    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The predictive parity difference.
        - ppv_protected: PPV for the protected group.
        - ppv_unprotected: PPV for the unprotected group.
        - npv_protected: NPV for the protected group.
        - npv_unprotected: NPV for the unprotected group.

    """
    ppv_prot   = _ppv(y_true, y_pred, group_mask)
    ppv_unprot = _ppv(y_true, y_pred, ~group_mask)
    npv_prot   = _npv(y_true, y_pred, group_mask)
    npv_unprot = _npv(y_true, y_pred, ~group_mask)
    raw_val = 0.5 * ((ppv_prot - ppv_unprot) + (npv_prot - npv_unprot))
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "ppv_protected": ppv_prot,
        "ppv_unprotected": ppv_unprot,
        "npv_protected": npv_prot,
        "npv_unprotected": npv_unprot,
        "favored_group": favored_group,
    }


def treatment_equality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Treatment Equality.

    TE = (FN/FP)_protected - (FN/FP)_unprotected

    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing:
        - value: The treatment equality difference.
        - fn_protected: False negatives for the protected group.
        - fp_protected: False positives for the protected group.
        - fn_fp_ratio_protected: Ratio of false negatives to false positives for the protected group.
        - fn_unprotected: False negatives for the unprotected group.
        - fp_unprotected: False positives for the unprotected group.
        - fn_fp_ratio_unprotected: Ratio of false negatives to false positives for the unprotected group.

    """
    def _fn_fp_ratio(mask):
        fn = int(((y_true[mask] == 1) & (y_pred[mask] == 0)).sum())
        fp = int(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum())
        return fn, fp, _safe_div(fn, fp)

    fn_p, fp_p, ratio_p   = _fn_fp_ratio(group_mask)
    fn_u, fp_u, ratio_u   = _fn_fp_ratio(~group_mask)

    if np.isnan(ratio_p) or np.isinf(ratio_p) or np.isnan(ratio_u) or np.isinf(ratio_u):
        raise ValueError("Treatment Equality computation resulted in NaN or Inf. Check if your data has enough samples in each group and class.")
    raw_val = ratio_p - ratio_u
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "fn_protected": fn_p,
        "fp_protected": fp_p,
        "fn_fp_ratio_protected": ratio_p,
        "fn_unprotected": fn_u,
        "fp_unprotected": fp_u,
        "fn_fp_ratio_unprotected": ratio_u,
        "favored_group": favored_group,
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

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_prob : np.ndarray
        Predicted probabilities (float between 0 and 1).
    group_mask : np.ndarray
        Boolean array indicating protected group membership (True for protected).
    n_bins : int, optional
        Number of bins for calibration, by default 10.

    Returns
    -------
    dict
        Dictionary containing:
        - value: The calibration gap.
        - n_bins: The number of bins used.
        - bins: A dictionary with bin information.
    """
    if hasattr(y_true, "dtype") and y_true.dtype.kind == 'f':
            raise ValueError("Not applicable: calibration_gap requires classification probabilities, but a regression problem was detected.")
    
    df = pd.DataFrame({
        "y": y_true,
        "score": y_prob,
        "group": group_mask.astype(int),
    })
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop") # binning by predicted score

    cal = (
        df.groupby(["group", "bin"])["y"]
        .mean()
        .unstack(level=0) # group on columns, bin on rows
    )

    # Ensure both groups exist even when one group has no samples in one or more bins.
    cal = cal.reindex(columns=[0, 1])
    gap = (cal[0] - cal[1]).abs().mean() # average absolute difference across bins
    if np.isnan(gap):
        raise ValueError("Calibration Gap is NaN. Check if your data has enough samples in each bin and group.")
    cal.index = cal.index.astype(str)
    return {
        "value": float(gap),
        "n_bins": n_bins,
        "bins": cal.to_dict(),
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

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_prob : np.ndarray
        Predicted probabilities (float between 0 and 1).
    n_bins : int, optional
        Number of bins for calibration, by default 10.

    Returns
    -------
    dict
        Dictionary containing:
        - value: The well-calibration error.
        - n_bins: The number of bins used.
        - bins: A dictionary with bin information.
    """
    if hasattr(y_true, "dtype") and y_true.dtype.kind == 'f':
            raise ValueError("Not applicable: well_calibration_error requires classification probabilities, but a regression problem was detected.")
    
    df = pd.DataFrame({"y": y_true, "score": y_prob})
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop") # binning by predicted score
    well_cal = df.groupby("bin").apply(
        lambda g: abs(g["y"].mean() - g["score"].mean()) # absolute difference between observed frequency and average predicted probability in the bin
    )
    err = float(well_cal.mean()) # average across bins
    if np.isnan(err):
        raise ValueError("Well-Calibration Error is NaN. Check if your data has enough samples in each bin.")
    well_cal.index = well_cal.index.astype(str)
    return {
        "value": float(err),
        "n_bins": n_bins,
        "bins-scores": well_cal.to_dict(),
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

    Measures algorithmic unfairness by quantifying the inequality in the 
    benefit distribution, based on Speicher et al. (2018).

    Benefit definition (b_i):
    - 1: Correctly predicted (TP or TN) -> Fair treatment.
    - 0: False Negative -> Unfairly disadvantaged.
    - 2: False Positive -> Unfairly advantaged.

    alpha=0 -> Mean Log Deviation
    alpha=1 -> Theil T Index
    alpha=2 -> Half squared coefficient of variation
    Ideal value: 0 (perfect equality / absolute fairness)

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).
    alpha : float, optional
        Parameter for the generalized entropy index, by default 2.

    Returns
    -------
    dict
        Dictionary containing:
        - value: The generalized entropy index.
        - alpha: The alpha parameter used.
        - mean_benefit: The mean benefit.

    References:
    .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
        "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
        ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.

    """
    # b = (y_true == y_pred).astype(float)
    b = (y_pred - y_true + 1).astype(float) # 1 for correct, 0 for false negative, 2 for false positive
    mu = b.mean()

    if mu == 0 or (mu == 1.0 and np.all(b == 1.0)):
        raise ValueError("Generalized Entropy Index is undefined when all benefits are zero or all are one. Check your data and predictions.")

    if alpha == 0:
        # Mean log deviation: -(1/N) * sum(ln(b_i / mu))
        ratio = b / mu
        val = float(np.mean(np.where(ratio > 0, -np.log(ratio), np.inf))) # log(0) treated as infinite inequality
    elif alpha == 1:
        # Theil index: avoid log(0)
        ratio = b / mu
        val = float(np.mean(np.where(ratio > 0, ratio * np.log(ratio), 0)))
    elif alpha == 2:
        val = float(np.mean((b / mu - 1) ** 2) / 2)
    else:
        val = float(np.mean((b / mu) ** alpha - 1) / (alpha * (alpha - 1)))

    return {"value": val, "alpha": alpha, "mean_benefit": float(mu)}


def theil_index(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convenience wrapper for GEI with alpha=1.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).

    Returns
    -------
    dict
        Dictionary containing the Theil index and related information.

    """
    result = generalized_entropy_index(y_true, y_pred, alpha=1)
    result["name"] = "Theil Index"
    return result


def coefficient_of_variation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Coefficient of Variation = sqrt(2 * GEI(alpha=2)).

    Measures relative spread of prediction benefits.
    Ideal value: 0

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    y_pred : np.ndarray
        Predicted labels (binary).

    Returns
    -------
    dict
        Dictionary containing the coefficient of variation and related information.
    """
    gei = generalized_entropy_index(y_true, y_pred, alpha=2)["value"]
    val = float(np.sqrt(2 * gei))
    return {"value": val, "gei_alpha2": gei}


# ─────────────────────────────────────────────────────────────────────────────
# Individual Fairness
# ─────────────────────────────────────────────────────────────────────────────

# def individual_consistency(
#     X: np.ndarray,
#     y_pred: np.ndarray,
#     k: int = 5,
# ) -> dict:
#     """
#     Individual Consistency Score.

#     Consistency = 1 - mean(|ŷ_i - mean(ŷ_neighbours)|)

#     A score of 1 means every individual gets the same prediction as
#     their k nearest neighbours.  Ideal value: 1
#     """
#     nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
#     _, idx = nn.kneighbors(X)

#     diffs = [
#         abs(y_pred[i] - np.mean(y_pred[idx[i][1:]]))
#         for i in range(len(X))
#     ]
#     val = 1.0 - float(np.mean(diffs))
#     return {"value": val, "k": k}

def individual_consistency(X: np.ndarray, y_pred: np.ndarray, k: int = 5) -> dict:
    """
    Individual Consistency Score. It measures how consistent the model's predictions are 
    for similar individuals.

    Consistency = 1 - mean(|ŷ_i - mean(ŷ_neighbours)|)

    A score of 1 means every individual gets the same prediction as
    their k nearest neighbours.  Ideal value: 1

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y_pred : np.ndarray
        Predicted labels (binary).
    k : int, optional
        Number of nearest neighbors to consider, by default 5

    Returns
    -------
    dict
        Dictionary containing the individual consistency score and related information.

    """
    X_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # AIF360 force the algorithm to use 'ball_tree' to avoid warnings about sparse data
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_safe)
    _, idx = nn.kneighbors(X_safe)
    
    # We obtain the predictions of the neighbors (including itself). Shape: (n_samples, k)
    neighbor_preds = y_pred[idx]
    
    # Average prediction of the neighbors (excluding itself). Shape: (n_samples,)
    mean_neighbor_preds = np.mean(neighbor_preds, axis=1)

    # Consistency = 1 - mean( | y_pred - mean_neighbors_pred | )
    diffs = np.abs(y_pred - mean_neighbor_preds)
    val = 1.0 - float(np.mean(diffs))
    
    return {
        "value": val, 
        "k": k
    }

# ─────────────────────────────────────────────────────────────────────────────
# Dataset / Distribution Metrics
# ─────────────────────────────────────────────────────────────────────────────

def class_balance(y_true: np.ndarray) -> dict:
    """
    Chi-square test for class balance in label distribution.

    p >= 0.05  -> classes are balanced
    Ideal p-value: 1.0  (uniform distribution)

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).

    Returns
    -------
    dict
        Dictionary containing the p-value and related information.

    """
    values, counts = np.unique(y_true, return_counts=True)
    stat, p_val = chisquare(counts)
    return {
        "p_value": float(p_val),
        #"chi2_stat": float(stat),
        "balanced": bool(p_val >= 0.05),
        "class_counts": dict(zip(values.tolist(), counts.tolist())),
    }


def class_imbalance(group_mask: np.ndarray) -> dict:
    """
    Class Imbalance between privileged and protected groups.

    CI = (N_unprot - N_prot) / (N_unprot + N_prot)

    Ideal value: 0  (equal group sizes)
    |CI| < 0.1 is considered balanced

    Parameters
    ----------
    group_mask : np.ndarray
        Boolean mask indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing the class imbalance score and related information.
    """
    n_prot   = int(group_mask.sum())
    n_unprot = int((~group_mask).sum())
    val = _safe_div(n_unprot - n_prot, n_unprot + n_prot, default=0.0)
    favored_group = "unprotected"
    if val < 0:
        val = -val  # flip to always be positive, where 0 is ideal and higher is more imbalanced
        favored_group = "protected"
    return {
        "value": float(abs(val)),
        "n_protected": n_prot,
        "n_unprotected": n_unprot,
        "balanced": bool(abs(val) < 0.1),
        "favored_group": favored_group,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Bias Amplification & Effect Size
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence(
    y_true: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    KL Divergence between label distributions of protected vs unprotected group.

    KL(P_privileged || P_protected)
    Ideal value: 0  (identical distributions)

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary).
    group_mask : np.ndarray
        Boolean mask indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing the KL divergence and related information.
    """

    Pp = pd.Series(y_true[~group_mask]).value_counts(normalize=True).sort_index() # label distribution for unprotected group
    Pu = pd.Series(y_true[group_mask]).value_counts(normalize=True).sort_index() # label distribution for protected group
    Pp, Pu = Pp.align(Pu, fill_value=1e-9) # align distributions and avoid log(0) by filling missing classes with a small value

    val = float(scipy_entropy(Pp.values, Pu.values)) # KL divergence from unprotected to protected group
    return {"value": val}

# ─────────────────────────────────────────────────────────────────────────────
# Conditional Demographic Disparity (CDD)
# ─────────────────────────────────────────────────────────────────────────────

def conditional_demographic_disparity(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Conditional Demographic Disparity (CDD) (Wachter et al., 2021).
    
    CDD = (1 / N_total) * sum(N_i * DD_i)
    where DD_i = (N_{i,-} / N_{total,-}) - (N_{i,+} / N_{total,+})
    
    Ideal value: 0

    :math:`N_{i, +}` signifies the number of samples belonging to group
    :math:`i` that have favorable labels while :math:`N_{i, -}` signifies those
    that have negative labels 

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels (binary).
    group_mask : np.ndarray
        Boolean mask indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing the conditional demographic disparity and related information.
    """
    n_prot = int(group_mask.sum())
    n_unprot = int((~group_mask).sum())
    n_total = n_prot + n_unprot
    
    total_pos = int((y_pred == 1).sum())
    total_neg = int((y_pred == 0).sum())

    def _calc_dd(mask):
        if mask.sum() == 0: 
            return 0.0
        n_pos = int((y_pred[mask] == 1).sum())
        n_neg = int((y_pred[mask] == 0).sum())
        
        term_neg = (n_neg / total_neg) if total_neg > 0 else 0.0
        term_pos = (n_pos / total_pos) if total_pos > 0 else 0.0
        return float(term_neg - term_pos)

    dd_prot = _calc_dd(group_mask)
    dd_unprot = _calc_dd(~group_mask)

    if n_total == 0:
        cdd = 0.0
    else:
        cdd = (n_prot * dd_prot + n_unprot * dd_unprot) / n_total

    val, favored_group = _positive_gap_with_favored_group(float(cdd))

    return {
        "value": val,
        "n_protected": n_prot,
        "n_unprotected": n_unprot,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "dd_protected": dd_prot,
        "dd_unprotected": dd_unprot,
        "favored_group": favored_group,
    }

# def smoothed_edf(
#     y_prob: np.ndarray,
#     group_values: np.ndarray,
#     alpha: float = 1.0,
# ) -> dict:
#     """
#     Smoothed Empirical Differential Fairness (EDF).

#     Based on log-ratio of (mean_prob + alpha) across groups.
#     Ideal value: 0
#     """
#     unique_groups = np.unique(group_values)
#     group_means = {g: float(y_prob[group_values == g].mean()) + alpha for g in unique_groups}
#     vals = list(group_means.values())
#     ratio = max(vals) / min(vals)
#     val = float(abs(np.log(ratio)))
#     return {
#         "value": val,
#         "group_smoothed_means": {str(k): v for k, v in group_means.items()},
#         "alpha": alpha,
#     }

def smoothed_edf(y_prob: np.ndarray, group_values: np.ndarray, alpha: float = 1.0) -> dict: # NO IGUAL QUE AIF
    """
    Smoothed Empirical Differential Fairness (EDF) manual.
    AIF360 replic: EDF = max_{i,j} max( |log((p_i + alpha) / (p_j + alpha))|, |log((1-p_i + alpha) / (1-p_j + alpha))| )
    Ideal value: 0

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities (binary).
    group_values : np.ndarray
        Group membership labels.
    alpha : float, optional
        Smoothing parameter (default: 1.0).

    Returns
    -------
    dict
        Dictionary containing the smoothed EDF and related information.

    """
    y_pred_bin = (np.array(y_prob) >= 0.5).astype(float) # binarice predictions
    
    unique_groups = np.unique(group_values)
    ssr = [] # smoothed_selection_rates
    
    for g in unique_groups:
        mask_g = (group_values == g)
        n_g = np.sum(mask_g)
        p_g = np.sum(y_pred_bin[mask_g] == 1.0)
        
        rate = (p_g + alpha) / (n_g + 2.0 * alpha) # smoothed selection rate for group g
        ssr.append(rate)
    
    max_edf = 0.0
    for i in range(len(ssr)):
        for j in range(len(ssr)):
            if i != j:
                pos_ratio = abs(np.log(ssr[i]) - np.log(ssr[j])) # log-ratio of smoothed selection rates
                neg_ratio = abs(np.log(1.0 - ssr[i]) - np.log(1.0 - ssr[j])) # log-ratio of smoothed non-selection rates
                max_edf = max(max_edf, pos_ratio, neg_ratio) # maximum of the two ratios across all pairs of groups

    return {
        "value": float(max_edf),
        "alpha": alpha,
        "group_smoothed_selection_rates": {str(g): ssr[i] for i, g in enumerate(unique_groups)}
    }

# def bias_amplification(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     group_mask: np.ndarray,
# ) -> dict:
#     """
#     Bias Amplification.

#     Measures whether the model *amplifies* the bias already present in labels.

#     BA = |bias(ŷ)| - |bias(y)|
#     where bias = mean(group_prot) - mean(group_unprot)

#     Ideal value: 0 or negative (model should not amplify bias)
#     """
#     bias_y    = abs(float(y_true[group_mask].mean()) - float(y_true[~group_mask].mean()))
#     bias_yhat = abs(float(y_pred[group_mask].mean()) - float(y_pred[~group_mask].mean()))
#     val = bias_yhat - bias_y
#     return {
#         "value": val,
#         "bias_in_labels": bias_y,
#         "bias_in_predictions": bias_yhat,
#         "amplified": bool(val > 0),
#     }


def bias_amplification(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray) -> dict: # NO IGUAL QUE AIF
    """
    Bias Amplification manual (Differential Fairness).
    AIF360 replic: BA = EDF(y_pred) - EDF(y_true)
    Ideal value: 0 or negative (model should not amplify bias)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    group_mask : np.ndarray
        Boolean mask indicating protected group membership (True for protected).

    Returns
    -------
    dict
        Dictionary containing the bias amplification and related information.
    """
    # Compute smoothed EDF for both labels and predictions
    bias_labels = smoothed_edf(y_true, group_mask, alpha=0.5)["value"]
    bias_preds = smoothed_edf(y_pred, group_mask, alpha=0.5)["value"]
    
    val = bias_preds - bias_labels
    
    return {
        "value": float(val),
        "bias_in_labels": float(bias_labels),
        "bias_in_predictions": float(bias_preds)
    }

# def bias_amplification(y_true: np.ndarray, y_pred: np.ndarray, group_mask: np.ndarray, alpha: float = 1.0) -> dict:
#     """
#     Bias Amplification replicando AIF360:
#     BA = EDF(y_pred) - EDF(y_true)
#     """
#     # Binarize if input is not binary yet
#     y_true_bin = (y_true >= 0.5).astype(int)
#     y_pred_bin = (y_pred >= 0.5).astype(int)
    
#     # EDF suavizado de labels
#     edf_labels = smoothed_edf(y_true_bin, group_mask, alpha)["value"]
#     # EDF suavizado de predicciones
#     edf_preds  = smoothed_edf(y_pred_bin, group_mask, alpha)["value"]
    
#     ba = edf_preds - edf_labels
    
#     return {
#         "value": ba,
#         "bias_in_labels": edf_labels,
#         "bias_in_predictions": edf_preds
#     }

# ─────────────────────────────────────────────────────────────────────────────
# Between-Group Generalized Entropy Error
# ─────────────────────────────────────────────────────────────────────────────

def between_group_generalized_entropy_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray,
    alpha: float = 2
) -> dict:
    """
    Between-Group Generalized Entropy Error (Speicher et al., 2018).
    
    Between-group generalized entropy index is proposed as a group
    fairness measure and is one of two terms that the
    generalized entropy index decomposes to.
    
    Parameters
    ----------
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        group_mask (array-like): Boolean mask indicating protected group membership (True for protected).
        alpha (scalar, optional): Parameter that regulates the weight given to
            distances between values at different parts of the distribution. A
            value of 0 is equivalent to the mean log deviation, 1 is the Theil
            index, and 2 is half the squared coefficient of variation.

    Returns
    -------
        dict: A dictionary containing the between-group generalized entropy error and related information.
    """
    b = np.empty_like(y_true, dtype=float)
    
    # Benefit = 1 + I(y_pred == 1) - I(y_true == 1)
    # Correct prediction (TP or TN) -> b = 1 (fair treatment)
    # False Negative -> b = 0 (unfairly disadvantaged)
    mask_u = ~group_mask
    if mask_u.sum() > 0:
        b[mask_u] = (1.0 + (y_pred[mask_u] == 1) - (y_true[mask_u] == 1)).mean()
        
    mask_p = group_mask
    if mask_p.sum() > 0:
        b[mask_p] = (1.0 + (y_pred[mask_p] == 1) - (y_true[mask_p] == 1)).mean()

    mu = float(b.mean())
    
    if mu == 0:
        val = 0.0
    elif alpha == 1:
        ratio = b / mu
        val = float(np.mean(np.where(ratio > 0, ratio * np.log(ratio), 0)))
    elif alpha == 2:
        val = float(np.mean((b / mu - 1) ** 2) / 2)
    else:
        val = float(np.mean((b / mu) ** alpha - 1) / (alpha * (alpha - 1)))

    return {
        "value": val,
        "alpha": alpha,
        "mean_benefit": mu,
    }

def cohens_d(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Cohen's D effect size between protected and unprotected groups.

    d = (μ_prot - μ_unprot) / sigma_pooled

    |d| < 0.2 -> negligible, 0.2-0.5 -> small, 0.5-0.8 -> medium, > 0.8 -> large

    Parameters
    ----------
    y_pred : array-like
        Predictions vector (binary)
    group_mask : array-like
        Boolean mask indicating protected group membership (True for protected)

    Returns
    -------
    dict
        A dictionary containing the Cohen's D effect size and related information.

    """
    g1 = y_pred[group_mask].astype(float)
    g2 = y_pred[~group_mask].astype(float)
    if len(g1) == 0 or len(g2) == 0:
        raise ValueError("Cohen's D requires samples in both protected and unprotected groups.")
    sigma = float(np.sqrt((g1.var() + g2.var()) / 2))
    raw_val = float((g1.mean() - g2.mean()) / sigma) if sigma > 0 else 0.0
    val, favored_group = _positive_gap_with_favored_group(raw_val)
    return {
        "value": val,
        "mean_protected": float(g1.mean()),
        "mean_unprotected": float(g2.mean()),
        "pooled_std": sigma,
        "favored_group": favored_group,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Z-Test Difference (2-SD Rule)
# ─────────────────────────────────────────────────────────────────────────────

def z_test_diff(
    y_pred: np.ndarray,
    group_mask: np.ndarray,
) -> dict:
    """
    Z Test (Difference) / 2-SD Statistic (Morris, 2001).
    This function computes the Z-test statistic for the difference\
    in success rates. Also known as 2-SD Statistic.

    A value of 0 is desired. This test considers the data unfair if\
    the computed value is greater than 2 or smaller than -2, indicating\
    a statistically significant difference in success rates.

    Parameters
    ----------
    y_pred : array-like
        Predictions vector (binary)
    group_mask : array-like
        Boolean mask indicating protected group membership (True for protected)

    Returns
    -------
    float
        Z test (difference version)

    Z = (SR_prot - SR_unprot) / sqrt((SR_tot * (1 - SR_tot)) / (N_tot * P_prot * (1 - P_prot)))
    """
    n_prot = int(group_mask.sum())
    n_unprot = int((~group_mask).sum())
    
    if n_prot == 0 or n_unprot == 0:
        raise ValueError("Z-test computation resulted in NaN or Inf. Check if your data has enough samples in each group and that success rates are not 0 or 1 for either group.")
        #return {"value": 0.0, "sr_protected": 0.0, "sr_unprotected": 0.0}

    sr_prot = float(y_pred[group_mask].mean())
    sr_unprot = float(y_pred[~group_mask].mean())
    sr_tot = float(y_pred.mean())

    n_tot = n_prot + n_unprot
    p_prot = n_prot / n_tot

    denom = np.sqrt((sr_tot * (1 - sr_tot)) / (n_tot * p_prot * (1 - p_prot)))
    
    raw_val = 0.0 if denom == 0 else (sr_prot - sr_unprot) / denom
    val, favored_group = _positive_gap_with_favored_group(raw_val)

    if np.isnan(val) or np.isinf(val):
        raise ValueError("Z-test computation resulted in NaN or Inf. Check if your data has enough samples in each group and that success rates are not 0 or 1 for either group.")
        #val = 0.0

    return {
        "value": float(val),
        "sr_protected": sr_prot,
        "sr_unprotected": sr_unprot,
        "total_success_rate": sr_tot,
        "n_protected": n_prot,
        "n_unprotected": n_unprot,
        "favored_group": favored_group,
    }


# For regression: https://github.com/holistic-ai/holisticai/blob/main/src/holisticai/bias/metrics/_regression.py#L56
# For multiclass: https://github.com/holistic-ai/holisticai/blob/main/src/holisticai/bias/metrics/_multiclass.py