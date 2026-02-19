"""
fairness.py
===========
Main entry point for the fairness analysis pipeline.

Contains in a single module:
  - EvaluationContext   — imported from utils; holds data, predictions, and factsheet
  - Scoring functions   — one per metric, delegating math to metrics_core
  - analyse()           — main orchestrator

External library dependencies (AIF360, HolisticAI) are isolated in
adapters.py and imported optionally in a non-blocking manner.
"""

from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from . import metrics_core as core
from trust_library.utils import Result, calculate_score, EvaluationContext


# ─────────────────────────────────────────────────────────────────────────────
# Threshold helper
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]

def _thresh(config: dict, key: str) -> list:
    """
    Retrieve threshold list for a given metric key from the config dict.
    Falls back to _DEFAULT_THRESHOLDS if the key is missing or None.
    """
    val = config.get(key, {}).get("thresholds", {}).get("value")
    if val is None:
        print(f"Warning: thresholds not found for '{key}'. Using default {_DEFAULT_THRESHOLDS}.")
        return _DEFAULT_THRESHOLDS
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def underfitting_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """Compares test accuracy against a baseline to detect underfitting."""
    try:
        res = core.underfitting(ctx.y_test, ctx.y_pred_test)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Compares the model's test accuracy against a baseline.",
            "Depends on": "Model, Test Data",
            "Test Accuracy": f"{res['test_accuracy']:.2%}",
            "Conclusion": "Model mildly underfitting" if score <= 3 else "Model well fitted",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def overfitting_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """Uses the train-test accuracy gap as a proxy for overfitting."""
    try:
        res = core.overfitting(ctx.y_train, ctx.y_pred_train, ctx.y_test, ctx.y_pred_test)
        if res["test_accuracy"] < 0.6:
            return Result(np.nan, {"Info": "Test accuracy < 60%; overfitting not evaluated."})
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Detects overfitting via the train-test accuracy difference.",
            "Depends on": "Model, Training Data, Test Data",
            "Training Accuracy": f"{res['train_accuracy']:.2%}",
            "Test Accuracy": f"{res['test_accuracy']:.2%}",
            "Train Test Accuracy Difference": f"{res['value']:.2%}",
            "Conclusion": "Model is overfitting" if res["overfitting"] else "Model is not overfitting",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def class_balance_score(ctx: EvaluationContext, thresholds=None) -> Result:
    """Checks whether the training label distribution is balanced (chi-square test)."""
    try:
        res = core.class_balance(ctx.train_data[ctx.target_column].to_numpy())
        score = 5 if res["balanced"] else 1
        return Result(score, {
            "Metric Description": "Measures whether the training data classes are balanced.",
            "Depends on": "Training Data",
            "P-Value": f"{res['p_value']:.4f}",
            "Class Counts": str(res["class_counts"]),
            "Conclusion": "Classes are balanced" if res["balanced"] else "Classes are imbalanced",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def stat_parity_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Statistical Parity Difference.
    Measures the gap between favorable prediction rates across groups.
    Ideal value: 0.
    """
    try:
        res = core.statistical_parity_difference(ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Spread between favorable outcome rates across protected and unprotected groups.",
            "Depends on": "Training Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "Favored Protected (Minority) Ratio": f"P(y_hat=favorable|protected=True) = {res['favored_ratio_protected']:.2%}",
            "Favored Unprotected (Majority) Ratio": f"P(y_hat=favorable|protected=False) = {res['favored_ratio_unprotected']:.2%}",
            "Formula": "Favored Majority Ratio - Favored Minority Ratio",
            "Statistical Parity Difference": f"{res['value']*100:.2f}%",
        })
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def disparate_impact_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Disparate Impact.
    Ratio of favorable prediction rates between protected and unprotected groups.
    Ideal value: 1. Values below 0.8 are commonly flagged as discriminatory.
    """
    try:
        res = core.disparate_impact(ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Ratio of favorable prediction rates between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "Protected Favored Ratio": f"P(y_hat=favorable|protected=True) = {res['favored_ratio_protected']:.2%}",
            "Unprotected Favored Ratio": f"P(y_hat=favorable|protected=False) = {res['favored_ratio_unprotected']:.2%}",
            "Formula": "Protected Favored Ratio / Unprotected Favored Ratio",
            "Disparate Impact": f"{res['value']:.4f}",
        })
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def eq_opp_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Equal Opportunity Difference.
    Difference in True Positive Rates between protected and unprotected groups.
    Ideal value: 0.
    """
    try:
        res = core.equal_opportunity_difference(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Difference in true positive rates between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "TPR Unprotected Group": f"{res['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{res['tpr_protected']:.2%}",
            "Formula": "Equal Opportunity Difference = TPR Protected - TPR Unprotected",
            "Equal Opportunity Difference": f"{res['value']*100:.2f}%",
        })
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def avg_odds_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Average Odds Difference.
    Average of the TPR gap and FPR gap between groups.
    Ideal value: 0.
    """
    try:
        res = core.average_odds_difference(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Average of the TPR and FPR differences between groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "FPR Unprotected Group": f"{res['fpr_unprotected']:.2%}",
            "FPR Protected Group": f"{res['fpr_protected']:.2%}",
            "TPR Unprotected Group": f"{res['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{res['tpr_protected']:.2%}",
            "Formula": "0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)",
            "Average Odds Difference": f"{res['value']*100:.2f}%",
        })
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def accuracy_parity_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Accuracy Parity.
    Checks whether prediction accuracy is equal across groups.
    Ideal value: 0.
    """
    try:
        res = core.accuracy_parity(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures whether prediction accuracy is equal across groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "Accuracy Unprotected Group": f"{res['accuracy_unprotected']:.2%}",
            "Accuracy Protected Group": f"{res['accuracy_protected']:.2%}",
            "Formula": "Accuracy Parity = Accuracy Protected - Accuracy Unprotected",
            "Accuracy Parity Difference": f"{res['value']*100:.2f}%",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def predictive_parity_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Predictive Parity.
    Checks equality of PPV and NPV between protected and unprotected groups.
    Ideal value: 0.
    """
    try:
        res = core.predictive_parity(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Checks equality of PPV and NPV between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "PPV Unprotected": f"{res['ppv_unprotected']:.2%}",
            "PPV Protected": f"{res['ppv_protected']:.2%}",
            "NPV Unprotected": f"{res['npv_unprotected']:.2%}",
            "NPV Protected": f"{res['npv_protected']:.2%}",
            "Formula": "0.5*(PPV Protected - PPV Unprotected) + 0.5*(NPV Protected - NPV Unprotected)",
            "Predictive Parity Difference": f"{res['value']*100:.2f}%",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def treatment_equality_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Treatment Equality.
    Measures equality of the FN/FP ratio across groups.
    Ideal value: 0.
    """
    try:
        res = core.treatment_equality(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures equality of the false negative / false positive ratio across groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "FN Unprotected": res["fn_unprotected"],
            "FP Unprotected": res["fp_unprotected"],
            "FN Protected":   res["fn_protected"],
            "FP Protected":   res["fp_protected"],
            "FN/FP Unprotected": f"{res['fn_fp_ratio_unprotected']:.4f}",
            "FN/FP Protected":   f"{res['fn_fp_ratio_protected']:.4f}",
            "Formula": "(FN/FP)_Protected - (FN/FP)_Unprotected",
            "Treatment Equality Difference": f"{res['value']:.4f}",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def calibration_score(ctx: EvaluationContext, thresholds: list, n_bins: int = 10) -> Result:
    """
    Calibration Gap.
    Checks whether individuals with the same predicted score have the same
    empirical outcome probability across groups.
    Ideal value: 0.
    """
    try:
        if ctx.y_prob_positive is None:
            return Result(np.nan, {"Info": "predict_proba is required for this metric."})
        res = core.calibration_gap(ctx.y_test, ctx.y_prob_positive, ctx.group_mask, n_bins)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": (
                "Checks whether individuals with the same predicted score have the same "
                "empirical outcome probability in both groups."
            ),
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Gap": f"{res['value']:.4f}",
            "Bins Used": n_bins,
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def well_calibration_score(ctx: EvaluationContext, thresholds: list, n_bins: int = 10) -> Result:
    """
    Well-Calibration Error.
    Checks whether predicted probabilities match observed outcome frequencies.
    Ideal value: 0.
    """
    try:
        if ctx.y_prob_positive is None:
            return Result(np.nan, {"Info": "predict_proba is required for this metric."})
        res = core.well_calibration_error(ctx.y_test, ctx.y_prob_positive, n_bins)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Checks whether predicted probabilities match empirical outcome frequencies.",
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Error": f"{res['value']:.4f}",
            "Bins Used": n_bins,
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def generalized_entropy_score(ctx: EvaluationContext, thresholds: list, alpha: float = 2) -> Result:
    """
    Generalized Entropy Index.
    Measures inequality in the distribution of correct predictions (benefits).
    alpha=2 → half squared coefficient of variation.
    Ideal value: 0 (perfect equality).
    """
    try:
        res = core.generalized_entropy_index(ctx.y_test, ctx.y_pred_test, alpha)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures inequality in the distribution of correct predictions.",
            "Depends on": "Model, Test Data",
            "Alpha": alpha,
            "Benefit Definition": "b_i = 1 if prediction is correct, 0 otherwise",
            "Generalized Entropy Index": f"{res['value']:.6f}",
            "Conclusion": "Perfect equality" if res["value"] == 0 else "Prediction benefits unevenly distributed",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def theil_index_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """Theil Index — Generalized Entropy Index with alpha=1."""
    return generalized_entropy_score(ctx, thresholds, alpha=1)


def coefficient_variation_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Coefficient of Variation.
    Measures the relative spread (instability) of prediction benefits.
    Ideal value: 0.
    """
    try:
        res = core.coefficient_of_variation(ctx.y_test, ctx.y_pred_test)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures the relative dispersion of prediction benefits.",
            "Depends on": "Model, Test Data",
            "Coefficient of Variation": f"{res['value']:.6f}",
            "Formula": "CV = sqrt(2 * GEI(alpha=2))",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def consistency_score(ctx: EvaluationContext, thresholds: list, k: int = 5) -> Result:
    """
    Individual Consistency.
    Similar individuals should receive similar predictions.
    Ideal value: 1.
    """
    try:
        res = core.individual_consistency(ctx.X_test, ctx.y_pred_test, k)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures whether similar individuals receive similar predictions.",
            "Depends on": "Model, Test Data",
            "k Neighbors": k,
            "Consistency Score": f"{res['value']:.4f}",
            "Formula": "Consistency = 1 - average(|y_hat_i - mean(y_hat_neighbors)|)",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def class_imbalance_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Class Imbalance.
    Measures the size imbalance between protected and unprotected groups.
    Ideal value: 0 (equal group sizes). |CI| < 0.1 is considered balanced.
    """
    try:
        res = core.class_imbalance(ctx.group_mask)
        score = 5 if res["balanced"] else 1
        return Result(score, {
            "Metric Description": "Measures imbalance in size between protected and unprotected groups.",
            "Depends on": "Dataset",
            "CI (manual)": f"{res['value']:.4f}",
            "N Protected": res["n_protected"],
            "N Unprotected": res["n_unprotected"],
            "Interpretation": "0 indicates perfect balance",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def kl_divergence_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    KL Divergence.
    Measures divergence between label distributions across groups.
    Ideal value: 0.
    """
    try:
        res = core.kl_divergence(ctx.y_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Divergence between label distributions across groups.",
            "Depends on": "Dataset",
            "KL Divergence": f"{res['value']:.6f}",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def smoothed_edf_score(ctx: EvaluationContext, thresholds: list, alpha: float = 1.0) -> Result:
    """
    Smoothed Empirical Differential Fairness (EDF).
    Log-ratio of smoothed mean predicted probabilities across groups.
    Ideal value: 0.
    """
    try:
        if ctx.y_prob_positive is None:
            return Result(np.nan, {"Info": "predict_proba is required for this metric."})
        group_values = ctx.test_data[ctx.protected_feature].to_numpy()
        res = core.smoothed_edf(ctx.y_prob_positive, group_values, alpha)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Smoothed Equality of Distributions Fairness.",
            "Depends on": "Model, Probabilistic Scores",
            "Alpha": alpha,
            "EDF Log-Ratio": f"{res['value']:.4f}",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def bias_amplification_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Bias Amplification.
    Measures whether the model amplifies the bias already present in the labels.
    Ideal value: 0 or negative (model should not amplify existing bias).
    """
    try:
        res = core.bias_amplification(ctx.y_test, ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Measures whether the model amplifies the bias present in the dataset.",
            "Bias Dataset": f"{res['bias_in_labels']:.4f}",
            "Bias Predictions": f"{res['bias_in_predictions']:.4f}",
            "Bias Amplification": f"{res['value']:.4f}",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def cohens_d_score(ctx: EvaluationContext, thresholds: list) -> Result:
    """
    Cohen's D.
    Standardised effect size of the prediction difference between groups.
    |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large.
    Ideal value: 0.
    """
    try:
        res = core.cohens_d(ctx.y_pred_test, ctx.group_mask)
        score = calculate_score(res["value"], thresholds)
        return Result(score, {
            "Metric Description": "Standardised effect size between group predictions.",
            "Cohen's D": f"{res['value']:.4f}",
            "Formula": "(mu1 - mu2) / sigma_pooled",
        })
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyse(context: EvaluationContext, config: dict) -> Result:
    """
    Run the full fairness analysis suite and return a structured Result.

    Parameters
    ----------
    context : EvaluationContext
        Holds the model, train/test DataFrames, feature arrays,
        predictions, probabilities, and factsheet.
    config  : dict
        Threshold configuration dict keyed by metric name
        (e.g. {"score_statistical_parity_difference": {"thresholds": {"value": [...]}}}).

    Returns
    -------
    Result
        .score      — dict mapping metric name to score in [1, 5] or np.nan
        .properties — dict mapping metric name to a detail dict
    """
    output = {
        "underfitting":                  underfitting_score(context, _thresh(config, "score_underfitting")),
        "overfitting":                   overfitting_score(context, _thresh(config, "score_overfitting")),
        "class_balance":                 class_balance_score(context),
        "statistical_parity_difference": stat_parity_score(context, _thresh(config, "score_statistical_parity_difference")),
        "disparate_impact":              disparate_impact_score(context, _thresh(config, "score_disparate_impact")),
        "equal_opportunity_difference":  eq_opp_score(context, _thresh(config, "score_equal_opportunity_difference")),
        "average_odds_difference":       avg_odds_score(context, _thresh(config, "score_average_odds_difference")),
        "accuracy_parity":               accuracy_parity_score(context, _thresh(config, "score_accuracy_parity")),
        "predictive_parity":             predictive_parity_score(context, _thresh(config, "score_predictive_parity")),
        "treatment_equality":            treatment_equality_score(context, _thresh(config, "score_treatment_equality")),
        "calibration":                   calibration_score(context, _thresh(config, "score_calibration"), n_bins=10),
        "well_calibration":              well_calibration_score(context, _thresh(config, "score_well_calibration"), n_bins=10),
        "generalized_entropy":           generalized_entropy_score(context, _thresh(config, "score_generalized_entropy"), alpha=2),
        "theil_index":                   theil_index_score(context, _thresh(config, "score_theil_index")),
        "coefficient_variation":         coefficient_variation_score(context, _thresh(config, "score_coefficient_variation")),
        "consistency":                   consistency_score(context, _thresh(config, "score_consistency")),
        "class_imbalance":               class_imbalance_score(context, _thresh(config, "score_class_imbalance")),
        "kl_divergence":                 kl_divergence_score(context, _thresh(config, "score_kl_divergence")),
        "smoothed_edf":                  smoothed_edf_score(context, _thresh(config, "score_smoothed_edf")),
        "bias_amplification":            bias_amplification_score(context, _thresh(config, "score_bias_amplification")),
        "cohens_d":                      cohens_d_score(context, _thresh(config, "score_cohens_d")),
    }

    scores     = {k: v.score for k, v in output.items()}
    properties = {k: v.properties for k, v in output.items()}

    return Result(score=scores, properties=properties)