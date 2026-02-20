# fairness/metrics.py

from trust_library.base_metric import BaseMetric
from . import fairness_metrics_core as core


# ─────────────────────────────────────────────────────────────
# Underfitting
# ─────────────────────────────────────────────────────────────

class UnderfittingMetric(BaseMetric):

    def __init__(self):
        super().__init__("underfitting", "score_underfitting")

    def compute(self, ctx):
        return core.underfitting(ctx.y_test, ctx.y_pred_test)

    def build_properties(self, raw):
        return {
            "Metric Description": "Compares the model's test accuracy against a baseline.",
            "Test Accuracy": f"{raw['test_accuracy']:.2%}",
        }


# ─────────────────────────────────────────────────────────────
# Overfitting
# ─────────────────────────────────────────────────────────────

class OverfittingMetric(BaseMetric):

    def __init__(self):
        super().__init__("overfitting", "score_overfitting")

    def compute(self, ctx):
        return core.overfitting(
            ctx.y_train,
            ctx.y_pred_train,
            ctx.y_test,
            ctx.y_pred_test,
        )

    def compute_score(self, raw, config):
        if raw["test_accuracy"] < 0.6:
            return None
        return super().compute_score(raw, config)

    def build_properties(self, raw):
        return {
            "Training Accuracy": f"{raw['train_accuracy']:.2%}",
            "Test Accuracy": f"{raw['test_accuracy']:.2%}",
            "Train-Test Gap": f"{raw['value']:.2%}",
        }


# ─────────────────────────────────────────────────────────────
# Statistical Parity
# ─────────────────────────────────────────────────────────────

class StatisticalParityMetric(BaseMetric):

    def __init__(self):
        super().__init__(
            "statistical_parity_difference",
            "score_statistical_parity_difference"
        )

    def compute(self, ctx):
        return core.statistical_parity_difference(
            ctx.y_pred_test,
            ctx.group_mask,
        )

    def build_properties(self, raw):
        return {
            "Favored Protected Ratio": f"{raw['favored_ratio_protected']:.2%}",
            "Favored Unprotected Ratio": f"{raw['favored_ratio_unprotected']:.2%}",
            "Statistical Parity Difference": f"{raw['value']:.4f}",
        }


# ─────────────────────────────────────────────────────────────
# Class Balance
# ─────────────────────────────────────────────────────────────

class ClassBalanceMetric(BaseMetric):

    def __init__(self):
        super().__init__("class_balance", None)

    def compute(self, ctx):
        return core.class_balance(
            ctx.train_data[ctx.target_column].to_numpy()
        )

    def custom_score(self, raw):
        return 5 if raw["balanced"] else 1

    def build_properties(self, raw):
        return {
            "P-Value": f"{raw['p_value']:.4f}",
            "Class Counts": raw["class_counts"],
        }


# ─────────────────────────────────────────────────────────────
# Disparate Impact
# ─────────────────────────────────────────────────────────────

class DisparateImpactMetric(BaseMetric):

    def __init__(self):
        super().__init__("disparate_impact", "score_disparate_impact")

    def compute(self, ctx):
        return core.disparate_impact(ctx.y_pred_test, ctx.group_mask)

    def build_properties(self, raw):
        return {
            "Metric Description": "Ratio of favorable prediction rates between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "Protected Favored Ratio": f"P(y_hat=favorable|protected=True) = {raw['favored_ratio_protected']:.2%}",
            "Unprotected Favored Ratio": f"P(y_hat=favorable|protected=False) = {raw['favored_ratio_unprotected']:.2%}",
            "Formula": "Protected Favored Ratio / Unprotected Favored Ratio",
            "Disparate Impact": f"{raw['value']:.4f}",
        }


# ─────────────────────────────────────────────────────────────
# Equal Opportunity Difference
# ─────────────────────────────────────────────────────────────

class EqualOpportunityMetric(BaseMetric):

    def __init__(self):
        super().__init__(
            "equal_opportunity_difference",
            "score_equal_opportunity_difference",
        )

    def compute(self, ctx):
        return core.equal_opportunity_difference(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Difference in true positive rates between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "TPR Unprotected Group": f"{raw['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{raw['tpr_protected']:.2%}",
            "Formula": "Equal Opportunity Difference = TPR Protected - TPR Unprotected",
            "Equal Opportunity Difference": f"{raw['value']*100:.2f}%",
        }


# ─────────────────────────────────────────────────────────────
# Average Odds Difference
# ─────────────────────────────────────────────────────────────

class AverageOddsMetric(BaseMetric):

    def __init__(self):
        super().__init__(
            "average_odds_difference",
            "score_average_odds_difference",
        )

    def compute(self, ctx):
        return core.average_odds_difference(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Average of the TPR and FPR differences between groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "FPR Unprotected Group": f"{raw['fpr_unprotected']:.2%}",
            "FPR Protected Group": f"{raw['fpr_protected']:.2%}",
            "TPR Unprotected Group": f"{raw['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{raw['tpr_protected']:.2%}",
            "Formula": "0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)",
            "Average Odds Difference": f"{raw['value']*100:.2f}%",
        }


# ─────────────────────────────────────────────────────────────
# Accuracy Parity
# ─────────────────────────────────────────────────────────────

class AccuracyParityMetric(BaseMetric):

    def __init__(self):
        super().__init__("accuracy_parity", "score_accuracy_parity")

    def compute(self, ctx):
        return core.accuracy_parity(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures whether prediction accuracy is equal across groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "Accuracy Unprotected Group": f"{raw['accuracy_unprotected']:.2%}",
            "Accuracy Protected Group": f"{raw['accuracy_protected']:.2%}",
            "Formula": "Accuracy Parity = Accuracy Protected - Accuracy Unprotected",
            "Accuracy Parity Difference": f"{raw['value']*100:.2f}%",
        }


# ─────────────────────────────────────────────────────────────
# Predictive Parity
# ─────────────────────────────────────────────────────────────

class PredictiveParityMetric(BaseMetric):

    def __init__(self):
        super().__init__("predictive_parity", "score_predictive_parity")

    def compute(self, ctx):
        return core.predictive_parity(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Checks equality of PPV and NPV between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "PPV Unprotected": f"{raw['ppv_unprotected']:.2%}",
            "PPV Protected": f"{raw['ppv_protected']:.2%}",
            "NPV Unprotected": f"{raw['npv_unprotected']:.2%}",
            "NPV Protected": f"{raw['npv_protected']:.2%}",
            "Formula": "0.5*(PPV Protected - PPV Unprotected) + 0.5*(NPV Protected - NPV Unprotected)",
            "Predictive Parity Difference": f"{raw['value']*100:.2f}%",
        }


# ─────────────────────────────────────────────────────────────
# Treatment Equality
# ─────────────────────────────────────────────────────────────

class TreatmentEqualityMetric(BaseMetric):

    def __init__(self):
        super().__init__("treatment_equality", "score_treatment_equality")

    def compute(self, ctx):
        return core.treatment_equality(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures equality of the false negative / false positive ratio across groups.",
            "Depends on": "Model, Test Data, Factsheet",
            "FN Unprotected": raw["fn_unprotected"],
            "FP Unprotected": raw["fp_unprotected"],
            "FN Protected":   raw["fn_protected"],
            "FP Protected":   raw["fp_protected"],
            "FN/FP Unprotected": f"{raw['fn_fp_ratio_unprotected']:.4f}",
            "FN/FP Protected":   f"{raw['fn_fp_ratio_protected']:.4f}",
            "Formula": "(FN/FP)_Protected - (FN/FP)_Unprotected",
            "Treatment Equality Difference": f"{raw['value']:.4f}",
        }


# ─────────────────────────────────────────────────────────────
# Calibration Gap
# ─────────────────────────────────────────────────────────────

class CalibrationGapMetric(BaseMetric):

    def __init__(self, n_bins: int = 10):
        super().__init__("calibration_gap", "score_calibration_gap")
        self.n_bins = n_bins

    def compute(self, ctx):
        if ctx.y_prob_positive is None:
            raise ValueError("predict_proba is required for this metric.")
        return core.calibration_gap(
            ctx.y_test, ctx.y_prob_positive, ctx.group_mask, self.n_bins
        )

    def build_properties(self, raw):
        return {
            "Metric Description": (
                "Checks whether individuals with the same predicted score have the same "
                "empirical outcome probability in both groups."
            ),
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Gap": f"{raw['value']:.4f}",
            "Bins Used": self.n_bins,
        }


# ─────────────────────────────────────────────────────────────
# Well-Calibration Error
# ─────────────────────────────────────────────────────────────

class WellCalibrationMetric(BaseMetric):

    def __init__(self, n_bins: int = 10):
        super().__init__("well_calibration_error", "score_well_calibration_error")
        self.n_bins = n_bins

    def compute(self, ctx):
        if ctx.y_prob_positive is None:
            raise ValueError("predict_proba is required for this metric.")
        return core.well_calibration_error(
            ctx.y_test, ctx.y_prob_positive, self.n_bins
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Checks whether predicted probabilities match empirical outcome frequencies.",
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Error": f"{raw['value']:.4f}",
            "Bins Used": self.n_bins,
        }


# ─────────────────────────────────────────────────────────────
# Generalized Entropy Index
# ─────────────────────────────────────────────────────────────

class GeneralizedEntropyMetric(BaseMetric):

    def __init__(self, alpha: float = 2):
        super().__init__(
            "generalized_entropy_index",
            "score_generalized_entropy_index",
        )
        self.alpha = alpha

    def compute(self, ctx):
        return core.generalized_entropy_index(
            ctx.y_test, ctx.y_pred_test, self.alpha
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures inequality in the distribution of correct predictions.",
            "Depends on": "Model, Test Data",
            "Alpha": self.alpha,
            "Benefit Definition": "b_i = 1 if prediction is correct, 0 otherwise",
            "Generalized Entropy Index": f"{raw['value']:.6f}",
            "Conclusion": (
                "Perfect equality" if raw["value"] == 0
                else "Prediction benefits unevenly distributed"
            ),
        }


# ─────────────────────────────────────────────────────────────
# Theil Index  (GEI with alpha=1)
# ─────────────────────────────────────────────────────────────

class TheilIndexMetric(GeneralizedEntropyMetric):

    def __init__(self):
        super().__init__(alpha=1)
        # Override the config / score keys to match the original naming
        self.metric_key = "theil_index"
        self.score_key  = "score_theil_index"


# ─────────────────────────────────────────────────────────────
# Coefficient of Variation
# ─────────────────────────────────────────────────────────────

class CoefficientVariationMetric(BaseMetric):

    def __init__(self):
        super().__init__(
            "coefficient_of_variation",
            "score_coefficient_of_variation",
        )

    def compute(self, ctx):
        return core.coefficient_of_variation(ctx.y_test, ctx.y_pred_test)

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures the relative dispersion of prediction benefits.",
            "Depends on": "Model, Test Data",
            "Coefficient of Variation": f"{raw['value']:.6f}",
            "Formula": "CV = sqrt(2 * GEI(alpha=2))",
        }


# ─────────────────────────────────────────────────────────────
# Individual Consistency
# ─────────────────────────────────────────────────────────────

class ConsistencyMetric(BaseMetric):

    def __init__(self, k: int = 5):
        super().__init__("individual_consistency", "score_individual_consistency")
        self.k = k

    def compute(self, ctx):
        return core.individual_consistency(ctx.X_test, ctx.y_pred_test, self.k)

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures whether similar individuals receive similar predictions.",
            "Depends on": "Model, Test Data",
            "k Neighbors": self.k,
            "Consistency Score": f"{raw['value']:.4f}",
            "Formula": "Consistency = 1 - average(|y_hat_i - mean(y_hat_neighbors)|)",
        }


# ─────────────────────────────────────────────────────────────
# Class Imbalance
# ─────────────────────────────────────────────────────────────

class ClassImbalanceMetric(BaseMetric):

    def __init__(self):
        super().__init__("class_imbalance", None)

    def compute(self, ctx):
        return core.class_imbalance(ctx.group_mask)

    def custom_score(self, raw):
        return 5 if raw["balanced"] else 1

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures imbalance in size between protected and unprotected groups.",
            "Depends on": "Dataset",
            "CI (manual)": f"{raw['value']:.4f}",
            "N Protected": raw["n_protected"],
            "N Unprotected": raw["n_unprotected"],
            "Interpretation": "0 indicates perfect balance",
        }


# ─────────────────────────────────────────────────────────────
# KL Divergence
# ─────────────────────────────────────────────────────────────

class KLDivergenceMetric(BaseMetric):

    def __init__(self):
        super().__init__("kl_divergence", "score_kl_divergence")

    def compute(self, ctx):
        return core.kl_divergence(ctx.y_test, ctx.group_mask)

    def build_properties(self, raw):
        return {
            "Metric Description": "Divergence between label distributions across groups.",
            "Depends on": "Dataset",
            "KL Divergence": f"{raw['value']:.6f}",
        }


# ─────────────────────────────────────────────────────────────
# Smoothed EDF
# ─────────────────────────────────────────────────────────────

class SmoothedEDFMetric(BaseMetric):

    def __init__(self, alpha: float = 1.0):
        super().__init__("smoothed_edf", "score_smoothed_edf")
        self.alpha = alpha

    def compute(self, ctx):
        if ctx.y_prob_positive is None:
            raise ValueError("predict_proba is required for this metric.")
        group_values = ctx.test_data[ctx.protected_feature].to_numpy()
        return core.smoothed_edf(ctx.y_prob_positive, group_values, self.alpha)

    def build_properties(self, raw):
        return {
            "Metric Description": "Smoothed Equality of Distributions Fairness.",
            "Depends on": "Model, Probabilistic Scores",
            "Alpha": self.alpha,
            "EDF Log-Ratio": f"{raw['value']:.4f}",
        }


# ─────────────────────────────────────────────────────────────
# Bias Amplification
# ─────────────────────────────────────────────────────────────

class BiasAmplificationMetric(BaseMetric):

    def __init__(self):
        super().__init__("bias_amplification", "score_bias_amplification")

    def compute(self, ctx):
        return core.bias_amplification(
            ctx.y_test, ctx.y_pred_test, ctx.group_mask
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Measures whether the model amplifies the bias present in the dataset.",
            "Depends on": "Model, Test Data, Factsheet",
            "Bias Dataset": f"{raw['bias_in_labels']:.4f}",
            "Bias Predictions": f"{raw['bias_in_predictions']:.4f}",
            "Bias Amplification": f"{raw['value']:.4f}",
        }


# ─────────────────────────────────────────────────────────────
# Cohen's D
# ─────────────────────────────────────────────────────────────

class CohensDMetric(BaseMetric):

    def __init__(self):
        super().__init__("cohens_d", "score_cohens_d")

    def compute(self, ctx):
        return core.cohens_d(ctx.y_pred_test, ctx.group_mask)

    def build_properties(self, raw):
        return {
            "Metric Description": "Standardised effect size between group predictions.",
            "Depends on": "Model, Test Data, Factsheet",
            "Cohen's D": f"{raw['value']:.4f}",
            "Formula": "(mu1 - mu2) / sigma_pooled",
        }