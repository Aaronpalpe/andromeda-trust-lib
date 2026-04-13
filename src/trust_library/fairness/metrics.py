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
            "Metric Description": "Test accuracy as a proxy for underfitting.",
            "Depends on": "Model, Test Data",
            "Formula": "Underfitting = Accuracy(y_test, y_pred_test)",
            "Test Accuracy": f"{raw['value']:.2%}",
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
            return None # Don't penalize overfitting if the model is performing poorly on the test set (i.e. likely underfitting)
        return super().compute_score(raw, config)

    def build_properties(self, raw):
        return {
            "Metric Description": "Train-test accuracy gap as a proxy for overfitting.",
            "Depends on": "Model, Training Data, Test Data",
            "Formula": "Overfitting Gap = Accuracy(y_train, y_pred_train) - Accuracy(y_test, y_pred_test)",
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
            "Metric Description": "Difference in favorable prediction rates between protected and unprotected groups.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "Formula": "Statistical Parity Difference = Favored Ratio Protected - Favored Ratio Unprotected",
            "Number Protected": raw["n_protected"],
            "Number Protected Favored": raw["n_protected_favored"],
            "Favored Protected Ratio": f"{raw['favored_ratio_protected']:.2%}",
            "Number Unprotected": raw["n_unprotected"],
            "Number Unprotected Favored": raw["n_unprotected_favored"],
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
            "Metric Description": "Chi-square test for class balance in label distribution.",
            "Depends on": "Training Data",
            "P-Value": f"{raw['p_value']:.4f}",
            "Balanced": f"{raw['balanced']}",
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
            "Formula": "Protected Favored Ratio / Unprotected Favored Ratio",
            "Number Protected": raw["n_protected"],
            "Number Protected Favored": raw["n_protected_favored"],
            "Protected Favored Ratio": f"P(y_hat=favorable|protected=True) = {raw['favored_ratio_protected']:.2%}",
            "Number Unprotected": raw["n_unprotected"],
            "Number Unprotected Favored": raw["n_unprotected_favored"],
            "Unprotected Favored Ratio": f"P(y_hat=favorable|protected=False) = {raw['favored_ratio_unprotected']:.2%}",
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
            "Formula": "Equal Opportunity Difference = TPR Protected - TPR Unprotected",
            "TPR Unprotected Group": f"{raw['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{raw['tpr_protected']:.2%}",
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
            "Formula": "0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)",
            "FPR Unprotected Group": f"{raw['fpr_unprotected']:.2%}",
            "FPR Protected Group": f"{raw['fpr_protected']:.2%}",
            "TPR Unprotected Group": f"{raw['tpr_unprotected']:.2%}",
            "TPR Protected Group": f"{raw['tpr_protected']:.2%}",
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
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "Accuracy Parity = Accuracy Protected - Accuracy Unprotected",
            "Accuracy Unprotected Group": f"{raw['accuracy_unprotected']:.2%}",
            "Accuracy Protected Group": f"{raw['accuracy_protected']:.2%}",
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
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)",
            "Formula": "0.5*(PPV Protected - PPV Unprotected) + 0.5*(NPV Protected - NPV Unprotected)",
            "PPV Unprotected": f"{raw['ppv_unprotected']:.2%}",
            "PPV Protected": f"{raw['ppv_protected']:.2%}",
            "NPV Unprotected": f"{raw['npv_unprotected']:.2%}",
            "NPV Protected": f"{raw['npv_protected']:.2%}",
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
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "(FN/FP)_Protected - (FN/FP)_Unprotected",
            "FN Unprotected": raw["fn_unprotected"],
            "FP Unprotected": raw["fp_unprotected"],
            "FN Protected":   raw["fn_protected"],
            "FP Protected":   raw["fp_protected"],
            "FN/FP Unprotected": f"{raw['fn_fp_ratio_unprotected']:.4f}",
            "FN/FP Protected":   f"{raw['fn_fp_ratio_protected']:.4f}",
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
            "Depends on": "Model, Test Data, Probabilistic Scores, Factsheet (Definition of Protected Group)",
            "Formula": "Calibration Gap = Mean_b(|P(y=1|score bin b, protected) - P(y=1|score bin b, unprotected)|)",
            "Bins Used": f"{raw['n_bins']}",
            "Calibration by Bin": raw.get("bins"),
            "Mean Calibration Gap": f"{raw['value']:.4f}",
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
            "Formula": "Well Calibration Error = Mean_b(|Mean(y in b) - Mean(score in b)|)",
            "Bins Used": f"{raw['n_bins']}",
            "Difference by Bin": raw.get("bins-scores"),
            "Mean Calibration Error": f"{raw['value']:.4f}",
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
            "Formula": "GEI(alpha) = Mean(((b/mu)^alpha - 1) / (alpha * (alpha - 1))), where b is individual benefit",
            "Alpha": f"{raw['alpha']:.6f}",
            "Mean Benefit": f"{raw['mean_benefit']:.6f}",
            "Generalized Entropy Index": f"{raw['value']:.6f}",
        }


# ─────────────────────────────────────────────────────────────
# Theil Index  (GEI with alpha=1)
# ─────────────────────────────────────────────────────────────

class TheilIndexMetric(GeneralizedEntropyMetric):

    def __init__(self):
        super().__init__(alpha=1)
        self.metric_key = "theil_index"
        self.score_config_key = "score_theil_index"


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
            "Formula": "CV = sqrt(2 * GEI(alpha=2))",
            "Alpha": "2",
            "GEI (alpha=2)": f"{raw['gei_alpha2']:.6f}",
            "Coefficient of Variation": f"{raw['value']:.6f}",
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
            "Formula": "Consistency = 1 - average(|y_hat_i - mean(y_hat_neighbors)|)",
            "k Neighbors": f"{raw['k']}",
            "Consistency Score": f"{raw['value']:.4f}",
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
            "Metric Description": "Measures imbalance in size between protected and unprotected groups. 0 indicates perfect balance.",
            "Depends on": "Dataset, Factsheet (Definition of Protected Group)",
            "Formula": "Class Imbalance = (N_unprotected - N_protected) / (N_unprotected + N_protected)",
            "N Protected": f"{raw['n_protected']}",
            "N Unprotected": f"{raw['n_unprotected']}",
            "Class Imbalance": f"{raw['value']:.4f}",
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
            "Depends on": "Dataset, Factsheet (Definition of Protected Group)",
            "Formula": "KL(P_unprotected || P_protected)",
            "KL Divergence": f"{raw['value']:.6f}",
        }

# ─────────────────────────────────────────────────────────────
# Conditional Demographic Disparity
# ─────────────────────────────────────────────────────────────

class ConditionalDemographicDisparityMetric(BaseMetric):

    def __init__(self):
        super().__init__(
            "conditional_demographic_disparity", 
            "score_conditional_demographic_disparity"
        )

    def compute(self, ctx):
        return core.conditional_demographic_disparity(
            ctx.y_pred_test, 
            ctx.group_mask
        )

    def build_properties(self, raw):
        props = {
            "Metric Description": "Conditional Demographic Disparity (Wachter et al., 2021).",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "CDD = (1/N) * sum_i(N_i * DD_i), DD_i = (N_i,negative/N_negative) - (N_i,positive/N_positive)",
        }
        
        if "dd_protected" in raw and "dd_unprotected" in raw:
            props["DD Protected Group"] = f"{raw['dd_protected']:.4f}"
            props["DD Unprotected Group"] = f"{raw['dd_unprotected']:.4f}"
            
        if "n_protected" in raw:
            props["N Protected"] = f"{raw['n_protected']}"
        if "n_unprotected" in raw:
            props["N Unprotected"] = f"{raw['n_unprotected']}"

        if "total_positive" in raw and "total_negative" in raw:
            props["Total Positive"] = f"{raw['total_positive']}"
            props["Total Negative"] = f"{raw['total_negative']}"

        props["Conditional Demographic Disparity"] = f"{raw['value']:.4f}"
            
        return props

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
            "Metric Description": "Smoothed Empirical Differential Fairness (EDF).",
            "Depends on": "Model, Test Data, Probabilistic Scores, Factsheet (Definition of Protected Group)",
            "Formula": "EDF = max_i,j max(|log((p_i+alpha)/(p_j+alpha))|, |log((1-p_i+alpha)/(1-p_j+alpha))|)",
            "Alpha": f"{raw['alpha']:.6f}",
            "Group Smoothed Selection Rates": f"{raw['group_smoothed_selection_rates']}",
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
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "Bias Amplification = EDF(predictions) - EDF(labels)",
            "Bias Dataset": f"{raw['bias_in_labels']:.4f}",
            "Bias Predictions": f"{raw['bias_in_predictions']:.4f}",
            "Bias Amplification": f"{raw['value']:.4f}",
        }

# ─────────────────────────────────────────────────────────────
# Between-Group Generalized Entropy Error
# ─────────────────────────────────────────────────────────────

class BetweenGroupGeneralizedEntropyMetric(BaseMetric):

    def __init__(self, alpha: float = 2):
        super().__init__(
            "between_group_generalized_entropy_error", 
            "score_between_group_generalized_entropy_error"
        )
        self.alpha = alpha

    def compute(self, ctx):
        return core.between_group_generalized_entropy_error(
            ctx.y_test, 
            ctx.y_pred_test, 
            ctx.group_mask, 
            self.alpha
        )

    def build_properties(self, raw):
        return {
            "Metric Description": "Between-Group Generalized Entropy Error (Speicher et al., 2018).",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "Between-group GEI computes GEI over group-level mean benefits.",
            "Alpha": f"{raw['alpha']:.6f}",
            "Mean Benefit": f"{raw['mean_benefit']:.6f}",
            "Between-Group GEI Error": f"{raw['value']:.6f}",
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
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "(mu1 - mu2) / sigma_pooled",
            "Mean Unprotected": f"{raw['mean_unprotected']:.4f}",
            "Mean Protected": f"{raw['mean_protected']:.4f}",
            "Pooled Std Dev": f"{raw['pooled_std']:.4f}",
            "Cohen's D": f"{raw['value']:.4f}",
        }
    

# ─────────────────────────────────────────────────────────────
# Z-Test Difference (2-SD Rule)
# ─────────────────────────────────────────────────────────────

class ZTestDiffMetric(BaseMetric):

    def __init__(self):
        super().__init__("z_test_diff", "score_z_test_diff")

    def compute(self, ctx):
        return core.z_test_diff(ctx.y_pred_test, ctx.group_mask)

    def build_properties(self, raw):
        return {
            "Metric Description": "Z test (difference) / 2-SD statistic for success rates. Fair if the value stays between -2 and 2.",
            "Depends on": "Model, Test Data, Factsheet (Definition of Protected Group)",
            "Formula": "Z = (SR_protected - SR_unprotected) / sqrt((SR_total * (1 - SR_total)) / (N_total * P_protected * (1 - P_protected)))",
            "Success Rate Protected": f"{raw['sr_protected']:.2%}",
            "Success Rate Unprotected": f"{raw['sr_unprotected']:.2%}",
            "Total Success Rate": f"{raw['total_success_rate']:.2%}",
            "N Protected": f"{raw['n_protected']}",
            "N Unprotected": f"{raw['n_unprotected']}",
            "Z-Test Score": f"{raw['value']:.4f}",
        }