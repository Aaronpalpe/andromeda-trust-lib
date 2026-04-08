from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from trust_library.base_metric import BaseMetric
from . import privacy_metrics_core as core


# =============================================================================
# Epsilon DP Leakage
# =============================================================================

class EpsilonMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("epsilon_dp", "score_epsilon_dp")

    def compute(self, ctx) -> Dict[str, float]:

        epsilon = (
            ctx.factsheet
            .get("privacy", {})
            .get("epsilon", {})
            .get("value", None)
        )

        return core.epsilon_dp(
            epsilon=epsilon,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Theoretical epsilon value from the factsheet's differential privacy analysis. Lower epsilon implies stronger privacy guarantees (less theoretical leakage).",
            "Depends on": "Training Mechanism",
            "Formula": "Epsilon-DP = epsilon (as declared in factsheet privacy analysis)",
            "Epsilon": f"{raw['value']:.6f}",
        }

# =============================================================================
# Epsilon Star (Empirical DP Leakage)
# =============================================================================

class EpsilonStarMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("epsilon_star", "score_epsilon_star")

    def compute(self, ctx) -> Dict[str, float]:

        return core.epsilon_star(
            model=ctx.model,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Empirical epsilon* estimated from the loss distribution. Lower is better (less membership leakage).",
            "Depends on": "Model and Train/Test Data",
            "Formula": "epsilon* = log(max over threshold-based membership ratios)",
            "Delta Used": f"{raw['delta']:.2e}",
            "Epsilon*": f"{raw['value']:.6f}",
        }


# =============================================================================
# SHAPr Membership Risk
# =============================================================================

class SHAPRMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("shapr", "score_shapr")

    def compute(self, ctx) -> Dict[str, float]:

        return core.shapr(
            model=ctx.model,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Approximate membership inference risk estimated with SHAPr. Lower values indicate lower risk.",
            "Depends on": "Model and Train/Test Data",
            "Formula": "SHAPr Risk = mean marginal contribution in prediction-space nearest-neighbor analysis",
            "Average Marginal Contribution": f"{raw['value']:.6f}",
            "Sample Size": raw["sample_size"],
            "k Neighbors": raw["k_neighbors"],
        }


# =============================================================================
# Attribute Inference
# =============================================================================

class AttributeInferenceMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("attribute_inference", "score_attribute_inference")

    def compute(self, ctx) -> Dict[str, float]:

        sensitive: List[str] = (
            ctx.factsheet
            .get("privacy", {})
            .get("sensitive_attribute", {})
            .get("value", [])
        )

        if not sensitive:
            raise ValueError("Sensitive attribute is required for attribute_inference metric. Please provide it in the factsheet under privacy.sensitive_attribute.")

        sensitive_attr = sensitive[0]

        return core.attribute_inference(
            X_train=ctx.X_train,
            X_test=ctx.X_test,
            y_train=ctx.y_train,
            y_test=ctx.y_test,
            sensitive_attribute=sensitive_attr,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Attribute inference risk for the declared sensitive attribute.",
            "Depends on": "Model, Train/Test Data, and Sensitive Attribute",
            "Formula": "Attribute Inference = attack score predicting sensitive attribute from non-sensitive features and labels",
            "Sensitive Attribute": raw["sensitive"],
            "Risk Score": (
                f"{raw['value']:.6f}" if raw["value"] is not None else "N/A"
            ),
        }

# =============================================================================
# Membership Privacy Risk
# =============================================================================

class PrivacyRiskMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("privacy_risk", "score_privacy_risk")

    def compute(self, ctx) -> Dict[str, float]:

        return core.privacy_risk(
            y_prob_train=ctx.y_prob_train,
            y_train=ctx.y_train,
            y_prob_test=ctx.y_prob_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Membership inference privacy risk derived from predicted probabilities.",
            "Depends on": "Model and Predicted Probabilities",
            "Formula": "Privacy Risk = mean posterior membership risk score",
            "Mean Privacy Risk": f"{raw['value']:.6f}",
        }


# =============================================================================
# Accuracy Ratio (Data Minimization)
# =============================================================================

class AccuracyRatioMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("accuracy_ratio", "score_accuracy_ratio")

    def compute(self, ctx) -> Dict[str, float]:

        return core.accuracy_ratio(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            y_pred_test=ctx.y_pred_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Accuracy ratio between the original model and the data-minimized variant (original/noisy). Values close to 1 indicate minimal performance loss; larger values indicate greater utility loss.",
            "Depends on": "Model and Test Data",
            "Formula": "Accuracy Ratio = Accuracy(original test) / Accuracy(noisy test)",
            "selector_type": f"{raw['args']['selector_type']}",
            "modifier_type": f"{raw['args']['modifier_type']}",
            "n_features" : raw["args"]["n_feats"],
            "Accuracy Ratio": f"{raw['value']:.6f}",
        }


# =============================================================================
# k-Anonymity
# =============================================================================

class KAnonymityMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("k_anonymity", "score_k_anonymity")

    def compute(self, ctx) -> Dict[str, Any]:

        quasi = (
            ctx.factsheet
            .get("privacy", {})
            .get("quasi_identifiers", {})
            .get("value", [])
        )

        df = pd.concat([ctx.train_data, ctx.test_data], ignore_index=True)

        return core.k_anonymity(
            df=df,
            quasi_identifiers=quasi,
        )

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "Minimum k value achieved for the quasi-identifier groups. Higher values indicate better privacy protection.",
            "Depends on": "Quasi Identifiers and Data",
            "Formula": "k-Anonymity = min over equivalence classes of group size",
            "Quasi Identifiers": raw["quasi_identifiers"],
            "Minimum k": raw["value"],
        }


# =============================================================================
# l-Diversity
# =============================================================================

class LDiversityMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("l_diversity", "score_l_diversity")

    def compute(self, ctx) -> Dict[str, Any]:

        quasi = (
            ctx.factsheet
            .get("privacy", {})
            .get("quasi_identifiers", {})
            .get("value", [])
        )

        sensitive = (
            ctx.factsheet
            .get("privacy", {})
            .get("sensitive_attribute", {})
            .get("value", [])
        )

        df = pd.concat([ctx.train_data, ctx.test_data], ignore_index=True)

        return core.l_diversity(
            df=df,
            quasi_identifiers=quasi,
            sensitive_attributes=sensitive,
        )

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "Minimum l value achieved for the quasi-identifier groups and sensitive attributes.",
            "Depends on": "Quasi Identifiers, Sensitive Attributes, and Data",
            "Formula": "l-Diversity = min over quasi-identifier groups of distinct sensitive values",
            "Quasi Identifiers": raw["quasi_identifiers"],
            "Sensitive Attributes": raw["sensitive_attributes"],
            "Minimum l": raw["value"],
        }


# =============================================================================
# t-Closeness
# =============================================================================

class TClosenessMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("t_closeness", "score_t_closeness")

    def compute(self, ctx) -> Dict[str, float]:

        quasi = (
            ctx.factsheet
            .get("privacy", {})
            .get("quasi_identifiers", {})
            .get("value", [])
        )

        sensitive = (
            ctx.factsheet
            .get("privacy", {})
            .get("sensitive_attribute", {})
            .get("value", [])
        )

        df = pd.concat([ctx.train_data, ctx.test_data], ignore_index=True)

        return core.t_closeness(
            df=df,
            quasi_identifiers=quasi,
            sensitive_attributes=sensitive,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Maximum t value measuring distribution closeness within quasi-identifier groups.",
            "Depends on": "Quasi Identifiers, Sensitive Attributes, and Data",
            "Formula": "t-Closeness = max distance between group-sensitive distribution and global-sensitive distribution",
            "Quasi Identifiers": raw["quasi_identifiers"],
            "Sensitive Attributes": raw["sensitive_attributes"],
            "Maximum t": raw["value"],
        }