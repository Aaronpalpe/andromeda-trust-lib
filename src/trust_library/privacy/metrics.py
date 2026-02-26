from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from trust_library.base_metric import BaseMetric
from . import privacy_metrics_core as core


# =============================================================================
# Epsilon Star (Empirical DP Leakage)
# =============================================================================

class EpsilonStarMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("epsilon_star", "score_epsilon_star")

    def compute(self, ctx) -> Dict[str, float]:

        return core.compute_epsilon_star(
            model=ctx.model,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Empirical epsilon* from loss distribution.",
            "Delta Used": f"{raw['delta']:.2e}",
            "Epsilon*": f"{raw['value']:.6f}",
            "Interpretation": "Lower is better (less membership leakage).",
        }


# =============================================================================
# SHAPr Membership Risk
# =============================================================================

class SHAPRMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("shapr", "score_shapr")

    def compute(self, ctx) -> Dict[str, float]:

        return core.compute_shapr(
            model=ctx.model,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Approximate SHAPr membership risk.",
            "Average Marginal Contribution": f"{raw['value']:.6f}",
            "Sample Size": raw["sample_size"],
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
            return {"value": float("nan"), "sensitive": None}

        sensitive_attr = sensitive[0]

        return core.compute_attribute_inference(
            X_train=ctx.X_train,
            X_test=ctx.X_test,
            y_train=ctx.y_train,
            y_test=ctx.y_test,
            sensitive_attribute=sensitive_attr,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Attribute inference risk.",
            "Sensitive Attribute": raw["sensitive"],
            "Risk Score (accuracy)": (
                f"{raw['value']:.6f}" if raw["value"] is not None else "N/A"
            ),
        }

# =============================================================================
# Accuracy Ratio (Data Minimization)
# =============================================================================

class AccuracyRatioMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("accuracy_ratio", "score_accuracy_ratio")

    def compute(self, ctx) -> Dict[str, float]:

        return core.compute_accuracy_ratio(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            y_pred_test=ctx.y_pred_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Accuracy ratio for data minimization.",
            "Accuracy Ratio": f"{raw['value']:.6f}",
            "Interpretation": "Values close to 1 indicate minimal performance loss.",
        }

# =============================================================================
# Membership Privacy Risk
# =============================================================================

class PrivacyRiskMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("privacy_risk", "score_privacy_risk")

    def compute(self, ctx) -> Dict[str, float]:

        return core.compute_privacy_risk(
            y_prob_train=ctx.y_prob_train,
            y_train=ctx.y_train,
            y_prob_test=ctx.y_prob_test,
            y_test=ctx.y_test,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "Membership inference privacy risk.",
            "Mean Privacy Risk": f"{raw['value']:.6f}",
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

        return core.compute_k_anonymity(
            df=df,
            quasi_identifiers=quasi,
        )

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "k-Anonymity.",
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

        return core.compute_l_diversity(
            df=df,
            quasi_identifiers=quasi,
            sensitive_attributes=sensitive,
        )

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "l-Diversity.",
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

        return core.compute_t_closeness(
            df=df,
            quasi_identifiers=quasi,
            sensitive_attributes=sensitive,
        )

    def build_properties(self, raw: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Metric Description": "t-Closeness.",
            "Quasi Identifiers": raw["quasi_identifiers"],
            "Sensitive Attributes": raw["sensitive_attributes"],
            "Maximum t": raw["value"],
        }