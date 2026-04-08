from __future__ import annotations

from typing import Dict, Any

from trust_library.base_metric import BaseMetric
from . import accountability_metrics_core as core


# =============================================================================
# Train/Test Split
# =============================================================================

class TrainTestSplitMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("train_test_split", "score_train_test_split")

    def compute(self, ctx) -> Dict[str, int]:
        return core.train_test_split_ratio(
            ctx.train_data,
            ctx.test_data,
        )

    def compute_score(self, raw: Dict[str, int], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key, {})
            .get("mappings", {})
            .get("value", {})
        )

        return core.train_test_split_mapping(
            raw["train_ratio"],
            mappings,
        )

    def build_properties(self, raw: Dict[str, int]) -> Dict[str, Any]:
        return {
            "Metric Description": "Ratio between the training and test set sizes.",
            "Depends on": "Training and Testing Data",
            "Formula": "Train Ratio = round(100 * N_train / (N_train + N_test)); Test Ratio = 100 - Train Ratio",
            "Train/Test split": f"{raw['train_ratio']}/{raw['test_ratio']}",
        }


# =============================================================================
# Missing Data
# =============================================================================

class MissingDataMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("missing_data", "score_missing_data")

    def compute(self, ctx) -> Dict[str, int]:
        return core.count_missing_values(
            ctx.train_data,
            ctx.test_data,
        )

    def compute_score(self, raw: Dict[str, Any], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key, {})
            .get("mappings", {})
            .get("value", {})
        )
        missing = 'null_values_exist' if raw["value"] > 0 else 'no_null_values'

        return mappings.get(missing)
    
    def build_properties(self, raw: Dict[str, int]) -> Dict[str, Any]:
        return {
            "Metric Description": "Number of missing values across the training and test data.",
            "Depends on": "Training and Test Data",
            "Formula": "Missing Data = Missing(train) + Missing(test)",
            "Train missing values": raw.get("missing_train", 0),
            "Test missing values": raw.get("missing_test", 0),
            "Null values count": raw["value"],
        }


# =============================================================================
# Normalization
# =============================================================================

class NormalizationMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("normalization", "score_normalization")

    def compute(self, ctx) -> Dict[str, Any]:

        stats = core.normalization_statistics(
            ctx.X_train,
            ctx.X_test,
        )

        norm_type = core.detect_normalization_type(
            stats,
            ctx.X_train,
            ctx.X_test,
        )

        return {
            "value": norm_type,
            **stats,
        }

    def compute_score(self, raw: Dict[str, Any], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key, {})
            .get("mappings", {})
            .get("value", {})
        )

        return mappings.get(raw["value"])

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "Normalization strategy inferred from training and test data statistics.",
            "Depends on": "Training and Testing Data",
            "Formula": "Heuristic classification based on train/test mean-std closeness to (0,1) and value ranges in [0,1]",
            "Training mean": f"{raw['train_mean']:.4f}",
            "Training std": f"{raw['train_std']:.4f}",
            "Test mean": f"{raw['test_mean']:.4f}",
            "Test std": f"{raw['test_std']:.4f}",
            "Detected Strategy": raw["value"],
        }


# =============================================================================
# Regularization
# =============================================================================

class RegularizationMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("regularization", "score_regularization")

    def compute(self, ctx) -> Dict[str, Any]:

        reg = ctx.factsheet.get(
            "general", {}
        ).get("regularization", {}).get("value", None)

        return {"regularization": reg}
    
    def compute_score(self, raw: Dict[str, int], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key, {})
            .get("mappings", {})
            .get("value", {})
        )

        return core.regularization_mapping(
            raw["regularization"],
            mappings,
        )

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "Regularization technique declared in the factsheet.",
            "Depends on": "Factsheet",
            "Regularization technique": (
                raw["regularization"]
                if raw["regularization"] is not None
                else "Not specified"
            ),
        }


# =============================================================================
# Factsheet Completeness
# =============================================================================

class FactsheetCompletenessMetric(BaseMetric):

    def __init__(self) -> None:
        super().__init__("factsheet_completeness", None)

    def compute(self, ctx) -> Dict[str, Any]:
        return core.factsheet_completeness(ctx.factsheet)

    def custom_score(self, raw: Dict[str, Any]) -> float:
        return round(raw["ratio"] * 5, 2)

    def build_properties(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Metric Description": "Completeness of the factsheet based on present versus expected fields.",
            "Depends on": "Factsheet",
            "Formula": "Completeness Ratio = Present Fields / Total Expected Fields",
            "Fields present": f"{raw['present']}/{raw['total']}",
            "Missing fields": raw["missing"] if raw["missing"] else "None",
            "Completeness": f"{raw['ratio']:.2%}",
        }