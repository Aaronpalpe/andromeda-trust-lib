# trust_library/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
from trust_library.utils import Result, EvaluationContext, calculate_weighted_score
from typing import Any, List, Tuple
from .base_metric import BaseMetric


class Pillar(ABC):
    """
    Base class for all evaluation pillars.
    Each pillar only implements `get_metrics() and prepare() (if necessary)`.
    `score()` is generic and delegates to `analyse()`.
    """

    @property
    @abstractmethod
    def pillar_key(self) -> str:
        """Pillar key in the config (e.g., 'fairness', 'privacy')."""
        ...

    # Optional hook to modify EvaluationContext
    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        return

    # Method to obtain metrics from each pillar
    @abstractmethod
    def get_metrics(self) -> List[BaseMetric]:
        ...

    def analyse(self, context: EvaluationContext, config: dict[str, dict]) -> Result:
        self.prepare(context, config)
        metrics = self.get_metrics()

        scores: dict[str, float] = {}
        properties: dict[str, dict[str, Any]] = {}

        for metric in metrics:
            #start_time = time.time()
            result = metric.evaluate(context, config)
            scores[metric.metric_key] = result.score
            properties[metric.metric_key] = result.properties
            #elapsed_time = time.time() - start_time
            #print(f"Metric '{metric.metric_key}' computed in {elapsed_time:.2f} seconds.")

        return Result(score=scores, properties=properties)

    def score(self, context: EvaluationContext, config: dict) -> Tuple[float, Result]: #, dict[str, Any]]:
        """Computes the aggregated weighted score of the pillar (0–5)."""
        result = self.analyse(context, config.get("mappings", {}).get(self.pillar_key))
        weights = config.get("weights", {}).get(self.pillar_key, {})
        return (calculate_weighted_score(result.score, weights), result)
    #     notions_cfg = config.get("notions", {}).get(self.pillar_key)

    #     if not notions_cfg:
    #         return (calculate_weighted_score(result.score, weights), result, {})

    #     notion_scores: dict[str, float] = {}
    #     notion_available_weights: dict[str, float] = {}
    #     notion_details: dict[str, Any] = {}

    #     for notion_key, notion_cfg in notions_cfg.items():
    #         notion_weight = float(notion_cfg.get("weight", 0.0))
    #         raw_metric_weights = notion_cfg.get("metrics", {})
    #         rel_total = float(sum(raw_metric_weights.values()))

    #         metric_details: dict[str, Any] = {}
    #         if rel_total <= 0 or notion_weight <= 0:
    #             notion_scores[notion_key] = np.nan
    #             notion_available_weights[notion_key] = 0.0
    #             notion_details[notion_key] = {
    #                 "weight": notion_weight,
    #                 "score": np.nan,
    #                 "available_weight": 0.0,
    #                 "metrics": metric_details,
    #             }
    #             continue

    #         weighted_sum = 0.0
    #         available_rel_weight = 0.0

    #         for metric_key, raw_rel_weight in raw_metric_weights.items():
    #             rel_weight = float(raw_rel_weight) / rel_total
    #             score = result.score.get(metric_key)
    #             metric_details[metric_key] = {
    #                 "score": score,
    #                 "relative_weight": rel_weight,
    #                 "effective_weight": notion_weight * rel_weight,
    #             }

    #             if self._is_valid_score(score):
    #                 weighted_sum += float(score) * rel_weight
    #                 available_rel_weight += rel_weight

    #         if available_rel_weight == 0:
    #             notion_score = np.nan
    #             available_weight = 0.0
    #         else:
    #             notion_score = round(weighted_sum / available_rel_weight, 2)
    #             available_weight = notion_weight * available_rel_weight

    #         notion_scores[notion_key] = notion_score
    #         notion_available_weights[notion_key] = available_weight
    #         notion_details[notion_key] = {
    #             "weight": notion_weight,
    #             "score": notion_score,
    #             "available_weight": available_weight,
    #             "metrics": metric_details,
    #         }

    #     pillar_score = calculate_weighted_score(notion_scores, notion_available_weights)
    #     return (pillar_score, result, notion_details)

    # @staticmethod
    # def _is_valid_score(value: Any) -> bool:
    #     if value is None:
    #         return False
    #     try:
    #         return not np.isnan(value)
    #     except TypeError:
    #         return False