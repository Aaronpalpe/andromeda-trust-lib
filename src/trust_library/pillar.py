# trust_library/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
import time
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
        t0 = time.time()
        self.prepare(context, config)
        metrics = self.get_metrics()
        t1 = time.time()
        print(f"Preparation completed in {t1 - t0:.2f} seconds. Starting metric evaluations...")

        scores: dict[str, float] = {}
        properties: dict[str, dict[str, Any]] = {}

        for metric in metrics:
            start_time = time.time()
            result = metric.evaluate(context, config)
            scores[metric.metric_key] = result.score
            properties[metric.metric_key] = result.properties
            elapsed_time = time.time() - start_time
            print(f"Metric '{metric.metric_key}' computed in {elapsed_time:.2f} seconds.")

        return Result(score=scores, properties=properties)

    def score(self, context: EvaluationContext, config: dict) -> Tuple[float, Result]:
        """Computes the aggregated weighted score of the pillar (0–5)."""
        result  = self.analyse(context, config.get("mappings", {}).get(self.pillar_key))
        weights = config.get("weights", {}).get(self.pillar_key, {})
        return (calculate_weighted_score(result.score, weights), result)