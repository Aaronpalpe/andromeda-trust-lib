# trust_library/base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from trust_library.utils import Result, EvaluationContext, calculate_weighted_score
from typing import Tuple


class Pillar(ABC):
    """
    Base class for all evaluation pillars.
    Each pillar only implements `analyse()`.
    `score()` is generic and delegates to `analyse()`.
    """

    @property
    @abstractmethod
    def pillar_key(self) -> str:
        """Pillar key in the config (e.g., 'fairness', 'privacy')."""
        ...

    @abstractmethod
    def analyse(self, context: EvaluationContext, config: dict) -> Result:
        """Executes all metrics and returns scores + properties."""
        ...

    def score(self, context: EvaluationContext, config: dict) -> Tuple[float, Result]:
        """Computes the aggregated weighted score of the pillar (0–5)."""
        result  = self.analyse(context, config.get("mappings", {}).get(self.pillar_key))
        weights = config.get("weights", {}).get(self.pillar_key, {})
        return (calculate_weighted_score(result.score, weights), result)