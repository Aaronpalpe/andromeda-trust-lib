from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import warnings

import numpy as np

#_DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]


class BaseMetric(ABC):
    """
    Unified base class for Trust metrics (Fairness, Accountability, etc.)

    Encapsulates:
        - raw computation
        - threshold-based scoring OR min-max normalization
        - optional custom scoring
        - safe evaluation
        - Result construction
    """

    class ProblemType(Enum):
        BINARY = "binary"
        MULTICLASS = "multiclass"
        BOTH = "both"

    def __init__(
        self,
        metric_key: str,
        score_config_key: str | None,
        problem_type: "BaseMetric.ProblemType | None" = None,
    ) -> None:
        self.metric_key = metric_key
        self.score_config_key = score_config_key
        self.problem_type = problem_type

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def evaluate(self, context, config: dict | None) -> Result:
        """
        Safe wrapper executing compute -> scoring -> property building.
        """
        from trust_library.utils import Result

        if not self.is_compatible_with(context):
            return Result(
                np.nan,
                {
                    "Status": "SKIPPED",
                    "Reason": self.incompatibility_reason(context),
                    "Supported Problem Type": (
                        self.problem_type.value if self.problem_type is not None else "agnostic"
                    ),
                    "Detected Problem Type": self._context_problem_type_label(context),
                },
            )

        try:
            raw = self.compute(context)

            score = self.compute_score(raw, config)
            properties = self.build_properties(raw)

            return Result(score, properties)

        except Exception as e:
            return Result(np.nan, {"Error": str(e)})

    def is_compatible_with(self, context) -> bool:
        """
        Return whether the metric can run for the inferred evaluation problem type.

        Metrics with ``problem_type=None`` are treated as task-agnostic in this
        first version and always run.
        """
        if self.problem_type is None:
            return True

        if not getattr(context, "is_classification", False):
            return False

        detected_problem_type = getattr(context, "problem_type", None)
        if detected_problem_type is None:
            return False

        if self.problem_type == BaseMetric.ProblemType.BOTH:
            return detected_problem_type in {
                BaseMetric.ProblemType.BINARY,
                BaseMetric.ProblemType.MULTICLASS,
            }

        return detected_problem_type == self.problem_type

    def incompatibility_reason(self, context) -> str:
        if self.problem_type is None:
            return "Metric is task-agnostic."

        if not getattr(context, "is_classification", False):
            return (
                "Metric requires a classification problem, but the evaluation "
                "context was not inferred as classification."
            )

        detected_problem_type = getattr(context, "problem_type", None)
        if detected_problem_type is None:
            return "Metric requires a binary or multiclass classification problem, but the problem type could not be inferred."

        if self.problem_type == BaseMetric.ProblemType.BOTH:
            return (
                "Metric supports binary and multiclass classification, but the "
                "evaluation context does not expose a compatible classification mode."
            )

        return (
            f"Metric supports {self.problem_type.value} classification, but the "
            f"evaluation context was inferred as {detected_problem_type.value}."
        )

    def _context_problem_type_label(self, context) -> str:
        detected_problem_type = getattr(context, "problem_type", None)
        if detected_problem_type is None:
            return "unknown"
        if isinstance(detected_problem_type, BaseMetric.ProblemType):
            return detected_problem_type.value
        return str(detected_problem_type)

    # ─────────────────────────────────────────────────────────────
    # Core metric logic (to implement)
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def compute(self, context) -> dict:
        """Return raw metric computation results."""
        ...

    @abstractmethod
    def build_properties(self, raw: dict) -> dict:
        """Return human-readable metric properties."""
        ...

    # ─────────────────────────────────────────────────────────────
    # Scoring logic
    # ─────────────────────────────────────────────────────────────

    def compute_score(self, raw: dict, config: dict | None) -> float:
        """
        Default scoring using thresholds or min-max normalization.
        Override if necessary.

        Config structure for thresholds:
            score_config_key:
                thresholds:
                    value: [0.8, 0.85, 0.9, 0.95]

        Config structure for normalized scoring:
            score_config_key:
                normalized:
                    min_val: 0.0
                    max_val: 1.0
                    higher_is_better: true
        """
        from trust_library.utils import calculate_score, calculate_score_normalized

        if self.score_config_key is None:
            return self.custom_score(raw)

        # Try normalized scoring first
        norm_config = self._get_normalized_config(config)
        if norm_config is not None:
            return calculate_score_normalized(
                value=raw.get("value"),
                min_val=norm_config.get("min_val", 0.0),
                max_val=norm_config.get("max_val", 1.0),
                higher_is_better=norm_config.get("higher_is_better", True),
            )

        # Fall back to threshold scoring
        thresholds = self._get_thresholds(config)
        if thresholds is None:
            raise ValueError(f"No thresholds or normalized config available for '{self.score_config_key}'.")
        return calculate_score(raw.get("value"), thresholds)

    def custom_score(self, raw: dict):
        """
        Override when metric does not use threshold scoring.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement custom_score "
            "or provide score_config_key."
        )

    def _get_thresholds(self, config: dict | None):
        if config is None:
            warnings.warn("Config is None. Using default thresholds.", RuntimeWarning, stacklevel=2,)
            return None

        thresholds = (
            config.get(self.score_config_key, {})
            .get("thresholds", {})
            .get("value")
        )

        if thresholds is None:
            # Don't warn if normalized config exists
            if self._get_normalized_config(config) is None:
                warnings.warn(f"No thresholds configured for '{self.score_config_key}'.", RuntimeWarning, stacklevel=2,)
            return None
        return thresholds

    def _get_normalized_config(self, config: dict | None) -> dict | None:
        """
        Get normalized scoring config if available.

        Example config:
            score_accuracy:
                normalized:
                    min_val: 0.6    # 60% accuracy = score 1
                    max_val: 1.0    # 100% accuracy = score 5
                    higher_is_better: true
        """
        if config is None:
            return None

        norm_config = (
            config.get(self.score_config_key, {})
            .get("normalized")
        )

        return norm_config
