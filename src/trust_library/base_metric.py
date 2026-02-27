from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from trust_library.utils import Result, calculate_score
import warnings

# _DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]


class BaseMetric(ABC):
    """
    Unified base class for Trust metrics (Fairness, Accountability, etc.)

    Encapsulates:
        - raw computation
        - threshold-based scoring
        - optional custom scoring
        - safe evaluation
        - Result construction
    """

    def __init__(
        self,
        metric_key: str,
        score_config_key: str | None,
    ):
        self.metric_key = metric_key
        self.score_config_key = score_config_key

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def evaluate(self, context, config: dict | None) -> Result:
        """
        Safe wrapper executing compute -> scoring -> property building.
        """

        try:
            raw = self.compute(context)

            score = self.compute_score(raw, config)
            properties = self.build_properties(raw)

            return Result(score, properties)

        except Exception as e:
            return Result(np.nan, {"Error": str(e)})

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

    def compute_score(self, raw: dict, config: dict | None) -> int:
        """
        Default scoring using thresholds.
        Override if necessary.
        """

        if self.score_config_key is None:
            return self.custom_score(raw)

        thresholds = self._get_thresholds(config)
        if thresholds is None:
            raise ValueError(f"No thresholds available for '{self.score_config_key}'.")
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
            #return _DEFAULT_THRESHOLDS
            return None

        thresholds = (
            config.get(self.score_config_key, {})
            .get("thresholds", {})
            .get("value")
        )

        if thresholds is None:
            warnings.warn(f"No thresholds configured for '{self.score_config_key}'. Using default thresholds.", RuntimeWarning, stacklevel=2,)
            # return _DEFAULT_THRESHOLDS
            return None
        return thresholds