from typing import Any, Dict, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext
from .metrics import (
    BaseMetric,
    TrainTestSplitMetric,
    MissingDataMetric,
    NormalizationMetric,
    RegularizationMetric,
    FactsheetCompletenessMetric,
)


class AccountabilityPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "accountability"
    
    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        return

    def get_metrics(self) -> List[BaseMetric]:
        metrics: List[Any] = [
            TrainTestSplitMetric(),
            MissingDataMetric(),
            NormalizationMetric(),
            RegularizationMetric(),
            FactsheetCompletenessMetric(),
        ]

        return metrics