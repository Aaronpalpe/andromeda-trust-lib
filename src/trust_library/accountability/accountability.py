from typing import Any, Dict, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext, Result
from .metrics import (
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

    def analyse(
        self,
        context: EvaluationContext,
        config: Dict[str, Any],
    ) -> Result:

        metrics: List[Any] = [
            TrainTestSplitMetric(),
            MissingDataMetric(),
            NormalizationMetric(),
            RegularizationMetric(),
            FactsheetCompletenessMetric(),
        ]

        scores = {}
        properties = {}

        for metric in metrics:
            # raw = metric.compute(context)
            # score = metric.compute_score(raw, config)
            # props = metric.build_properties(raw)

            # scores[metric.metric_key] = score
            # properties[metric.metric_key] = props

            result = metric.evaluate(context, config)
            scores[metric.metric_key]     = result.score
            properties[metric.metric_key] = result.properties

        return Result(score=scores, properties=properties)