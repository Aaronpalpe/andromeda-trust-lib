from __future__ import annotations

from typing import Dict, Any, List
from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext, Result
from . import sustainability_metrics_core as core
from .metrics import (
    EnergyConsumptionMetric,
    EmissionsMetric,
    CarbonIntensityMetric,
    # EnergyEfficiencyMetric,
)


class SustainabilityPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "sustainability"

    def analyse(
        self,
        context: EvaluationContext,
        config: Dict[str, Any],
    ) -> Result:
        """
        Executes training tracking once and computes all sustainability metrics.
        """

        # SINGLE TRAINING EXECUTION
        run_data = core.track_training_run(
            model=context.model,
            train_data=context.train_data,
        )

        metrics: List[Any] = [
            EnergyConsumptionMetric(),
            EmissionsMetric(),
            CarbonIntensityMetric(),
            # EnergyEfficiencyMetric(),
        ]

        scores: Dict[str, float] = {}
        properties: Dict[str, Dict[str, Any]] = {}

        for metric in metrics:

            # raw = metric.compute(context, run_data)
            # score = metric.compute_score(raw, config)
            # props = metric.build_properties(raw)

            # scores[metric.metric_key] = score
            # properties[metric.metric_key] = props

            result = metric.evaluate(context, config)
            scores[metric.metric_key]     = result.score
            properties[metric.metric_key] = result.properties

        return Result(score=scores, properties=properties)