from __future__ import annotations

from typing import Dict, Any, List
from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext, Result
from . import sustainability_metrics_core as core
from .metrics import (
    BaseMetric,
    EnergyConsumptionMetric,
    EmissionsMetric,
    CarbonIntensityMetric,
    # EnergyEfficiencyMetric,
)


class SustainabilityPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "sustainability"
    
    # def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
    #     run_data = core.track_training_run(
    #         model=context.model,
    #         train_data=context.train_data,
    #     )   

    #     context.extras["run_data"] = run_data

    def get_metrics(self) -> List[BaseMetric]:
        metrics: List[Any] = [
            EnergyConsumptionMetric(),
            EmissionsMetric(),
            CarbonIntensityMetric(),
        ]

        return metrics