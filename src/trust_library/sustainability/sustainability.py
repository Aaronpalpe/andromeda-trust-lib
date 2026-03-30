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

    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        sustainability = context.factsheet.get("sustainability", {})
        use_codecarbon = sustainability.get("use_codecarbon", {}).get("value", False)

        if not use_codecarbon:
            return

        print("CodeCarbon enabled - tracking training run...")

        try:
            run_data = core.track_training_run(
                model=context.model,
                train_data=context.train_data,
            )

            # Update the in-memory factsheet with the calculated values
            for key, value in run_data.items():
                if key in sustainability:
                    sustainability[key]["value"] = value
                    formatted_value = f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
                    print(f"   Updated factsheet: {key} = {formatted_value}")

            # Save indicator that CodeCarbon was executed
            context.extras["codecarbon_executed"] = True
            context.extras["codecarbon_data"] = run_data

            print("CodeCarbon tracking completed successfully.")

        except Exception as e:
            print(f"Warning: CodeCarbon tracking failed: {e}")
            context.extras["codecarbon_error"] = str(e)


    def get_metrics(self) -> List[BaseMetric]:
        metrics: List[Any] = [
            EnergyConsumptionMetric(),
            EmissionsMetric(),
            CarbonIntensityMetric(),
        ]

        return metrics