from __future__ import annotations

"""
fairness.py
===========

Main entry point for the fairness analysis pipeline.

Delegates metric computation to metric classes defined in metrics.py.
"""

"""
fairness.py
===========
Main entry point for the fairness analysis pipeline.
"""

from typing import Any, Dict, List
import warnings
warnings.filterwarnings("ignore")

from trust_library.pillar import BaseMetric, Pillar

from trust_library.utils import Result, EvaluationContext
from .metrics import (
    UnderfittingMetric,
    OverfittingMetric,
    ClassBalanceMetric,
    StatisticalParityMetric,
    DisparateImpactMetric,
    EqualOpportunityMetric,
    AverageOddsMetric,
    AccuracyParityMetric,
    PredictiveParityMetric,
    TreatmentEqualityMetric,
    CalibrationGapMetric,
    WellCalibrationMetric,
    GeneralizedEntropyMetric,
    TheilIndexMetric,
    CoefficientVariationMetric,
    ConsistencyMetric,
    ClassImbalanceMetric,
    KLDivergenceMetric,
    ConditionalDemographicDisparityMetric,
    SmoothedEDFMetric,
    BiasAmplificationMetric,
    BetweenGroupGeneralizedEntropyMetric,
    CohensDMetric,
    ZTestDiffMetric,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry
# ─────────────────────────────────────────────────────────────────────────────



class FairnessPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "fairness"
    
    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        return


    def get_metrics(self) -> List[BaseMetric]:
        metrics: List[Any] = [
            UnderfittingMetric(),
            OverfittingMetric(),
            ClassBalanceMetric(),
            StatisticalParityMetric(),
            DisparateImpactMetric(),
            EqualOpportunityMetric(),
            AverageOddsMetric(),

            AccuracyParityMetric(),
            PredictiveParityMetric(),
            TreatmentEqualityMetric(),
            CalibrationGapMetric(n_bins=10),
            WellCalibrationMetric(n_bins=10),

            GeneralizedEntropyMetric(alpha=2),
            TheilIndexMetric(),
            CoefficientVariationMetric(),
            ConsistencyMetric(k=5),
            ClassImbalanceMetric(),
            KLDivergenceMetric(),
            ConditionalDemographicDisparityMetric(),
            SmoothedEDFMetric(alpha=0.5),
            BiasAmplificationMetric(),
            BetweenGroupGeneralizedEntropyMetric(),
            CohensDMetric(),
            ZTestDiffMetric(),
        ]

        return metrics