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

from trust_library.pillar import Pillar

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
    SmoothedEDFMetric,
    BiasAmplificationMetric,
    CohensDMetric,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry
# ─────────────────────────────────────────────────────────────────────────────



class FairnessPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "fairness"

    def analyse(
        self,
        context: EvaluationContext,
        config: Dict[str, Any],
    ) -> Result:
        
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
            SmoothedEDFMetric(alpha=1.0),
            BiasAmplificationMetric(),
            CohensDMetric(),
        ]
        
        scores: Dict[str, float] = {}
        properties: Dict[str, Dict[str, Any]] = {}
        
        for metric in metrics:
            result = metric.evaluate(context, config)
            scores[metric.metric_key]     = result.score
            properties[metric.metric_key] = result.properties
        return Result(score=scores, properties=properties)
