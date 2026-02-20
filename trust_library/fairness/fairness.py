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

_METRICS = [
    ("underfitting",                  UnderfittingMetric()),
    ("overfitting",                   OverfittingMetric()),
    ("class_balance",                 ClassBalanceMetric()),
    ("statistical_parity_difference", StatisticalParityMetric()),
    ("disparate_impact",              DisparateImpactMetric()),
    ("equal_opportunity_difference",  EqualOpportunityMetric()),
    ("average_odds_difference",       AverageOddsMetric()),
    ("accuracy_parity",               AccuracyParityMetric()),
    ("predictive_parity",             PredictiveParityMetric()),
    ("treatment_equality",            TreatmentEqualityMetric()),
    ("calibration",                   CalibrationGapMetric(n_bins=10)),
    ("well_calibration",              WellCalibrationMetric(n_bins=10)),
    ("generalized_entropy",           GeneralizedEntropyMetric(alpha=2)),
    ("theil_index",                   TheilIndexMetric()),
    ("coefficient_variation",         CoefficientVariationMetric()),
    ("consistency",                   ConsistencyMetric(k=5)),
    ("class_imbalance",               ClassImbalanceMetric()),
    ("kl_divergence",                 KLDivergenceMetric()),
    ("smoothed_edf",                  SmoothedEDFMetric(alpha=1.0)),
    ("bias_amplification",            BiasAmplificationMetric()),
    ("cohens_d",                      CohensDMetric()),
]

class FairnessPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "fairness"

    def analyse(self, context: EvaluationContext, config: dict) -> Result:
        scores, properties = {}, {}
        for name, metric in _METRICS:
            result = metric.evaluate(context, config)
            scores[name]     = result.score
            properties[name] = result.properties
        return Result(score=scores, properties=properties)
