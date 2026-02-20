from __future__ import annotations

from typing import Dict, Any, List

from trust_library.pillar import Pillar
from trust_library.utils import Result

from .metrics import (
    EpsilonStarMetric,
    SHAPRMetric,
    AttributeInferenceMetric,
    PrivacyRiskMetric,
    KAnonymityMetric,
    LDiversityMetric,
    TClosenessMetric,
)


# =============================================================================
# Privacy Pillar
# =============================================================================

class PrivacyPillar(Pillar):

    @property
    def pillar_key(self) -> str:
        return "privacy"

    def analyse(
        self,
        context: Any,
        config: Dict[str, Any],
    ) -> Result:
        """
        Execute all privacy metrics and aggregate results.
        """

        metrics: List[Any] = [
            EpsilonStarMetric(),
            SHAPRMetric(),
            AttributeInferenceMetric(),
            PrivacyRiskMetric(),
            KAnonymityMetric(),
            LDiversityMetric(),
            TClosenessMetric(),
        ]

        scores: Dict[str, float] = {}
        properties: Dict[str, Dict[str, Any]] = {}

        for metric in metrics:

            raw = metric.compute(context)
            score = metric.compute_score(raw, config)
            props = metric.build_properties(raw)

            scores[metric.metric_key] = score
            properties[metric.metric_key] = props

        return Result(
            score=scores,
            properties=properties,
        )