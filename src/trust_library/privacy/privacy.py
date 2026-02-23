from __future__ import annotations

from typing import Dict, Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext, Result

from .metrics import (
    BaseMetric,
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
    
        
    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        return

    def get_metrics(self) -> List[BaseMetric]:
        metrics: List[Any] = [
            EpsilonStarMetric(),
            SHAPRMetric(),
            AttributeInferenceMetric(),
            PrivacyRiskMetric(),
            KAnonymityMetric(),
            LDiversityMetric(),
            TClosenessMetric(),
        ]

        return metrics