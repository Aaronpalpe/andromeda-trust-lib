from __future__ import annotations

from typing import Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext

from .metrics import (
    SparsityMetric,
    FeatureEntropyMetric,
    TopKConcentrationMetric,
)
from . import explainability_metrics_core as core


class ExplainabilityPillar(Pillar):
    """Explainability pillar based on SHAP-derived metrics."""

    @property
    def pillar_key(self) -> str:
        return "explainability"

    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        # Optional parameters (kept under mappings.explainability.params)
        params = (config or {}).get("params", {})

        n_samples = int(params.get("n_samples", 50))
        shap_threshold = float(params.get("shap_threshold", 1e-3))
        top_k = int(params.get("top_k", 5))
        seed = int(params.get("seed", 42))

        # Store params for lazy computation in metrics (and for reproducibility)
        context.extras["explainability_params"] = {
            "n_samples": n_samples,
            "shap_threshold": shap_threshold,
            "top_k": top_k,
            "seed": seed,
        }

        # Best-effort eager compute (but never let it crash the whole evaluation)
        try:
            metrics = core.compute_shap_based_metrics(
                model=context.model,
                X=context.X_test,
                n_samples=n_samples,
                shap_threshold=shap_threshold,
                top_k=top_k,
                seed=seed,
            )
            context.extras["explainability_shap_metrics"] = metrics
        except Exception as exc:
            context.extras["explainability_error"] = str(exc)
            # Leave computation to metric wrappers (they will surface the error safely)

    def get_metrics(self) -> List[Any]:
        return [
            SparsityMetric(),
            FeatureEntropyMetric(),
            TopKConcentrationMetric(),
        ]
