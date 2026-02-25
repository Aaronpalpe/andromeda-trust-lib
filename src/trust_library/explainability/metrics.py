from __future__ import annotations

from trust_library.base_metric import BaseMetric
from trust_library.utils import EvaluationContext

from . import explainability_metrics_core as core


_EXPL_KEY = "explainability_shap_metrics"
_EXPL_PARAMS_KEY = "explainability_params"
_EXPL_ERROR_KEY = "explainability_error"


def _get_or_compute(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_EXPL_KEY)
    if isinstance(cached, dict):
        return cached

    # Si ya falló antes, no reintentar
    if _EXPL_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_EXPL_ERROR_KEY]))

    params = ctx.extras.get(_EXPL_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}

    # (Opcional) validación rápida sklearn
    if not hasattr(ctx.model, "predict"):
        ctx.extras[_EXPL_ERROR_KEY] = "Model must implement predict() to compute explainability."
        raise RuntimeError(ctx.extras[_EXPL_ERROR_KEY])

    try:
        metrics = core.compute_shap_based_metrics(
            model=ctx.model,
            X=ctx.X_test,
            n_samples=int(params.get("n_samples", 50)),
            shap_threshold=float(params.get("shap_threshold", 1e-3)),
            top_k=int(params.get("top_k", 5)),
            seed=int(params.get("seed", 42)),
        )
    except Exception as exc:
        ctx.extras[_EXPL_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_EXPL_KEY] = metrics
    return metrics


class SparsityMetric(BaseMetric):
    """Fraction of features with |SHAP| > threshold (lower is better)."""

    def __init__(self):
        super().__init__("sparsity", "score_sparsity")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute(ctx)
        return {"value": float(m["sparsity"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Average fraction of active features (|SHAP| above threshold). Lower is more interpretable.",
            "Value": float(raw["value"]),
            "N Features": int(raw.get("n_features", 0)),
            "Sample Size": int(raw.get("sample_size", 0)),
            "SHAP Threshold": float(raw.get("shap_threshold", 0.0)),
            "Explainer": raw.get("explainer"),
        }


class FeatureEntropyMetric(BaseMetric):
    """Normalized entropy of global SHAP importances (lower is better)."""

    def __init__(self):
        super().__init__("feature_entropy", "score_feature_entropy")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute(ctx)
        return {"value": float(m["feature_entropy"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Normalized entropy of mean(|SHAP|) across features. Lower indicates explanations concentrated on fewer features.",
            "Value": float(raw["value"]),
            "N Features": int(raw.get("n_features", 0)),
            "Sample Size": int(raw.get("sample_size", 0)),
            "Explainer": raw.get("explainer"),
        }


class TopKConcentrationMetric(BaseMetric):
    """Share of global SHAP importance captured by top-k features (higher is better)."""

    def __init__(self):
        super().__init__("topk_concentration", "score_topk_concentration")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute(ctx)
        return {"value": float(m["topk_concentration"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Fraction of total global SHAP importance accounted for by the top-k features. Higher indicates more concentrated explanations.",
            "Value": float(raw["value"]),
            "Top K": int(raw.get("top_k", 0)),
            "N Features": int(raw.get("n_features", 0)),
            "Sample Size": int(raw.get("sample_size", 0)),
            "Explainer": raw.get("explainer"),
        }
