from __future__ import annotations

from trust_library.base_metric import BaseMetric
from trust_library.utils import EvaluationContext

from . import explainability_metrics_core as core
import numpy as np

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
        metrics = core.shap_based_metrics(
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

class InteractionStrengthMetric(BaseMetric):
    def __init__(self): super().__init__("interaction_strength", "score_interaction_strength")
    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute(ctx)
        return {"value": float(m["interaction_strength"]), **m}
    def build_properties(self, raw: dict) -> dict: 
        return {"Metric Description": "Proportion of SHAP importance coming from feature interactions.", 
                "Value": f"{raw['value']:.6f}",
                "N Features": int(raw.get("n_features", 0)),
                "Sample Size": int(raw.get("sample_size", 0)),
                "Explainer": raw.get("explainer"),
                }
    

class AlgorithmClassMetric(BaseMetric):
    def __init__(self):
        super().__init__("algorithm_class", "score_algorithm_class") #AÑADIR

    def compute(self, ctx: EvaluationContext) -> dict:
        model_type = ctx.factsheet.get("general", {}).get("model_type", {}).get("value", None)
        return core.algorithm_class(ctx.model, model_type=model_type)

    def custom_score(self, raw: dict):
        return raw.get("value")
    
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Score assigned based on model class type.",
            "Value": f"{raw['value']:.6f}",
            "Depends on" : "Model",
            "Model Type": raw.get("model_type"),
        }


class CorrelatedFeaturesMetric(BaseMetric):
    def __init__(self):
        super().__init__("correlated_features", "score_correlated_features")

    def compute(self, ctx: EvaluationContext) -> dict:
        return core.correlated_features(
            X_train=ctx.X_train,
            X_test=ctx.X_test,
        )

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Penalty based on percentage of highly correlated features.",
            "Depends on": "Training Data",
            "Percentage of highly correlated features": f"{raw['value']:.6f}",
        }


class ModelSizeMetric(BaseMetric):
    def __init__(self):
        super().__init__("model_size", "score_model_size")

    def compute(self, ctx: EvaluationContext) -> dict:
        return core.model_size(ctx.X_train)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Score based on number of input features.",
            "Depends on": "Training Data",
            "Number of Features": f"{raw['value']:.6f}",
        }


class FeatureRelevanceMetric(BaseMetric):
    def __init__(self):
        super().__init__("feature_relevance", "score_feature_relevance")

    def compute(self, ctx: EvaluationContext) -> dict:
        return core.feature_relevance(
            model=ctx.model,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
        )

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Evaluates concentration and outliers in feature importance distribution.",
            "Outliers": raw.get("n_outliers", 0),
            "Depends on": "Model and Training Data",
            "Importances": raw.get("importances", []),
            "Percentage of feature that make up over 60% of all features importance": f"{raw['value']:.6f}",
        }


# =============================================================================
# Additional Explainability & Surrogate Metrics Wrappers
# =============================================================================

# class PerformanceDifferenceMetric(BaseMetric):
#     def __init__(self): 
#         super().__init__("performance_difference", "score_performance_difference")
        
#     def compute(self, ctx: EvaluationContext) -> dict:
#         surrogate = ctx.extras.get("surrogate")
#         if surrogate is None:
#             raise ValueError("Missing 'surrogate' in ctx.extras.")
            
#         return core.performance_difference(ctx.model, surrogate, ctx.X_test, ctx.y_test)
        
#     def build_properties(self, raw: dict) -> dict: 
#         return {
#             "Metric Description": "Difference in accuracy between original model and surrogate explainer.", 
#             "Performance Difference": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A",
#             "Original Perf": f"{raw.get('perf_original', 0):.4f}",
#             "Surrogate Perf": f"{raw.get('perf_explainer', 0):.4f}"
#         }

class NumberOfRulesMetric(BaseMetric):
    def __init__(self): super().__init__("number_of_rules", "score_number_of_rules")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.number_of_rules(ctx.model) # return core.number_of_rules(getattr(ctx.model, "tree_", ctx.model))
    def build_properties(self, raw: dict) -> dict: 
        return {"Metric Description": "Number of leaves/rules in the tree.", "Value": f"{raw['value']:.0f}"}

class AverageRuleLengthMetric(BaseMetric):
    def __init__(self): super().__init__("average_rule_length", "score_average_rule_length")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.average_rule_length(ctx.model)
    def build_properties(self, raw: dict) -> dict: 
        return {"Metric Description": "Average depth/length of paths in the tree.", "Value": f"{raw['value']:.4f}"}

class RuleStatsMetric(BaseMetric):
    def __init__(self): super().__init__("rule_stats", "score_rule_stats")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.rule_stats(ctx.model)
    def build_properties(self, raw: dict) -> dict: 
        return {"Metric Description": "Average rule length for rule-based models.", "Value": f"{raw['value']:.4f}"}

class TreeDepthMetric(BaseMetric):
    def __init__(self): super().__init__("tree_depth", "score_tree_depth")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_depth(ctx.model)
    def build_properties(self, raw: dict) -> dict: 
        return {"Metric Description": "Maximum depth of the tree model.", "Value": f"{raw['value']:.0f}"}


# =============================================================================
# Instance-Based Fidelity Metrics Wrappers
# =============================================================================

class FaithfulnessMetric(BaseMetric):
    def __init__(self): super().__init__("faithfulness", "score_faithfulness")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        # Extraemos la instancia, coeficientes (importancias locales) y la base
        # Si no se proveen para una instancia, promediamos sobre las 10 primeras de X_test por defecto
        X_np = np.asarray(ctx.X_test)
        base = ctx.extras.get("base_values", np.mean(X_np, axis=0))
        local_importances = ctx.extras.get("feature_weights") 

        if local_importances is None:
            raise ValueError("Missing 'feature_weights' in ctx.extras for FaithfulnessMetric.")

        scores = []
        # Evaluamos las primeras N instancias si no se especifica una sola
        limit = min(len(X_np), 10) 
        for i in range(limit):
            res = core.faithfulness_metric(ctx.model, X_np[i], local_importances[i], base)
            scores.append(res["value"])

        return {"value": float(np.mean(scores))}
        
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Correlation between feature importance and model performance drop (Faithfulness).", 
            "Value": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
        }

class MonotonicityMetric(BaseMetric):
    def __init__(self): super().__init__("monotonicity", "score_monotonicity")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        X_np = np.asarray(ctx.X_test)
        base = ctx.extras.get("base_values", np.mean(X_np, axis=0))
        local_importances = ctx.extras.get("feature_weights") 

        if local_importances is None:
            raise ValueError("Missing 'feature_weights' in ctx.extras for MonotonicityMetric.")

        scores = []
        limit = min(len(X_np), 10) 
        for i in range(limit):
            res = core.monotonicity_metric(ctx.model, X_np[i], local_importances[i], base)
            scores.append(res["value"])

        return {"value": float(np.mean(scores))}
        
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Proportion of instances showing monotonic performance increase when adding features by importance.", 
            "Value (Ratio)": f"{raw['value']:.4f}" if not np.isnan(raw['value']) else "N/A"
        }
    
# =============================================================================
# Perturbation & Correlation Metrics Wrappers (Xplique-style)
# =============================================================================

# class ShapleyCorrMetric(BaseMetric):
#     def __init__(self): 
#         super().__init__("shapley_corr", "score_shapley_corr")
        
#     def compute(self, ctx: EvaluationContext) -> dict:
#         feature_weights = ctx.extras.get("feature_weights")
#         ground_truth_weights = ctx.extras.get("ground_truth_weights")
        
#         if feature_weights is None or ground_truth_weights is None:
#             raise ValueError("Missing 'feature_weights' or 'ground_truth_weights' in ctx.extras for ShapleyCorrMetric.")
            
#         return core.shapley_corr(feature_weights, ground_truth_weights)
        
#     def build_properties(self, raw: dict) -> dict: 
#         return {
#             "Metric Description": "Pearson correlation between generated feature weights and ground truth weights.", 
#             "Shapley Correlation": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
#         }

# class ROARMetric(BaseMetric):
#     def __init__(self): 
#         super().__init__("roar", "score_roar")
        
#     def compute(self, ctx: EvaluationContext) -> dict:
#         train_feature_weights = ctx.extras.get("train_feature_weights")
#         test_feature_weights = ctx.extras.get("test_feature_weights")
        
#         if train_feature_weights is None or test_feature_weights is None:
#             raise ValueError("Missing 'train_feature_weights' or 'test_feature_weights' in ctx.extras for ROARMetric.")
            
#         return core.roar(
#             model=ctx.model, 
#             X_train=ctx.X_train, 
#             y_train=ctx.y_train, 
#             X_test=ctx.X_test, 
#             y_test=ctx.y_test, 
#             train_feature_weights=train_feature_weights, 
#             test_feature_weights=test_feature_weights
#         )
        
#     def build_properties(self, raw: dict) -> dict: 
#         return {
#             "Metric Description": "Remove and Retrain (ROAR) Score. Higher is better (indicates important features were accurately identified).", 
#             "ROAR AUC": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
#         }

class InfidelityMetric(BaseMetric):
    def __init__(self): 
        super().__init__("infidelity", "score_infidelity")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        feature_weights = ctx.extras.get("feature_weights") # Se asumen los pesos sobre X_test
        
        if feature_weights is None:
            raise ValueError("Missing 'feature_weights' in ctx.extras for InfidelityMetric.")
            
        return core.infidelity(
            model=ctx.model, 
            X_test=ctx.X_test, 
            feature_weights=feature_weights
        )
        
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Measures if perturbations in important features proportionally affect the model's output. Lower is better.", 
            "Infidelity Score": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
        }


_GLOBAL_KEY = "xai_global_metrics"
_LOCAL_KEY = "xai_local_metrics"
_SURROGATE_KEY = "xai_surrogate_metrics"

# Separamos las claves de error para evitar fallos en cascada
_GLOBAL_ERROR_KEY = "xai_global_error"
_LOCAL_ERROR_KEY = "xai_local_error"
_SURROGATE_ERROR_KEY = "xai_surrogate_error"
_CONSISTENCY_ERROR_KEY = "xai_consistency_error"

# =============================================================================
# Cache Managers
# =============================================================================

# =============================================================================
# Cache Managers
# =============================================================================

# def _get_or_compute_global(ctx: EvaluationContext) -> dict:
#     cached = ctx.extras.get(_GLOBAL_KEY)
#     if isinstance(cached, dict): return cached
#     if _GLOBAL_ERROR_KEY in ctx.extras: raise RuntimeError(str(ctx.extras[_GLOBAL_ERROR_KEY]))

#     importances = ctx.extras.get("importances")
#     partial_dependencies = ctx.extras.get("partial_dependencies")
#     conditional_importances = ctx.extras.get("conditional_importances")
    
#     if importances is None:
#         ctx.extras[_GLOBAL_ERROR_KEY] = "Missing global explainability objects in ctx.extras (importances, partial_dependencies...)"
#         raise ValueError(ctx.extras[_GLOBAL_ERROR_KEY])

#     try:
#         metrics = core.global_explainability_metrics(importances, partial_dependencies, conditional_importances)
#         ctx.extras[_GLOBAL_KEY] = metrics
#         return metrics
#     except Exception as exc:
#         ctx.extras[_GLOBAL_ERROR_KEY] = str(exc)
#         raise

# def _get_or_compute_local(ctx: EvaluationContext) -> dict:
#     cached = ctx.extras.get(_LOCAL_KEY)
#     if isinstance(cached, dict): return cached
#     if _LOCAL_ERROR_KEY in ctx.extras: raise RuntimeError(str(ctx.extras[_LOCAL_ERROR_KEY]))

#     local_importances = ctx.extras.get("feature_weights")
#     if local_importances is None:
#         ctx.extras[_LOCAL_ERROR_KEY] = "Missing 'local_importances' in ctx.extras."
#         raise ValueError(ctx.extras[_LOCAL_ERROR_KEY])

#     try:
#         metrics = core.local_explainability_metrics(local_importances, ctx.X_test)
#         ctx.extras[_LOCAL_KEY] = metrics
#         return metrics
#     except Exception as exc:
#         ctx.extras[_LOCAL_ERROR_KEY] = str(exc)
#         raise

# def _get_or_compute_surrogate(ctx: EvaluationContext) -> dict:
#     cached = ctx.extras.get(_SURROGATE_KEY)
#     if isinstance(cached, dict): return cached
#     if _SURROGATE_ERROR_KEY in ctx.extras: raise RuntimeError(str(ctx.extras[_SURROGATE_ERROR_KEY]))

#     surrogate = ctx.extras.get("surrogate")
#     if surrogate is None:
#         ctx.extras[_SURROGATE_ERROR_KEY] = "Missing 'surrogate' model in ctx.extras."
#         raise ValueError(ctx.extras[_SURROGATE_ERROR_KEY])

#     try:
#         metrics = core.surrogate_explainability_metrics(
#             X_test=ctx.X_test, y_test=ctx.y_test, y_pred=ctx.y_pred_test,
#             surrogate=surrogate, is_regression=ctx.extras.get("is_regression", False)
#         )
#         ctx.extras[_SURROGATE_KEY] = metrics
#         return metrics
#     except Exception as exc:
#         ctx.extras[_SURROGATE_ERROR_KEY] = str(exc)
#         raise


# =============================================================================
# Global Metrics Wrappers
# =============================================================================

_GLOBAL_KEY = "xai_global_metrics"
_GLOBAL_ERROR_KEY = "xai_global_error"

def _get_or_compute_global_xai(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_GLOBAL_KEY)
    if isinstance(cached, dict):
        return cached

    if _GLOBAL_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_GLOBAL_ERROR_KEY]))

    try:
        # Extraemos los datos precalculados asumiendo estructuras nativas de Python:
        # global_importances: dict[str, float]
        # conditional_importances: dict[group, dict[str, float]]
        # pdp_averages: dict[str, list[float]]
        # pdp_individuals: dict[str, list[list[float]]]
        # pdp_grids: dict[str, list[float]]
        global_imps = ctx.extras.get("global_importances", {})
        cond_imps = ctx.extras.get("conditional_importances", {})
        pdp_avgs = ctx.extras.get("pdp_averages", {})

        global_vals = list(global_imps.values())
        global_ranked = sorted(global_imps.keys(), key=lambda k: global_imps[k], reverse=True)
        cond_ranked = {
            g: sorted(imps.keys(), key=lambda k: imps[k], reverse=True) 
            for g, imps in cond_imps.items()
        }

        metrics = {}
        metrics["alpha_score"] = core.alpha_score(global_vals) if global_vals else np.nan
        metrics["spread_ratio"] = core.spread_ratio(global_vals) if global_vals else np.nan
        metrics["spread_divergence"] = core.spread_divergence(global_vals) if global_vals else np.nan
        metrics["position_parity"] = core.position_parity(cond_ranked, global_ranked) if cond_ranked else np.nan
        metrics["rank_alignment"] = core.rank_alignment(cond_imps, global_imps) if cond_imps else np.nan
        metrics["xai_ease_score"] = core.xai_ease_score(pdp_avgs, global_ranked) if pdp_avgs else np.nan

    except Exception as exc:
        ctx.extras[_GLOBAL_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_GLOBAL_KEY] = metrics
    return metrics


class AlphaImportanceScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("alpha_score", "score_alpha_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["alpha_score"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Smallest proportion of features that account for alpha (0.8) of the overall feature importance.",
            "Value": f"{raw['value']:.4f}"
        }


class SpreadRatioMetric(BaseMetric):
    def __init__(self):
        super().__init__("spread_ratio", "score_spread_ratio")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["spread_ratio"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Degree of evenness in the distribution of feature importance values (0 to 1).",
            "Value": f"{raw['value']:.4f}"
        }


class SpreadDivergenceMetric(BaseMetric):
    def __init__(self):
        super().__init__("spread_divergence", "score_spread_divergence")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["spread_divergence"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Inverse of Jensen-Shannon distance for feature importances. Lower concentrates interpretability.",
            "Value": f"{raw['value']:.4f}"
        }


class PositionParityMetric(BaseMetric):
    def __init__(self):
        super().__init__("position_parity", "score_position_parity")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["position_parity"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Measures how well top feature importances maintain ranking considering conditional classes/regions.",
            "Value": f"{raw['value']:.4f}"
        }


class RankAlignmentMetric(BaseMetric):
    def __init__(self):
        super().__init__("rank_alignment", "score_rank_alignment")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["rank_alignment"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Jaccard similarity of top alpha features between global and conditional importances.",
            "Value": f"{raw['value']:.4f}"
        }


class XAIEaseScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("xai_ease_score", "score_xai_ease_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_global_xai(ctx)
        return {"value": float(m["xai_ease_score"])}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Measures ease of explaining predictions using partial dependence plot similarity.",
            "Value": f"{raw['value']:.4f}"
        }


# class FluctuationRatioMetric(BaseMetric):
#     def __init__(self):
#         super().__init__("fluctuation_ratio", "score_fluctuation_ratio")

#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute_global_xai(ctx)
#         return {"value": float(m["fluctuation_ratio"])}

#     def build_properties(self, raw: dict) -> dict:
#         return {
#             "Metric Description": "Average number of sign changes in the derivatives of individual conditional expectation curves.",
#             "Value": f"{raw['value']:.4f}"
#         }

# =============================================================================
# Local Metrics Wrappers
# =============================================================================

# class RankConsistencyMetric(BaseMetric):
#     def __init__(self): super().__init__("rank_consistency", "score_rank_consistency")
#     def compute(self, ctx: EvaluationContext) -> dict: return {"value": _get_or_compute_local(ctx)["rank_consistency"]}
#     def build_properties(self, raw: dict) -> dict: return {"Metric Description": "Rank consistency across local explanations.", "Value": f"{raw['value']:.6f}"}

# class ImportanceStabilityMetric(BaseMetric):
#     def __init__(self): super().__init__("importance_stability", "score_importance_stability")
#     def compute(self, ctx: EvaluationContext) -> dict: return {"value": _get_or_compute_local(ctx)["importance_stability"]}
#     def build_properties(self, raw: dict) -> dict: return {"Metric Description": "Stability of local feature importances.", "Value": f"{raw['value']:.6f}"}


# =============================================================================
# Surrogate Metrics Wrappers
# =============================================================================

# class MSEDegradationMetric(BaseMetric):
#     def __init__(self): super().__init__("mse_degradation", "score_mse_degradation")
#     def compute(self, ctx: EvaluationContext) -> dict: return {"value": _get_or_compute_surrogate(ctx)["mse_degradation"]}
#     def build_properties(self, raw: dict) -> dict: return {"Metric Description": "Degradation in MSE using surrogate.", "Value": f"{raw['value']:.6f}"}

# class SurrogateFidelityMetric(BaseMetric):
#     def __init__(self): super().__init__("surrogate_fidelity", "score_surrogate_fidelity")
#     def compute(self, ctx: EvaluationContext) -> dict: return {"value": _get_or_compute_surrogate(ctx)["surrogate_fidelity"]}
#     def build_properties(self, raw: dict) -> dict: return {"Metric Description": "Fidelity of surrogate vs original model.", "Value": f"{raw['value']:.6f}"}

# class SurrogateFeatureStabilityMetric(BaseMetric):
#     def __init__(self): super().__init__("surrogate_feature_stability", "score_surrogate_feature_stability")
#     def compute(self, ctx: EvaluationContext) -> dict: return {"value": _get_or_compute_surrogate(ctx)["surrogate_feature_stability"]}
#     def build_properties(self, raw: dict) -> dict: return {"Metric Description": "Stability of features in surrogate.", "Value": f"{raw['value']:.6f}"}


# =============================================================================
# Tree Structural Metrics Wrappers
# =============================================================================

class WeightedAverageDepthMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_depth", "score_weighted_average_depth")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_depth(getattr(ctx.model, "tree_", None))
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Average depth of a tree weighted by samples.", "Value": f"{raw['value']:.6f}"}

class WeightedAverageExplainabilityScoreMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_explainability_score", "score_weighted_average_explainability")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_explainability_score(getattr(ctx.model, "tree_", None))
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Average explainability score of a tree.", "Value": f"{raw['value']:.6f}"}

class WeightedTreeGiniMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_tree_gini", "score_weighted_tree_gini")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_tree_gini(getattr(ctx.model, "tree_", None))
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Weighted Gini index for the tree (WGNI).", "Value": f"{raw['value']:.6f}"}

class TreeDepthVarianceMetric(BaseMetric):
    def __init__(self): super().__init__("tree_depth_variance", "score_tree_depth_variance")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_depth_variance(getattr(ctx.model, "tree_", None))
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Variance of the depths of the leaves.", "Value": f"{raw['value']:.6f}"}

class TreeNumberOfRulesMetric(BaseMetric):
    def __init__(self): super().__init__("tree_number_of_rules", "score_tree_number_of_rules")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_number_of_rules(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Number of rules in the surrogate model.", "Value": f"{raw['value']:.0f}"}

class TreeNumberOfFeaturesMetric(BaseMetric):
    def __init__(self): super().__init__("tree_number_of_features", "score_tree_number_of_features")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_number_of_features(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Number of features actively used.", "Value": f"{raw['value']:.0f}"}

# =============================================================================
# Custom Ensemble XAI Consistency Wrapper
# =============================================================================

_XAI_CONSISTENCY_KEY = "xai_ensemble_consistency"

def _get_or_compute_xai_consistency(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_XAI_CONSISTENCY_KEY)
    if isinstance(cached, dict):
        return cached

    # Aquí usamos su propia clave de error
    if _CONSISTENCY_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CONSISTENCY_ERROR_KEY]))

    # Determinamos si es clasificación o regresión basado en el modelo
    mode = 'classification' if hasattr(ctx.model, 'predict_proba') else 'regression'
    
    # Parámetro 'k' por defecto a 5 si no se pasa por extras
    k = ctx.extras.get("xai_consistency_k", 5)

    try:
        # Usamos X_test y y_test para la consistencia
        metrics = core.xai_consistency(
            model=ctx.model, 
            X=ctx.X_test, 
            y=ctx.y_test, 
            k=k, 
            mode=mode
        )
        ctx.extras[_XAI_CONSISTENCY_KEY] = metrics
        return metrics
    except Exception as exc:
        ctx.extras[_CONSISTENCY_ERROR_KEY] = str(exc)
        raise


class XAIConsistencyScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("xai_consistency_score", "score_xai_consistency")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_xai_consistency(ctx)
        return {
            "value": float(m["value"]), 
            "matrix": m["consistency_matrix"],
            "top_k_details": m["top_k_details"]
        }

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Global XAI Consistency Score. Average Jaccard similarity of top-K features across LIME, SHAP, PDP, PFI, and Surrogate.",
            "XAI Consistency Score": f"{raw['value']:.4f}" if not np.isnan(raw.get("value", np.nan)) else "N/A",
            "Top-K Rankings Details": raw.get("top_k_details", "No details available"),
            "Consistency Matrix": raw.get("matrix", "No matrix available")
        }
    
