from __future__ import annotations
from typing import Any

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

    # If it already failed, don't retry
    if _EXPL_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_EXPL_ERROR_KEY]))

    params = ctx.extras.get(_EXPL_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}

    # (Optional) quick sklearn validation
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
        model_type = ctx.factsheet.get("general", {}).get("model_type", {}).get("value")
        # Si es None o null 
        if model_type is None:
            raise ValueError("Model type not found in factsheet under general.model_type.value.")
        return core.algorithm_class(ctx.model, model_type=model_type)

    # def custom_score(self, raw: dict):
    #     return raw.get("value")

    def compute_score(self, raw: Dict[str, Any], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key, {})
            .get("mappings", {})
            .get("value", {})
        )

        value = mappings.get(raw["model_type"])
        if value is None:
            raise ValueError(f"No score mapping found for model type '{raw['model_type']}' in config under '{self.score_config_key}.mappings.value'.")
        return value

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Score assigned based on model class type.",
            #"Value": f"{raw['value']:.6f}",
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
            high_cor=ctx.extras.get("high_cor")
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
        params = ctx.extras.get("explainability_params", {})
        threshold = params.get("threshold_outlier", 0.03)
        if threshold is None:
            threshold = 0.03

        # Try model's native feature importances first
        if hasattr(ctx.model, "feature_importances_") or hasattr(ctx.model, "coef_"):
            return core.feature_relevance(
                model=ctx.model,
                X_train=ctx.X_train,
                y_train=ctx.y_train,
                threshold_outlier=threshold
            )

        # Fallback to SHAP importances if available
        global_importances = ctx.extras.get("global_importances")
        if global_importances:
            importance = np.array(list(global_importances.values()))
            irrelevant_features = np.sum(importance <= threshold)
            pct_irrelevant = irrelevant_features / len(importance)
            return {
                "value": float(pct_irrelevant),
                "n_outliers": int(irrelevant_features),
                "importances": importance.tolist(),
            }

        raise ValueError("Model does not provide feature importances and SHAP importances are not available.")

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
# Perturbation & Correlation Metrics Wrappers (Xplique)
# =============================================================================

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
_GLOBAL_ERROR_KEY = "xai_global_error"
_CONSISTENCY_ERROR_KEY = "xai_consistency_error"

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

        # alpha_score
        try:
            if not global_vals or all(v == 0 for v in global_vals):
                metrics["alpha_score"] = 2.5
            else:
                metrics["alpha_score"] = core.alpha_score(global_vals)
        except Exception as e:
            metrics["alpha_score"] = 2.5

        # spread_ratio
        try:
            if not global_vals or all(v == 0 for v in global_vals):
                metrics["spread_ratio"] = 2.5
            else:
                metrics["spread_ratio"] = core.spread_ratio(global_vals)
        except Exception as e:
            metrics["spread_ratio"] = 2.5

        # spread_divergence
        try:
            if not global_vals or all(v == 0 for v in global_vals):
                metrics["spread_divergence"] = 2.5
            else:
                metrics["spread_divergence"] = core.spread_divergence(global_vals)
        except Exception as e:
            metrics["spread_divergence"] = 2.5

        # position_parity
        try:
            if not cond_ranked or not global_ranked:
                metrics["position_parity"] = 2.5
            else:
                metrics["position_parity"] = core.position_parity(cond_ranked, global_ranked)
        except Exception as e:
            metrics["position_parity"] = 2.5

        # rank_alignment
        try:
            if not cond_imps or not global_imps:
                metrics["rank_alignment"] = 2.5
            else:
                metrics["rank_alignment"] = core.rank_alignment(cond_imps, global_imps)
        except Exception as e:
            metrics["rank_alignment"] = 2.5

        # xai_ease_score
        try:
            if not pdp_avgs or not global_ranked:
                metrics["xai_ease_score"] = 2.5
            else:
                metrics["xai_ease_score"] = core.xai_ease_score(pdp_avgs, global_ranked)
        except Exception as e:
            metrics["xai_ease_score"] = 2.5

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

# =============================================================================
# Tree Structural Metrics Wrappers
# =============================================================================

class WeightedAverageDepthMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_depth", "score_weighted_average_depth")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_depth(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Average depth of a tree weighted by samples.", "Value": f"{raw['value']:.6f}"}

class WeightedAverageExplainabilityScoreMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_explainability_score", "score_weighted_average_explainability")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_explainability_score(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Average explainability score of a tree.", "Value": f"{raw['value']:.6f}"}

class WeightedTreeGiniMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_tree_gini", "score_weighted_tree_gini")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_tree_gini(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {"Metric Description": "Weighted Gini index for the tree (WGNI).", "Value": f"{raw['value']:.6f}"}

class TreeDepthVarianceMetric(BaseMetric):
    def __init__(self): super().__init__("tree_depth_variance", "score_tree_depth_variance")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_depth_variance(ctx.model)
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
    k = ctx.extras.get("xai_consistency_k", 3)

    try:
        # Usamos X_test y y_test para la consistencia
        metrics = core.xai_consistency(
            model=ctx.model, 
            shap_values=ctx.extras.get("feature_weights"),
            X=ctx.X_test, 
            y=ctx.y_test, 
            k=k, 
            mode=mode,
            random_state=ctx.extras.get(_EXPL_PARAMS_KEY, {}).get("seed", 42)
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
    
