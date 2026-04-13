from __future__ import annotations
from typing import Any, Dict

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

    params = ctx.extras.get(_EXPL_PARAMS_KEY)
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
            n_samples=int(params.get("n_samples")),
            shap_threshold=float(params.get("shap_threshold")),
            top_k=int(params.get("top_k")),
            seed=int(params.get("seed")),
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
            "Metric Description": "Fraction of features whose SHAP importance is above the configured threshold.",
            "Depends on": "Model and Test Data",
            "Formula": "Sparsity = (# features with |importance| > threshold) / (# features)",
            "N Features": int(raw.get("n_features")),
            "Sample Size": int(raw.get("sample_size")),
            "SHAP Threshold": float(raw.get("shap_threshold")),
            "Explainer": raw.get("explainer"),
            "Sparsity": float(raw["value"]),
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
            "Metric Description": "Normalized entropy of the global SHAP importance distribution across features. Lower values indicate that importance is concentrated in fewer features, while higher values suggest a more even distribution of importance.",
            "Depends on": "Model and Test Data",
            "Formula": "Feature Entropy = -sum_i p_i log(p_i) / log(n_features)",
            "N Features": int(raw.get("n_features")),
            "Sample Size": int(raw.get("sample_size")),
            "Explainer": raw.get("explainer"),
            "Global Feature Importances": raw.get("global_imps_array").tolist(),
            "Feature Entropy": float(raw["value"]),
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
            "Metric Description": "Fraction of total global SHAP importance captured by the top-k most important features. Higher values indicate that a small subset of features accounts for most of the importance.",
            "Depends on": "Model and Test Data",
            "Formula": "Top-K Concentration = sum(top-k global importances) / sum(all global importances)",
            "Top K": int(raw.get("top_k")),
            "N Features": int(raw.get("n_features")),
            "Sample Size": int(raw.get("sample_size")),
            "Explainer": raw.get("explainer"),
            "Global Feature Importances": raw.get("global_imps_array").tolist(),
            "Top-K Concentration": float(raw["value"]),
        }

# class InteractionStrengthMetric(BaseMetric):
#     def __init__(self): super().__init__("interaction_strength", "score_interaction_strength")
#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute(ctx)
#         return {"value": float(m["interaction_strength"]), **m}
#     def build_properties(self, raw: dict) -> dict: 
#         return {"Metric Description": "Proportion of SHAP importance coming from feature interactions.", 
#                 "Value": f"{raw['value']:.6f}",
#                 "N Features": int(raw.get("n_features")),
#                 "Sample Size": int(raw.get("sample_size")),
#                 "Explainer": raw.get("explainer"),
#                 }
    

class AlgorithmClassMetric(BaseMetric):
    def __init__(self):
        super().__init__("algorithm_class", "score_algorithm_class")

    def compute(self, ctx: EvaluationContext) -> dict:
        model_type = ctx.factsheet.get("general").get("model_type").get("value")
        # If model_type is missing in the factsheet, fail explicitly.
        if model_type is None:
            raise ValueError("Model type not found in factsheet under general.model_type.value.")
        return core.algorithm_class(ctx.model, model_type=model_type)

    # def custom_score(self, raw: dict):
    #     return raw.get("value")

    def compute_score(self, raw: Dict[str, Any], config: Dict[str, Any]) -> float:
        mappings = (
            config.get(self.score_config_key)
            .get("mappings")
            .get("value")
        )

        value = mappings.get(raw["model_type"])
        if value is None:
            raise ValueError(f"No score mapping found for model type '{raw['model_type']}' in config under '{self.score_config_key}.mappings.value'.")
        return value

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Identifies the model class using the provided model type or inferred model class name.",
            "Depends on": "Model Metadata",
            "Formula": "Algorithm Class Score is assigned via mapping by model_type",
            "Model Type": raw.get("model_type"),
        }


class CorrelatedFeaturesMetric(BaseMetric):
    def __init__(self):
        super().__init__("correlated_features", "score_correlated_features")

    def compute(self, ctx: EvaluationContext) -> dict:
        params = ctx.extras.get("explainability_params")
        return core.correlated_features(
            X_train=ctx.X_train,
            X_test=ctx.X_test,
            high_cor=params.get("high_cor")
        )

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Percentage of features highly correlated with at least one other feature in combined train and test data.",
            "Depends on": "Training and Test Data",
            "Formula": "Correlated Features = (# features with any |corr| >= threshold) / (# features)",
            "Highly correlated features": raw.get("highly_correlated_features"),
            "High correlation threshold": f"{raw['threshold']:.2f}",
            "Percentage of highly correlated features": f"{raw['value']:.6f}",
        }


class ModelSizeMetric(BaseMetric):
    def __init__(self):
        super().__init__("model_size", "score_model_size")

    def compute(self, ctx: EvaluationContext) -> dict:
        return core.model_size(ctx.X_train)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Number of input features in the training data.",
            "Depends on": "Training Data",
            "Formula": "Model Size = Number of input features",
            "Number of Features": f"{raw['value']:.6f}",
        }


class FeatureRelevanceMetric(BaseMetric):
    def __init__(self):
        super().__init__("feature_relevance", "score_feature_relevance")

    def compute(self, ctx: EvaluationContext) -> dict:
        params = ctx.extras.get("explainability_params")
        threshold = params.get("threshold_outlier")
        if threshold is None:
            threshold = 0.03

        global_importances = ctx.extras.get("global_importances")
        if global_importances:
            importance = np.array(list(global_importances.values()))
        if hasattr(ctx.model, "feature_importances_") or hasattr(ctx.model, "coef_") or global_importances is not None:
            return core.feature_relevance(
                model=ctx.model,
                global_imps_array=importance if global_importances else None,
                threshold_outlier=threshold
            )

        raise ValueError("Model does not provide feature importances and SHAP importances are not available.")

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Percentage of features considered irrelevant because their importance is below the configured threshold.",
            "Depends on": "Model or Global Feature Importances",
            "Formula": "Feature Relevance = (# features with importance > threshold) / (# features)",
            "Threshold for irrelevance": f"{raw.get('threshold'):.2f}",
            "Number of irrelevant features": raw.get("n_outliers"),
            "Importances": raw.get("importances"),
            "Percentage of features whose importance contributes is greater than threshold": f"{raw['value']:.6f}",
        }



class AlphaImportanceScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("alpha_score", "score_alpha_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        global_imps = ctx.extras.get("global_importances")
        global_vals = list(global_imps.values())
        return core.alpha_score(global_vals)
    
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Fraction of top features needed to reach the alpha share of total feature importance.",
            "Depends on": "Global Feature Importances",
            "Formula": "Alpha Score = (# top features needed to reach alpha cumulative importance) / (# features)",
            "Feature Importances": raw.get("feature_importances"),
            "Alpha": f"{raw.get('alpha'):.2f}",
            "Total Features": int(raw.get("n_features")),
            "Top Features for Alpha": int(raw.get("n_top_features")),
            "Alpha Score": f"{raw['value']:.4f}",
        }

class SpreadRatioMetric(BaseMetric):
    def __init__(self):
        super().__init__("spread_ratio", "score_spread_ratio")

    def compute(self, ctx: EvaluationContext) -> dict:
        global_imps = ctx.extras.get("global_importances")
        global_vals = list(global_imps.values())
        return core.spread_ratio(global_vals)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Evenness of the feature importance distribution compared against a uniform distribution.",
            "Depends on": "Global Feature Importances",
            "Formula": "Spread Ratio = min(normalized importances) / max(normalized importances)",
            "Feature Importances": raw.get("feature_importances"),
            "Spread Ratio": f"{raw['value']:.4f}",
        }

class SpreadDivergenceMetric(BaseMetric):
    def __init__(self):
        super().__init__("spread_divergence", "score_spread_divergence")

    def compute(self, ctx: EvaluationContext) -> dict:
        global_imps = ctx.extras.get("global_importances")
        global_vals = list(global_imps.values())
        return core.spread_divergence(global_vals)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Jensen-Shannon divergence between the feature importance distribution and an even distribution.",
            "Depends on": "Global Feature Importances",
            "Formula": "Spread Divergence = JSD(importance_distribution, uniform_distribution)",
            "Feature Importances": raw.get("feature_importances"),
            "Spread Divergence": f"{raw['value']:.4f}",
        }

class PositionParityMetric(BaseMetric):
    def __init__(self):
        super().__init__("position_parity", "score_position_parity")

    def compute(self, ctx: EvaluationContext) -> dict:
        global_imps = ctx.extras.get("global_importances")
        global_ranked = sorted(global_imps.keys(), key=lambda k: global_imps[k], reverse=True)
        cond_imps = ctx.extras.get("conditional_importances")
        cond_ranked = {
            g: sorted(imps.keys(), key=lambda k: imps[k], reverse=True)
            for g, imps in cond_imps.items()
        }
        return core.position_parity(cond_ranked, global_ranked)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Average cumulative alignment between conditional feature rankings and the global feature ranking.",
            "Depends on": "Global and Conditional Feature Rankings",
            "Formula": "Position Parity = mean over groups of cumulative positional overlap with global ranking",
            "Conditional Rankings": raw.get("conditional_rankings"),
            "Global Ranking": raw.get("global_ranking"),
            "Conditional Position Parity": raw.get("conditional_position_parity"),
            "Position Parity": f"{raw['value']:.4f}",
        }

class RankAlignmentMetric(BaseMetric):
    def __init__(self):
        super().__init__("rank_alignment", "score_rank_alignment")

    def compute(self, ctx: EvaluationContext) -> dict:
        global_imps = ctx.extras.get("global_importances")
        cond_imps = ctx.extras.get("conditional_importances")
        return core.rank_alignment(cond_imps, global_imps)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Jaccard similarity between top-alpha features from conditional and global importance distributions.",
            "Depends on": "Global and Conditional Feature Importances",
            "Formula": "Rank Alignment = mean_g Jaccard(Top(global), Top(group g))",
            "Conditional Importances": raw.get("conditional_importances"),
            "Global Importances": raw.get("global_importances"),
            "Top Global Features": raw.get("top_global_features"),
            "Top Conditional Features": raw.get("top_conditional_features"),
            "Rank Alignment": f"{raw['value']:.4f}",
        }

class XAIEaseScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("xai_ease_score", "score_xai_ease_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        pdp_avgs = ctx.extras.get("pdp_averages")
        global_imps = ctx.extras.get("global_importances")
        global_ranked = sorted(global_imps.keys(), key=lambda k: global_imps[k], reverse=True)
        top_k_features = global_ranked[:5] 
        return core.xai_ease_score(pdp_avgs, top_k_features)

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Ease of interpreting top features based on similarity of PDP curve tangents across sections.",
            "Depends on": "PDP Averages and Global Ranked Features",
            "Formula": "XAI Ease Score = mean pairwise similarity of PDP tangent profiles over top-ranked features",
            "PDP Averages": raw.get("pdp_averages"),
            "Global Ranked Features": raw.get("global_ranked_features"),
            "XAI Ease Score": f"{raw['value']:.4f}",
        }
    

# =============================================================================
# Instance-Based Fidelity Metrics Wrappers
# =============================================================================

class FaithfulnessMetric(BaseMetric):
    def __init__(self): super().__init__("faithfulness", "score_faithfulness")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        # Use local feature weights and a baseline vector; aggregate over first test instances.
        X_np = np.asarray(ctx.X_test)
        base = ctx.extras.get("base_values")
        local_importances = ctx.extras.get("feature_weights") 

        if local_importances is None:
            raise ValueError("Missing 'feature_weights' in ctx.extras for FaithfulnessMetric.")

        scores = []
        # Evaluate first N instances for a stable aggregate value.
        limit = min(len(X_np), 10) 
        for i in range(limit):
            res = core.faithfulness_metric(ctx.model, X_np[i], local_importances[i], base)
            scores.append(res["value"])

        return {"value": float(np.mean(scores))}
        
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Correlation between feature importance weights and the prediction change when each feature is replaced by a baseline value.",
            "Depends on": "Model, Test Data, Local Feature Weights, and Base Values",
            "Formula": "Faithfulness = corr(feature importance, prediction change after feature removal)",
            "Faithfulness": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
        }

class MonotonicityMetric(BaseMetric):
    def __init__(self): super().__init__("monotonicity", "score_monotonicity")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        X_np = np.asarray(ctx.X_test)
        base = ctx.extras.get("base_values")
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
            "Metric Description": "Proportion of instances where prediction confidence increases monotonically as features are added by importance.",
            "Depends on": "Model, Test Data, Local Feature Weights, and Base Values",
            "Formula": "Monotonicity = (# instances with monotonic confidence progression) / (# instances)",
            "Monotonicity": f"{raw['value']:.4f}" if not np.isnan(raw['value']) else "N/A"
        }

# =============================================================================
# Perturbation & Correlation Metrics Wrappers (Xplique)
# =============================================================================

class InfidelityMetric(BaseMetric):
    def __init__(self): 
        super().__init__("infidelity", "score_infidelity")
        
    def compute(self, ctx: EvaluationContext) -> dict:
        feature_weights = ctx.extras.get("feature_weights")
        
        if feature_weights is None:
            raise ValueError("Missing 'feature_weights' in ctx.extras for InfidelityMetric.")
            
        return core.infidelity(
            model=ctx.model, 
            X_test=ctx.X_test, 
            feature_weights=feature_weights
        )
        
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Measures whether perturbations on important features produce proportional changes in model output.",
            "Depends on": "Model, Test Data, and Local Feature Weights",
            "Formula": "Infidelity = expected discrepancy between attribution-weighted perturbation and model output change",
            "Infidelity Score": f"{raw['value']:.6f}" if not np.isnan(raw['value']) else "N/A"
        }


 # =============================================================================
# Additional Explainability & Surrogate Metrics Wrappers
# =============================================================================

class NumberOfRulesMetric(BaseMetric):
    def __init__(self): super().__init__("number_of_rules", "score_number_of_rules")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.number_of_rules(ctx.model) # return core.number_of_rules(getattr(ctx.model, "tree_", ctx.model))
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Number of rules in a tree or rule-based model (mean across trees for ensembles).",
            "Depends on": "Tree or Rule-Based Model",
            "Value": f"{raw['value']:.0f}"
        }

class AverageRuleLengthMetric(BaseMetric):
    def __init__(self): super().__init__("average_rule_length", "score_average_rule_length")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.average_rule_length(ctx.model)
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Average rule length or path depth in a tree or rule-based model.",
            "Depends on": "Tree or Rule-Based Model",
            "Value": f"{raw['value']:.4f}"
        }

class TreeDepthMetric(BaseMetric):
    def __init__(self): super().__init__("tree_depth", "score_tree_depth")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_depth(ctx.model)
    def build_properties(self, raw: dict) -> dict: 
        return {
            "Metric Description": "Maximum tree depth (mean depth for tree ensembles).",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.0f}"
        }
       


# =============================================================================
# Tree Structural Metrics Wrappers
# =============================================================================

class WeightedAverageDepthMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_depth", "score_weighted_average_depth")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_depth(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Sample-weighted average depth of tree leaf paths.",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.6f}",
            "Depth*Weight List": raw.get("list")
        }

class WeightedAverageExplainabilityScoreMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_average_explainability_score", "score_weighted_average_explainability")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_average_explainability_score(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Sample-weighted average explainability score based on the number of unique cuts per path.",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.6f}",
            "Explainability*Weight List": raw.get("list")
        }

class WeightedTreeGiniMetric(BaseMetric):
    def __init__(self): super().__init__("weighted_tree_gini", "score_weighted_tree_gini")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.weighted_tree_gini(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Weighted impurity score of leaf nodes across the tree.",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.6f}"
        }

class TreeDepthVarianceMetric(BaseMetric):
    def __init__(self): super().__init__("tree_depth_variance", "score_tree_depth_variance")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_depth_variance(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Variance of leaf depths in the tree structure.",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.6f}",
            "Leaf Depths": raw.get("leaf_depths")
        }

class TreeNumberOfFeaturesMetric(BaseMetric):
    def __init__(self): super().__init__("tree_number_of_features", "score_tree_number_of_features")
    def compute(self, ctx: EvaluationContext) -> dict:
        return core.tree_number_of_features(ctx.model)
    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Number of distinct features actively used by the tree.",
            "Depends on": "Tree-Based Model",
            "Value": f"{raw['value']:.0f}"
        }

# =============================================================================
# Custom Ensemble XAI Consistency Wrapper
# =============================================================================

_XAI_CONSISTENCY_KEY = "xai_ensemble_consistency"
_CONSISTENCY_ERROR_KEY = "xai_ensemble_consistency_error"

def _get_or_compute_xai_consistency(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_XAI_CONSISTENCY_KEY)
    if isinstance(cached, dict):
        return cached

    # Use a dedicated cache error key for this metric family.
    if _CONSISTENCY_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CONSISTENCY_ERROR_KEY]))

    # Detect mode from model capabilities.
    mode = 'classification' if hasattr(ctx.model, 'predict_proba') else 'regression'
    
    params = ctx.extras.get("explainability_params")

    try:
        metrics = core.xai_consistency(
            model=ctx.model, 
            global_importances=ctx.extras.get("global_importances"),
            pdp_std=ctx.extras.get("pdp_std"),
            X=ctx.X_test, 
            k=params.get("top_k"),
            mode=mode,
            random_state=params.get("seed")
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
            "Metric Description": "Average pairwise Jaccard similarity of top-k features across LIME, SHAP, and PDP importance rankings.",
            "Depends on": "Model, Test Data, SHAP Global Importances, and PDP Importances",
            "Formula": "XAI Consistency = mean pairwise Jaccard similarity among top-k feature sets from multiple XAI methods",
            "Top-K Rankings Details": raw.get("top_k_details"),
            "Consistency Matrix": raw.get("matrix"),
            "XAI Consistency Score": f"{raw['value']:.4f}" if not np.isnan(raw.get("value")) else "N/A",
        }
    
