"""Explainability Module

Implements SHAP-based explainability evaluation.

Public API:
  - ExplainabilityPillar
"""

from .explainability import ExplainabilityPillar
from .explainability_metrics_core import (
    shap_based_metrics, #interaction_strength, 
    algorithm_class, 
    correlated_features, 
    model_size, 
    feature_relevance, 
    # performance_difference, 
    number_of_rules, 
    average_rule_length, 
    #rule_stats, 
    tree_depth, 
    faithfulness_metric, 
    monotonicity_metric, 
    # shapley_corr, 
    # roar, 
    infidelity, 
    # global_explainability_metrics, 
    # local_explainability_metrics, 
    # surrogate_explainability_metrics, 
    alpha_score,
    xai_ease_score,
    position_parity,
    rank_alignment,
    spread_ratio,
    spread_divergence,
    weighted_average_depth, 
    weighted_average_explainability_score, 
    weighted_tree_gini, 
    tree_depth_variance, 
    tree_number_of_features, 
    xai_consistency,
)

__all__ = [
    "ExplainabilityPillar",
    "shap_based_metrics", #"interaction_strength",
    "algorithm_class",
    "correlated_features",
    "model_size",
    "feature_relevance",
    # "performance_difference",
    "number_of_rules",
    "average_rule_length",
    "rule_stats",
    "tree_depth",
    "faithfulness_metric",
    "monotonicity_metric",
    # "shapley_corr",
    # "roar",
    "infidelity",
    #"global_explainability_metrics",
    # "local_explainability_metrics",
    # "surrogate_explainability_metrics",
    "alpha_score",
    "xai_ease_score",
    "position_parity",
    "rank_alignment",
    "spread_ratio",
    "spread_divergence",
    "weighted_average_depth",
    "weighted_average_explainability_score",
    "weighted_tree_gini",
    "tree_depth_variance",
    "tree_number_of_features",
    "xai_consistency",
]


