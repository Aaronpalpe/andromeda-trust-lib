"""Explainability Module

Implements SHAP-based explainability evaluation.

Public API:
  - ExplainabilityPillar
"""

from .explainability import ExplainabilityPillar
from .explainability_metrics_core import (
    compute_shap_based_metrics, 
    compute_algorithm_class, 
    compute_correlated_features, 
    compute_model_size, 
    compute_feature_relevance, 
    compute_performance_difference, 
    compute_number_of_rules, 
    compute_average_rule_length, 
    compute_rule_stats, 
    compute_tree_depth, 
    compute_interaction_strength, 
    compute_faithfulness_metric, 
    compute_monotonicity_metric, 
    compute_shapley_corr, 
    compute_roar, 
    compute_infidelity, 
    compute_global_explainability_metrics, 
    compute_local_explainability_metrics, 
    compute_surrogate_explainability_metrics, 
    compute_weighted_average_depth, 
    compute_weighted_average_explainability_score, 
    compute_weighted_tree_gini, 
    compute_tree_depth_variance, 
    compute_tree_number_of_rules, 
    compute_tree_number_of_features, 
    compute_xai_consistency,
)

__all__ = [
    "ExplainabilityPillar",
    "compute_shap_based_metrics",
    "compute_algorithm_class",
    "compute_correlated_features",
    "compute_model_size",
    "compute_feature_relevance",
    "compute_performance_difference",
    "compute_number_of_rules",
    "compute_average_rule_length",
    "compute_rule_stats",
    "compute_tree_depth",
    "compute_interaction_strength",
    "compute_faithfulness_metric",
    "compute_monotonicity_metric",
    "compute_shapley_corr",
    "compute_roar",
    "compute_infidelity",
    "compute_global_explainability_metrics",
    "compute_local_explainability_metrics",
    "compute_surrogate_explainability_metrics",
    "compute_weighted_average_depth",
    "compute_weighted_average_explainability_score",
    "compute_weighted_tree_gini",
    "compute_tree_depth_variance",
    "compute_tree_number_of_rules",
    "compute_tree_number_of_features",
    "compute_xai_consistency",
]


