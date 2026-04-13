from __future__ import annotations

from trust_library.base_metric import BaseMetric
from trust_library.utils import EvaluationContext
from typing import Dict, Any

from . import robustness_metrics_core as core

# ============================================================
# CACHE SYSTEM
# ============================================================
# Generic cache key builder for different attack types
def _get_cache_keys(attack_type: str) -> tuple[str, str, str]:
    """
    Generate cache keys for a given attack type.
    
    Args:
        attack_type: One of 'hsj', 'fgm', 'cw', 'df', 'clever', 'clique', 'loss', 'confidence', 'ece'
    
    Returns:
        (metrics_key, params_key, error_key)
    """
    prefix = f"robustness_{attack_type}"
    return (f"{prefix}_metrics", f"{prefix}_params", f"{prefix}_error")


_GENERAL_KEY = "robustness_art_metrics"
_GENERAL_PARAMS_KEY = "robustness_params"
_GENERAL_ERROR_KEY = "robustness_error"

# Attack-specific cache keys
_FGM_KEY, _FGM_PARAMS_KEY, _FGM_ERROR_KEY = _get_cache_keys("fgm")
_CW_KEY, _CW_PARAMS_KEY, _CW_ERROR_KEY = _get_cache_keys("cw")
_DF_KEY, _DF_PARAMS_KEY, _DF_ERROR_KEY = _get_cache_keys("df")

# Other robustness metrics
_CLIQUE_KEY, _CLIQUE_PARAMS_KEY, _CLIQUE_ERROR_KEY = _get_cache_keys("clique")
_CLEVER_KEY, _CLEVER_PARAMS_KEY, _CLEVER_ERROR_KEY = _get_cache_keys("clever")
_LOSS_KEY, _LOSS_PARAMS_KEY, _LOSS_ERROR_KEY = _get_cache_keys("loss")
_CONF_KEY, _CONF_PARAMS_KEY, _CONF_ERROR_KEY = _get_cache_keys("confidence")

# Calibration metrics
_ECE_KEY, _ECE_PARAMS_KEY, _ECE_ERROR_KEY = _get_cache_keys("ece")

# Ensemble metrics
_ENSEMBLE_KEY = "robustness_ensemble_metrics"
_ENSEMBLE_ERROR_KEY = "robustness_ensemble_error"

# ============================================================
# CACHED COMPUTES - Generic Attack Framework
# ============================================================

def _get_or_compute_attack(
    ctx: EvaluationContext,
    attack_type: str,
    compute_fn,
    metrics_key: str | None = None,
    error_key: str | None = None,
    **compute_kwargs
) -> dict:
    """
    Generic function to compute and cache attack metrics.
    
    Args:
        ctx: EvaluationContext
        attack_type: Attack type identifier (e.g., 'hsj', 'fgm', 'cw', 'df')
        compute_fn: Function from robustness_metrics_core (e.g., core.hopskipjump_metrics)
        **compute_kwargs: Keyword arguments to pass to compute_fn
    
    Returns:
        Dictionary with computed metrics
    
    Raises:
        RuntimeError: If metric computation fails or already failed previously
    """
    if metrics_key is None or error_key is None:
        default_metrics_key, _, default_error_key = _get_cache_keys(attack_type)
        metrics_key = metrics_key or default_metrics_key
        error_key = error_key or default_error_key
    
    # Check if already computed
    cached = ctx.extras.get(metrics_key)
    if isinstance(cached, dict):
        return cached
    
    # Check if previously failed
    if error_key in ctx.extras:
        raise RuntimeError(str(ctx.extras[error_key]))
    
    try:
        metrics = compute_fn(**compute_kwargs)
    except Exception as exc:
        ctx.extras[error_key] = str(exc)
        raise
    
    ctx.extras[metrics_key] = metrics
    return metrics


def _get_or_compute_hsj(ctx: EvaluationContext) -> dict:
    """Compute and cache HopSkipJump attack metrics."""
    params = ctx.extras.get(_GENERAL_PARAMS_KEY) or {}
    if not isinstance(params, dict):
        params = {}

    if not hasattr(ctx.model, "predict"):
        error_msg = "Model must implement predict() to run HopSkipJump."
        ctx.extras[_GENERAL_ERROR_KEY] = error_msg
        raise RuntimeError(error_msg)

    return _get_or_compute_attack(
        ctx,
        "hsj",
        core.hopskipjump_metrics,
        metrics_key=_GENERAL_KEY,
        error_key=_GENERAL_ERROR_KEY,
        model=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        X_train=getattr(ctx, "X_train", None),
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
        max_iter=int(params.get("max_iter")),
        max_eval=int(params.get("max_eval")),
        init_eval=int(params.get("init_eval")),
        init_size=int(params.get("init_size")),
        norm=params.get("norm"),
        beta=float(params.get("beta")),
    )


def _get_or_compute_fgm(ctx: EvaluationContext) -> dict:
    """Compute and cache Fast Gradient Method attack metrics."""
    params = ctx.extras.get(_GENERAL_PARAMS_KEY) or {}
    # art_clf = ctx.extras.get("art_classifier")

    return _get_or_compute_attack(
        ctx,
        "fgm",
        core.fgm_attack_metrics,
        art_clf=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        eps=float(params.get("eps")),
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
    )


def _get_or_compute_cw(ctx: EvaluationContext) -> dict:
    """Compute and cache Carlini-Wagner attack metrics."""
    params = ctx.extras.get(_GENERAL_PARAMS_KEY) or {}
    # art_clf = ctx.extras.get("art_classifier")

    return _get_or_compute_attack(
        ctx,
        "cw",
        core.carlini_wagner_metrics,
        art_clf=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
    )


def _get_or_compute_df(ctx: EvaluationContext) -> dict:
    """Compute and cache DeepFool attack metrics."""
    params = ctx.extras.get(_GENERAL_PARAMS_KEY) or {}
    # art_clf = ctx.extras.get("art_classifier")

    return _get_or_compute_attack(
        ctx,
        "df",
        core.deepfool_metrics,
        art_clf=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
    )


def _get_or_compute_clique(ctx: EvaluationContext) -> dict:
    """Compute and cache Clique Method metrics (tree-based models only)."""
    if _CLIQUE_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CLIQUE_ERROR_KEY]))

    params = ctx.extras.get(_CLIQUE_PARAMS_KEY) or {}
    if not isinstance(params, dict):
        params = {}

    return _get_or_compute_attack(
        ctx,
        "clique",
        core.clique_method_metrics,
        model=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        X_train=getattr(ctx, "X_train", None),
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
        eps_init=float(params.get("eps_init")),
        norm=float(params.get("norm")),
        nb_search_steps=int(params.get("nb_search_steps")),
        max_clique=int(params.get("max_clique")),
        max_level=int(params.get("max_level")),
    )


def _get_or_compute_clever(ctx: EvaluationContext) -> dict:
    """Compute and cache CLEVER score metrics (gradient-capable models only)."""
    if _CLEVER_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CLEVER_ERROR_KEY]))

    params = ctx.extras.get(_CLEVER_PARAMS_KEY)
    if not isinstance(params, dict):
        params = {}

    # classifier = ctx.extras.get("art_classifier")

    return _get_or_compute_attack(
        ctx,
        "clever",
        core.clever_score_metrics,
        classifier=ctx.model,
        x=ctx.X_test,
        n_samples=int(params.get("n_samples")),
        seed=int(params.get("seed")),
        nb_batches=int(params.get("nb_batches")),
        batch_size=int(params.get("batch_size")),
        radius=float(params.get("radius")),
        norm=int(params.get("norm")),
    )


def _get_or_compute_loss(ctx: EvaluationContext) -> dict:
    """Compute and cache Loss Sensitivity metrics (gradient-capable models only)."""
    #classifier = ctx.extras.get("art_classifier")

    return _get_or_compute_attack(
        ctx,
        "loss",
        core.loss_sensitivity_metrics,
        classifier=ctx.model,
        X_test=ctx.X_test,
    )


def _get_or_compute_confidence(ctx: EvaluationContext) -> dict:
    """Compute and cache Confidence Score metrics."""
    return _get_or_compute_attack(
        ctx,
        "confidence",
        core.confidence_score_metrics,
        model=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
    )


def _get_or_compute_ece(ctx: EvaluationContext) -> dict:
    """Compute and cache Expected Calibration Error metrics."""
    if _ECE_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_ECE_ERROR_KEY]))

    params = ctx.extras.get(_GENERAL_PARAMS_KEY) or {}

    return _get_or_compute_attack(
        ctx,
        "ece",
        core.ece_metrics,
        model=ctx.model,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        n_bins=int(params.get("n_bins")),
    )

# # ============================================================
# # PROPERTIES HELPERS
# # ============================================================

# def _extract_params(raw: dict) -> dict:
#     """Extract and validate params dictionary from raw metrics."""
#     return raw.get("params") if isinstance(raw.get("params"), dict) else {}


# def _to_percentage(value: float) -> float:
#     """Normalize fractions [0,1] and percentages [0,100] to percentage scale."""
#     v = float(value)
#     return v * 100.0 if v <= 1.0 else v


# def _get_attack_props(raw: dict) -> dict:
#     """Build common attack properties available in most robustness metrics.
    
#     Args:
#         raw: Dictionary with computed metrics
        
#     Returns:
#         Dictionary with standardized attack properties
#     """
#     params = _extract_params(raw)
    
#     props = {
#         "Attack Type": raw.get("attack"),
#         "Value": float(raw.get("value")),
#         "Clean Accuracy (%)": _to_percentage(raw.get("clean_accuracy")),
#         "Adversarial Accuracy (%)": _to_percentage(raw.get("adv_accuracy")),
#         "Accuracy Drop (%)": float(raw.get("accuracy_drop_pct (effective_robustness)", 0.0)),
#         "Robustness Ratio (adv/clean)": float(raw.get("robustness_ratio (adv/clean)", 0.0)),
#         "Adversarial Accuracy (correct-only) (%)": float(raw.get("adv_accuracy_correct_only")),
#     }
    
#     # Optional: Add ASR if present
#     if "attack_success_rate_pct (correct_only)" in raw:
#         props["ASR (%)"] = float(raw.get("attack_success_rate_pct (correct_only)", 0.0))
    
#     # Optional: Add perturbation metrics if present
#     if "mean_l2" in raw:
#         props["Mean L2 Perturbation"] = float(raw.get("mean_l2"))
#     if "mean_linf" in raw:
#         props["Mean Linf Perturbation"] = float(raw.get("mean_linf"))
#     if "er_l2_success" in raw:
#         props["ER L2 (success-only)"] = float(raw.get("er_l2_success"))
#     if "er_linf_success" in raw:
#         props["ER Linf (success-only)"] = float(raw.get("er_linf_success"))
    
#     # Optional: Add sample info if present
#     if "sample_size (n_eval)" in raw:
#         props["Sample Size (n_eval)"] = int(raw.get("sample_size (n_eval)", 0))
#     if "n_attacked" in raw:
#         props["N Attacked (correct-only)"] = int(raw.get("n_attacked"))
    
#     # Add params and note at the end
#     props["Parameters"] = params
#     if "note" in raw:
#         props["Note"] = raw.get("note")
    
#     return props

# # ============================================================
# # HSJ metric (grouped)
# # ============================================================

# class HopSkipJumpAttackMetric(BaseMetric):
#     """HopSkipJump attack summary metric including drop, ASR, adversarial accuracy and perturbation stats."""

#     def __init__(self):
#         super().__init__("er_hopskipjump_attack", "score_hopskipjump_attack")

#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute_fgm(ctx)
#         drop = float(m.get("accuracy_drop_pct (effective_robustness)", 0.0))
#         return {"value": drop, **m}

#     def build_properties(self, raw: dict) -> dict:
#         return {
#             "Metric Description": (
#                 "HopSkipJump consolidated metric: reports accuracy drop, ASR, adversarial accuracy, "
#                 "L2/Linf perturbation statistics and robustness ratio in one metric."
#             ),
#             **_get_attack_props(raw),
#         }


# # ============================================================
# # METRIC CLASSES - Individual Attacks
# # ============================================================

# class FastGradientAttackMetric(BaseMetric):
#     """Fast Gradient Method attack metrics (requires gradient-capable model)."""
    
#     def __init__(self):
#         super().__init__("fgm_accuracy_drop", "score_fgm_accuracy_drop")

#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute_fgm(ctx)
#         drop = float(m.get("accuracy_drop_pct (effective_robustness)", 0.0))
#         return {"value": drop, **m}

#     def build_properties(self, raw: dict) -> dict:
#         return {
#             "Metric Description": (
#                 "Accuracy drop (%) after Fast Gradient Method attack. "
#                 "Requires gradient-capable classifier (PyTorch/TensorFlow). Lower is better."
#             ),
#             **_get_attack_props(raw),
#         }


# class CarliniWagnerAttackMetric(BaseMetric):
#     """Carlini-Wagner (L2) attack metrics (requires gradient-capable model)."""
    
#     def __init__(self):
#         super().__init__("cw_accuracy_drop", "score_cw_accuracy_drop")

#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute_cw(ctx)
#         drop = float(m.get("accuracy_drop_pct (effective_robustness)", 0.0))
#         return {"value": drop, **m}

#     def build_properties(self, raw: dict) -> dict:
#         return {
#             "Metric Description": (
#                 "Accuracy drop (%) after Carlini-Wagner (L2) attack. "
#                 "Requires gradient-capable classifier (PyTorch/TensorFlow). Lower is better."
#             ),
#             **_get_attack_props(raw),
#         }


# class DeepFoolAttackMetric(BaseMetric):
#     """DeepFool attack metrics (requires gradient-capable model)."""
    
#     def __init__(self):
#         super().__init__("deepfool_accuracy_drop", "score_deepfool_accuracy_drop")

#     def compute(self, ctx: EvaluationContext) -> dict:
#         m = _get_or_compute_df(ctx)
#         drop = float(m.get("accuracy_drop_pct (effective_robustness)", 0.0))
#         return {"value": drop, **m}

#     def build_properties(self, raw: dict) -> dict:
#         return {
#             "Metric Description": (
#                 "Accuracy drop (%) after DeepFool attack. "
#                 "Requires gradient-capable classifier (PyTorch/TensorFlow). Lower is better."
#             ),
#             **_get_attack_props(raw),
#         }

# # ============================================================
# # ENSEMBLE ROBUSTNESS METRIC
# # ============================================================

# class EnsembleRobustnessMetric(BaseMetric):
#     """
#     Ensemble robustness metric combining multiple attack methods.
    
#     Computes robustness metrics using available attack methods (HSJ, FGM, CW, DeepFool)
#     and aggregates results to provide a comprehensive robustness assessment:
    
#     - Worst-case accuracy drop (across all attacks)
#     - Worst-case ASR (across all attacks)  
#     - Average accuracy drop
#     - Most effective attack identifier
#     - Individual attack results
    
#     This provides a more robust assessment than single-attack metrics.
#     """
    
#     def __init__(self):
#         super().__init__("ensemble_robustness", "score_ensemble_robustness")
    
#     def compute(self, ctx: EvaluationContext) -> dict:
#         """Compute ensemble robustness by running available attacks."""
#         results = {}
#         errors = {}
        
#         # Try each attack method in order of preference
#         attacks_to_run = [
#             ("HSJ", _get_or_compute_hsj, "HopSkipJump (black-box, decision-based)"),
#             ("FGM", _get_or_compute_fgm, "Fast Gradient Method (requires gradients)"),
#             ("CW", _get_or_compute_cw, "Carlini-Wagner L2 (requires gradients)"),
#             ("DeepFool", _get_or_compute_df, "DeepFool (requires gradients)"),
#         ]
        
#         for attack_name, compute_fn, attack_desc in attacks_to_run:
#             try:
#                 metrics = compute_fn(ctx)
#                 results[attack_name] = {
#                     "description": attack_desc,
#                     "clean_accuracy": float(metrics.get("clean_accuracy")),
#                     "adv_accuracy": float(metrics.get("adv_accuracy")),
#                     "accuracy_drop_pct (effective_robustness)": float(
#                         metrics.get("accuracy_drop_pct (effective_robustness)", 0.0)
#                     ),
#                     "robustness_ratio (adv/clean)": float(metrics.get("robustness_ratio (adv/clean)", 0.0)),
#                     "attack_success_rate_pct (correct_only)": float(
#                         metrics.get("attack_success_rate_pct (correct_only)", 0.0)
#                     ),
#                     "adv_accuracy_correct_only": float(metrics.get("adv_accuracy_correct_only")),
#                     "mean_l2": float(metrics.get("mean_l2")),
#                     "mean_linf": float(metrics.get("mean_linf")),
#                     "er_l2_success": float(metrics.get("er_l2_success")),
#                     "er_linf_success": float(metrics.get("er_linf_success")),
#                     "sample_size (n_eval)": int(metrics.get("sample_size (n_eval)", 0)),
#                     "n_attacked": int(metrics.get("n_attacked")),
#                     "params": metrics.get("params"),
#                     "note": metrics.get("note"),
#                 }
#             except Exception as e:
#                 # Silently skip attacks that fail (e.g., non-gradient models with FGM)
#                 errors[attack_name] = str(e)
        
#         if not results:
#             raise RuntimeError(
#                 f"Ensemble robustness requires at least one successful attack. Errors: {errors}"
#             )
        
#         # Aggregate metrics
#         accuracy_drops = [r["accuracy_drop_pct (effective_robustness)"] for r in results.values()]
#         asrs = [r["attack_success_rate_pct (correct_only)"] for r in results.values()]
#         adv_accs = [r["adv_accuracy"] for r in results.values()]
#         adv_accs_correct_only = [r["adv_accuracy_correct_only"] for r in results.values()]
#         robustness_ratios = [r["robustness_ratio (adv/clean)"] for r in results.values()]
#         er_l2_successes = [r["er_l2_success"] for r in results.values() if r["er_l2_success"] > 0]
#         er_linf_successes = [r["er_linf_success"] for r in results.values() if r["er_linf_success"] > 0]

#         worst_drop = float(max(accuracy_drops))
#         worst_asr = float(max(asrs))
#         worst_adv_acc = float(min(adv_accs))
#         worst_adv_acc_correct_only = float(min(adv_accs_correct_only))
#         worst_robustness_ratio = float(min(robustness_ratios))
#         worst_l2_success = float(min(er_l2_successes)) if er_l2_successes else 0.0
#         worst_linf_success = float(min(er_linf_successes)) if er_linf_successes else 0.0
        
#         avg_drop = float(sum(accuracy_drops) / len(accuracy_drops))
#         avg_asr = float(sum(asrs) / len(asrs))
#         avg_adv_acc = float(sum(adv_accs) / len(adv_accs))
#         avg_adv_acc_correct_only = float(sum(adv_accs_correct_only) / len(adv_accs_correct_only))
#         avg_robustness_ratio = float(sum(robustness_ratios) / len(robustness_ratios))
#         avg_l2_success = float(sum(er_l2_successes) / len(er_l2_successes)) if er_l2_successes else 0.0
#         avg_linf_success = float(sum(er_linf_successes) / len(er_linf_successes)) if er_linf_successes else 0.0

#         # Find most effective attack (highest accuracy drop)
#         most_effective = max(results.items(), key=lambda x: x[1]["accuracy_drop_pct (effective_robustness)"])[0]
        
#         return {
#             "most_effective_attack": most_effective,
#             "worst_case_accuracy_drop": worst_drop,
#             "worst_case_asr": worst_asr,
#             "worst_case_adv_accuracy": worst_adv_acc,
#             "worst_case_adv_accuracy_correct_only": worst_adv_acc_correct_only,
#             "worst_case_robustness_ratio": worst_robustness_ratio,
#             "worst_case_er_l2_success": worst_l2_success,
#             "worst_case_er_linf_success": worst_linf_success,

#             "average_accuracy_drop": avg_drop,
#             "average_asr": avg_asr,
#             "average_adv_accuracy": avg_adv_acc,
#             "average_adv_accuracy_correct_only": avg_adv_acc_correct_only,
#             "average_robustness_ratio": avg_robustness_ratio,
#             "average_er_l2_success": avg_l2_success,
#             "average_er_linf_success": avg_linf_success,

#             "n_attacks_executed": len(results),
#             "attacks": results,
#             "failed_attacks": errors,
#         }
    
#     def build_properties(self, raw: dict) -> dict:
#         """Build comprehensive properties for ensemble robustness."""
#         attacks = raw.get("attacks")
        
#         props = {
#             "Metric Description": (
#                 "Ensemble robustness combining multiple adversarial attack methods. "
#                 "Reports worst-case scenario across attacks for comprehensive robustness assessment."
#             ),
#             "Worst-Case Accuracy Drop (%)": float(raw.get("worst_case_accuracy_drop")),
#             "Worst-Case ASR (%)": float(raw.get("worst_case_asr")),
#             "Worst-Case Adv Accuracy (%)": float(raw.get("worst_case_adv_accuracy")),
#             "Worst-Case Adv Accuracy (Correct Only) (%)": float(raw.get("worst_case_adv_accuracy_correct_only")),
#             "Worst-Case Robustness Ratio": float(raw.get("worst_case_robustness_ratio")),
#             "Worst-Case ER L2 (success-only)": float(raw.get("worst_case_er_l2_success")),
#             "Worst-Case ER Linf (success-only)": float(raw.get("worst_case_er_linf_success")),
#             "Average Accuracy Drop (%)": float(raw.get("average_accuracy_drop")),
#             "Average ASR (%)": float(raw.get("average_asr")),
#             "Average Adv Accuracy (%)": float(raw.get("average_adv_accuracy")),
#             "Average Adv Accuracy (Correct Only) (%)": float(raw.get("average_adv_accuracy_correct_only")),
#             "Average Robustness Ratio": float(raw.get("average_robustness_ratio")),
#             "Average ER L2 (success-only)": float(raw.get("average_er_l2_success")),
#             "Average ER Linf (success-only)": float(raw.get("average_er_linf_success")),
#             "Most Effective Attack": raw.get("most_effective_attack"),
#             "Attacks Executed": raw.get("n_attacks_executed"),
#         }
        
#         # Add individual attack results
#         for attack_name, attack_data in attacks.items():
#             props[f"{attack_name} - Drop (%)"] = float(attack_data.get("accuracy_drop_pct (effective_robustness)", 0.0))
#             props[f"{attack_name} - ASR (%)"] = float(attack_data.get("attack_success_rate_pct (correct_only)", 0.0))
#             props[f"{attack_name} - Adv Acc (%)"] = float(attack_data.get("adv_accuracy"))
#             props[f"{attack_name} - Adv Acc (Correct Only) (%)"] = float(attack_data.get("adv_accuracy_correct_only"))
#             props[f"{attack_name} - Robustness Ratio"] = float(attack_data.get("robustness_ratio (adv/clean)", 0.0))
#             props[f"{attack_name} - ER L2 (success-only)"] = float(attack_data.get("er_l2_success"))
#             props[f"{attack_name} - ER Linf (success-only)"] = float(attack_data.get("er_linf_success"))
#             props[f"{attack_name} - Sample Size"] = int(attack_data.get("sample_size (n_eval)", 0))
#             props[f"{attack_name} - N Attacked"] = int(attack_data.get("n_attacked"))
#             props[f"{attack_name} - Note"] = attack_data.get("note")
#         # Add failed attacks information if any
#         failed = raw.get("failed_attacks")
#         if failed:
#             props["Failed Attacks"] = ", ".join(failed.keys())
        
#         return props    

# ============================================================
# ENSEMBLE ROBUSTNESS (Hidden Compute Function)
# ============================================================

def _get_or_compute_ensemble(ctx: EvaluationContext) -> dict:
    """
    Compute ensemble robustness by running available attacks and caching the result.
    
    Computes robustness metrics using available attack methods (HSJ, FGM, CW, DeepFool)
    and aggregates results to provide a comprehensive robustness assessment.
    """
    # Check if already computed
    cached = ctx.extras.get(_ENSEMBLE_KEY)
    if isinstance(cached, dict):
        return cached
    
    # Check if previously failed
    if _ENSEMBLE_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_ENSEMBLE_ERROR_KEY]))

    results = {}
    errors = {}
    
    # Try each attack method in order of preference
    attacks_to_run = [
        ("HSJ", _get_or_compute_hsj, "HopSkipJump (black-box, decision-based)"),
        ("FGM", _get_or_compute_fgm, "Fast Gradient Method (requires gradients)"),
        ("CW", _get_or_compute_cw, "Carlini-Wagner L2 (requires gradients)"),
        ("DeepFool", _get_or_compute_df, "DeepFool (requires gradients)"),
    ]
    
    for attack_name, compute_fn, attack_desc in attacks_to_run:
        try:
            metrics = compute_fn(ctx)
            results[attack_name] = {
                "description": attack_desc,
                "clean_accuracy": float(metrics.get("clean_accuracy")),
                "adv_accuracy": float(metrics.get("adv_accuracy")),
                "accuracy_drop_pct (effective_robustness)": float(
                    metrics.get("accuracy_drop_pct (effective_robustness)")
                ),
                "robustness_ratio (adv/clean)": float(metrics.get("robustness_ratio (adv/clean)")),
                "adv_accuracy_correct_only": float(metrics.get("adv_accuracy_correct_only")),
                "attack_success_rate_pct (correct_only)": float(
                    metrics.get("attack_success_rate_pct (correct_only)")
                ),
                "mean_l2": float(metrics.get("mean_l2")),
                "mean_linf": float(metrics.get("mean_linf")),
                "er_l2_success": float(metrics.get("er_l2_success")),
                "er_linf_success": float(metrics.get("er_linf_success")),
                "sample_size (n_eval)": int(metrics.get("sample_size (n_eval)")),
                "n_attacked": int(metrics.get("n_attacked")),
                "params": metrics.get("params"),
                "note": metrics.get("note"),
            }
        except Exception as e:
            # Silently skip attacks that fail (e.g., non-gradient models with FGM)
            errors[attack_name] = str(e)
    
    if not results:
        error_msg = f"Ensemble robustness requires at least one successful attack. Errors: {errors}"
        ctx.extras[_ENSEMBLE_ERROR_KEY] = error_msg
        raise RuntimeError(error_msg)
    
    # Aggregate metrics
    accuracy_drops = [r["accuracy_drop_pct (effective_robustness)"] for r in results.values()]
    asrs = [r["attack_success_rate_pct (correct_only)"] for r in results.values()]
    adv_accs = [r["adv_accuracy"] for r in results.values()]
    adv_accs_correct_only = [r["adv_accuracy_correct_only"] for r in results.values()]
    robustness_ratios = [r["robustness_ratio (adv/clean)"] for r in results.values()]
    er_l2_successes = [r["er_l2_success"] for r in results.values() if r["er_l2_success"] > 0]
    er_linf_successes = [r["er_linf_success"] for r in results.values() if r["er_linf_success"] > 0]

    worst_drop = float(max(accuracy_drops))
    worst_asr = float(max(asrs))
    worst_adv_acc = float(min(adv_accs))
    worst_adv_acc_correct_only = float(min(adv_accs_correct_only))
    worst_robustness_ratio = float(min(robustness_ratios))
    worst_l2_success = float(min(er_l2_successes)) if er_l2_successes else 0.0
    worst_linf_success = float(min(er_linf_successes)) if er_linf_successes else 0.0
    
    avg_drop = float(sum(accuracy_drops) / len(accuracy_drops))
    avg_asr = float(sum(asrs) / len(asrs))
    avg_adv_acc = float(sum(adv_accs) / len(adv_accs))
    avg_adv_acc_correct_only = float(sum(adv_accs_correct_only) / len(adv_accs_correct_only))
    avg_robustness_ratio = float(sum(robustness_ratios) / len(robustness_ratios))
    avg_l2_success = float(sum(er_l2_successes) / len(er_l2_successes)) if er_l2_successes else 0.0
    avg_linf_success = float(sum(er_linf_successes) / len(er_linf_successes)) if er_linf_successes else 0.0

    # Find most effective attack (highest accuracy drop)
    most_effective = max(results.items(), key=lambda x: x[1]["accuracy_drop_pct (effective_robustness)"])[0]
    
    final_metrics = {
        "most_effective_attack": most_effective,
        "worst_case_accuracy_drop": worst_drop,
        "worst_case_asr": worst_asr,
        "worst_case_adv_accuracy": worst_adv_acc,
        "worst_case_adv_accuracy_correct_only": worst_adv_acc_correct_only,
        "worst_case_robustness_ratio": worst_robustness_ratio,
        "worst_case_er_l2_success": worst_l2_success,
        "worst_case_er_linf_success": worst_linf_success,

        "average_accuracy_drop": avg_drop,
        "average_asr": avg_asr,
        "average_adv_accuracy": avg_adv_acc,
        "average_adv_accuracy_correct_only": avg_adv_acc_correct_only,
        "average_robustness_ratio": avg_robustness_ratio,
        "average_er_l2_success": avg_l2_success,
        "average_er_linf_success": avg_linf_success,

        "n_attacks_executed": len(results),
        "attacks": results,
        "failed_attacks": errors,
    }
    
    # Save to context cache
    ctx.extras[_ENSEMBLE_KEY] = final_metrics
    return final_metrics

# ============================================================
# ENSEMBLE SUB-METRICS (Extracted from EnsembleRobustnessMetric)
# ============================================================

class AdversarialAccuracyMetric(BaseMetric):
    """Worst-case adversarial accuracy across the ensemble."""
    
    def __init__(self):
        super().__init__("adversarial_accuracy", "score_adversarial_accuracy")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_adv_accuracy")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case adversarial accuracy across the available attack suite.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "Adversarial Accuracy = Accuracy(model(x_adv), y_true)",
            "Average Adv Accuracy (%)": float(raw.get("average_adv_accuracy")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Adversarial Accuracy (%)": float(raw.get("value")),
        }
    
class AccuracyDropMetric(BaseMetric):
    """Worst-case accuracy drop across the ensemble of attacks."""
    
    def __init__(self):
        super().__init__("accuracy_drop", "score_accuracy_drop")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_accuracy_drop")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case accuracy drop across the available attack suite.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "Accuracy Drop = Clean Accuracy - Adversarial Accuracy",
            "Average Drop (%)": float(raw.get("average_accuracy_drop")),
            "Results from Individual Attacks": raw.get("attacks"),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Accuracy Drop (%)": float(raw.get("value")),
        }

class RobustnessRatioMetric(BaseMetric):
    """Worst-case robustness ratio (adv/clean) across the ensemble."""
    
    def __init__(self):
        super().__init__("robustness_ratio", "score_robustness_ratio")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_robustness_ratio")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case robustness ratio (Adversarial Accuracy / Clean Accuracy) across the available attack suite.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "Robustness Ratio = Adversarial Accuracy / Clean Accuracy",
            "Average Robustness Ratio": float(raw.get("average_robustness_ratio")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Robustness Ratio": float(raw.get("value")),
        }


class AdversarialAccuracyCorrectOnlyMetric(BaseMetric):
    """Worst-case adversarial accuracy (correct-only) across the ensemble."""
    
    def __init__(self):
        super().__init__("adversarial_accuracy_correct_only", "score_adversarial_accuracy_correct_only")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_adv_accuracy_correct_only")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case adversarial accuracy restricted to originally correctly classified samples.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "Adversarial Accuracy (Correct-Only) = Accuracy on adversarial examples generated from originally correct samples",
            "Average Adv Accuracy (Correct Only) (%)": float(raw.get("average_adv_accuracy_correct_only")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Adversarial Accuracy (Correct Only) (%)": float(raw.get("value")),
        }
    

class ASRMetric(BaseMetric):
    """Worst-case Attack Success Rate (ASR) across the ensemble."""
    
    def __init__(self):
        super().__init__("asr", "score_asr")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_asr")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case attack success rate across the available attack suite.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "ASR = Successful Attacks / Attacks Attempted",
            "Average ASR (%)": float(raw.get("average_asr")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Attack Success Rate (%)": float(raw.get("value")),
        }


class EmpiricalRobustnessL2Metric(BaseMetric):
    """Worst-case empirical robustness (L2 norm) across the ensemble."""
    
    def __init__(self):
        super().__init__("empirical_robustness_l2", "score_empirical_robustness_l2")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_er_l2_success")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case empirical robustness measured with L2 distance for successful attacks.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "ER-L2 = mean(||x_adv - x||_2) over successful attacks",
            "Average ER L2": float(raw.get("average_er_l2_success")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Empirical Robustness L2": float(raw.get("value")),
        }


class EmpiricalRobustnessLinfMetric(BaseMetric):
    """Worst-case empirical robustness (Linf norm) across the ensemble."""
    
    def __init__(self):
        super().__init__("empirical_robustness_linf", "score_empirical_robustness_linf")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ensemble(ctx)
        return {"value": float(m.get("worst_case_er_linf_success")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Worst-case empirical robustness measured with Linf distance for successful attacks.",
            "Depends on": "Model, Test Data, and Attack Results",
            "Formula": "ER-Linf = mean(||x_adv - x||_inf) over successful attacks",
            "Average ER Linf": float(raw.get("average_er_linf_success")),
            "Most Effective Attack": raw.get("most_effective_attack"),
            "Empirical Robustness Linf": float(raw.get("value")),
        }
    
# ============================================================
# Clique Method
# ============================================================

class CliqueMethodMetric(BaseMetric):
    """Clique Method verification error (lower is better)."""

    def __init__(self):
        super().__init__("clique_method", "score_clique_method")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_clique(ctx)
        return {"value": float(m.get("verification_error")), **m}

    def build_properties(self, raw: dict) -> dict:
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        return {
            "Metric Description": "Clique Method verification error for tree-based models. Lower indicates stronger verified robustness.",
            "Depends on": "Tree-Based Model and Test Data",
            "Formula": "Clique Method Score = verification error from ART tree robustness verification",
            "Robustness Bound": float(raw.get("robustness_bound")),
            "Verification Error": float(raw.get("verification_error")),
            "Sample Size": int(raw.get("sample_size")),
            "Params": params,
            "Metric": raw.get("metric"),
            "Clique Method Score": float(raw.get("value")),
        }


# ============================================================
# CLEVER
# ============================================================

class CleverScoreMetric(BaseMetric):
    """CLEVER score mean (higher is better). Requires gradient-capable ART classifier."""

    def __init__(self):
        super().__init__("clever_score", "score_clever_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_clever(ctx)
        return {"value": float(m.get("clever_score_mean")), **m}

    def build_properties(self, raw: dict) -> dict:
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        return {
            "Metric Description": "Mean CLEVER score for a gradient-capable classifier. Higher values indicate greater robustness.",
            "Depends on": "Gradient-Capable Model and Test Data",
            "Formula": "CLEVER = estimate of minimum adversarial perturbation norm via local Lipschitz constants",
            "CLEVER Std": float(raw.get("clever_score_std")),
            "N Eval": int(raw.get("n_eval")),
            "Params": params,
            "Metric": raw.get("metric"),
            "CLEVER Mean": float(raw.get("value")),
        }


class ConfidenceScoreMetric(BaseMetric):
    """Model confidence metric based on confusion matrix diagonal."""
    
    def __init__(self):
        super().__init__("confidence_score", "score_confidence_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_confidence(ctx)
        return {"value": float(m["confidence_score"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Confidence score computed from thresholded prediction consistency. Average Jaccard index (TP / (TP + FP + FN)) over multiple probability thresholds, where predictions are binarized at each threshold. Higher indicates more confident predictions.",
            "Depends on": "Model and Test Data",
            "Formula": "Confidence Score = mean_t Jaccard(y_true_bin(t), y_pred_bin(t))",
            "Metric": raw.get("metric"),
            "Thresholds": raw.get("thresholds"),
            "Confidence Score (%)": float(raw.get("value")),
        }

    
class LossSensitivityMetric(BaseMetric):
    """Loss sensitivity metric via gradient-based estimations."""
    
    def __init__(self):
        super().__init__("loss_sensitivity", "score_loss_sensitivity")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_loss(ctx)
        return {"value": float(m["loss_sensitivity"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Local loss sensitivity estimated from gradients. Measures how sensitive model loss is to input perturbations.",
            "Depends on": "Gradient-Capable Model and Test Data",
            "Formula": "Loss Sensitivity = expected norm of gradient of loss with respect to input",
            "Metric": raw.get("metric"),
            "Loss Sensitivity": float(raw.get("value")),
        }


class ExpectedCalibrationErrorMetric(BaseMetric):
    """
    Expected Calibration Error (ECE).
    Lower is better (0 = perfectly calibrated).
    """

    def __init__(self):
        super().__init__("expected_calibration_error", "score_expected_calibration_error")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_ece(ctx)
        return {"value": float(m.get("ece")), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Expected Calibration Error computed as the weighted gap between confidence and accuracy across bins.",
            "Depends on": "Model and Test Data",
            "Formula": "ECE = sum_b (n_b / n) * |acc(b) - conf(b)|",
            "Number of Bins": raw.get("n_bins"),
            "Bins stats": raw.get("bins"),
            "Metric": raw.get("metric"),
            "Expected Calibration Error": float(raw.get("value")),
        }