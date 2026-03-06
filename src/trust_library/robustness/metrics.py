from __future__ import annotations

from trust_library.base_metric import BaseMetric
from trust_library.utils import EvaluationContext

from . import robustness_metrics_core as core

# -----------------------
# Cache keys (aligned with RobustnessPillar.prepare)
# -----------------------
_HSJ_KEY = "robustness_art_metrics"
_HSJ_PARAMS_KEY = "robustness_params"
_HSJ_ERROR_KEY = "robustness_error"

_CLIQUE_KEY = "robustness_clique_metrics"
_CLIQUE_PARAMS_KEY = "robustness_clique_params"
_CLIQUE_ERROR_KEY = "robustness_clique_error"

_CLEVER_KEY = "robustness_clever_metrics"
_CLEVER_PARAMS_KEY = "robustness_clever_params"
_CLEVER_ERROR_KEY = "robustness_clever_error"

# ============================================================
# EXTRA CACHE KEYS
# ============================================================

_FGM_KEY = "robustness_fgm_metrics"
_FGM_PARAMS_KEY = "robustness_fgm_params"
_FGM_ERROR_KEY = "robustness_fgm_error"

_CW_KEY = "robustness_cw_metrics"
_CW_PARAMS_KEY = "robustness_cw_params"
_CW_ERROR_KEY = "robustness_cw_error"

_DF_KEY = "robustness_df_metrics"
_DF_PARAMS_KEY = "robustness_df_params"
_DF_ERROR_KEY = "robustness_df_error"

_LOSS_KEY = "robustness_loss_sensitivity_metrics"
_LOSS_PARAMS_KEY = "robustness_loss_sensitivity_params"
_LOSS_ERROR_KEY = "robustness_loss_sensitivity_error"

_CONF_KEY = "robustness_confidence_metrics"
_CONF_ERROR_KEY = "robustness_confidence_error"


# ============================================================
# CACHED COMPUTES
# ============================================================

def _get_or_compute_hsj(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_HSJ_KEY)
    if isinstance(cached, dict):
        return cached

    if _HSJ_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_HSJ_ERROR_KEY]))

    params = ctx.extras.get(_HSJ_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}

    if not hasattr(ctx.model, "predict"):
        ctx.extras[_HSJ_ERROR_KEY] = "Model must implement predict() to run HopSkipJump."
        raise RuntimeError(ctx.extras[_HSJ_ERROR_KEY])

    try:
        metrics = core.hopskipjump_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            X_train=getattr(ctx, "X_train", None),
            n_samples=int(params.get("n_samples", 30)),
            seed=int(params.get("seed", 42)),
            max_iter=int(params.get("max_iter", 10)),
            max_eval=int(params.get("max_eval", 1000)),
            init_eval=int(params.get("init_eval", 10)),
            init_size=int(params.get("init_size", 10)),
            norm=params.get("norm", 2),
        )
    except Exception as exc:
        ctx.extras[_HSJ_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_HSJ_KEY] = metrics
    return metrics


def _get_or_compute_clique(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_CLIQUE_KEY)
    if isinstance(cached, dict):
        return cached

    if _CLIQUE_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CLIQUE_ERROR_KEY]))

    params = ctx.extras.get(_CLIQUE_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}

    try:
        metrics = core.clique_method_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            X_train=getattr(ctx, "X_train", None),
            n_samples=int(params.get("n_samples", 200)),
            seed=int(params.get("seed", 42)),
            eps_init=float(params.get("eps_init", 0.1)),
            norm=float(params.get("norm", float("inf"))),
            nb_search_steps=int(params.get("nb_search_steps", 10)),
            max_clique=int(params.get("max_clique", 2)),
            max_level=int(params.get("max_level", 2)),
        )
    except Exception as exc:
        ctx.extras[_CLIQUE_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_CLIQUE_KEY] = metrics
    return metrics


def _get_or_compute_clever(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_CLEVER_KEY)
    if isinstance(cached, dict):
        return cached

    if _CLEVER_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_CLEVER_ERROR_KEY]))

    params = ctx.extras.get(_CLEVER_PARAMS_KEY, {})
    if not isinstance(params, dict):
        params = {}

    # Requires a gradient-capable ART classifier.
    classifier = ctx.extras.get("art_classifier", None)
    if classifier is None:
        classifier = ctx.model  # will raise a clear error in core if not gradient-capable

    try:
        metrics = core.clever_score_metrics(
            classifier=classifier,
            x=ctx.X_test,
            n_samples=int(params.get("n_samples", 5)),
            seed=int(params.get("seed", 42)),
            nb_batches=int(params.get("nb_batches", 10)),
            batch_size=int(params.get("batch_size", 32)),
            radius=float(params.get("radius", 0.5)),
            norm=int(params.get("norm", 2)),
        )
    except Exception as exc:
        ctx.extras[_CLEVER_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_CLEVER_KEY] = metrics
    return metrics


# ============================================================
# CACHED COMPUTES
# ============================================================

def _get_or_compute_fgm(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_FGM_KEY)
    if isinstance(cached, dict):
        return cached

    params = ctx.extras.get(_FGM_PARAMS_KEY, {}) or {}

    try:
        metrics = core.fgm_attack_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            eps=float(params.get("eps", 0.2)),
            n_samples=int(params.get("n_samples", 50)),
            seed=int(params.get("seed", 42)),
        )
    except Exception as exc:
        ctx.extras[_FGM_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_FGM_KEY] = metrics
    return metrics


def _get_or_compute_cw(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_CW_KEY)
    if isinstance(cached, dict):
        return cached

    params = ctx.extras.get(_CW_PARAMS_KEY, {}) or {}

    try:
        metrics = core.carlini_wagner_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            n_samples=int(params.get("n_samples", 10)),
            seed=int(params.get("seed", 42)),
        )
    except Exception as exc:
        ctx.extras[_CW_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_CW_KEY] = metrics
    return metrics


def _get_or_compute_df(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_DF_KEY)
    if isinstance(cached, dict):
        return cached

    params = ctx.extras.get(_DF_PARAMS_KEY, {}) or {}

    try:
        metrics = core.deepfool_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            n_samples=int(params.get("n_samples", 10)),
            seed=int(params.get("seed", 42)),
        )
    except Exception as exc:
        ctx.extras[_DF_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_DF_KEY] = metrics
    return metrics


def _get_or_compute_loss(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_LOSS_KEY)
    if isinstance(cached, dict):
        return cached

    params = ctx.extras.get(_LOSS_PARAMS_KEY, {}) or {}

    classifier = ctx.extras.get("art_classifier", ctx.model)

    try:
        metrics = core.loss_sensitivity_metrics(
            classifier=classifier,
            X_test=ctx.X_test,
        )
    except Exception as exc:
        ctx.extras[_LOSS_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_LOSS_KEY] = metrics
    return metrics


def _get_or_compute_confidence(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_CONF_KEY)
    if isinstance(cached, dict):
        return cached

    try:
        metrics = core.confidence_score_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
        )
    except Exception as exc:
        ctx.extras[_CONF_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_CONF_KEY] = metrics
    return metrics

# ============================================================
# PROPERTIES HELPERS
# ============================================================

def _hsj_common_props(raw: dict) -> dict:
    params = raw.get("params", {}) if isinstance(raw.get("params"), dict) else {}
    return {
        "Attack": raw.get("attack"),
        "Clean Accuracy": float(raw.get("clean_accuracy", 0.0)),
        "Adversarial Accuracy": float(raw.get("adv_accuracy", 0.0)),
        "Accuracy Drop (%)": float(raw.get("accuracy_drop_pct", 0.0)),
        "ASR (%)": float(raw.get("asr_pct", raw.get("attack_success_rate_pct", 0.0))),
        "Mean L2 Perturbation": float(raw.get("mean_l2", 0.0)),
        "Mean Linf Perturbation": float(raw.get("mean_linf", 0.0)),
        "ER L2 (success-only)": float(raw.get("er_l2_success", 0.0)),
        "ER Linf (success-only)": float(raw.get("er_linf_success", 0.0)),
        "N Eval (subset)": int(raw.get("n_eval", 0.0)),
        "N Attacked (correct-only)": int(raw.get("n_attacked", 0.0)),
        "Sample Size": int(raw.get("sample_size", 0.0)),
        "Params": params,
        "Note": raw.get("note"),
    }


# ============================================================
# HSJ-derived metrics
# ============================================================

class HopSkipJumpAccuracyDropMetric(BaseMetric):
    """Accuracy drop (%) on evaluation subset after HSJ (lower is better)."""

    def __init__(self):
        super().__init__("er_hopskipjump_attack", "score_hopskipjump_attack")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)
        return {"value": float(m.get("accuracy_drop_pct", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Accuracy drop (percentage points) after ART HopSkipJump attack "
                "(black-box, decision-based). Lower is better."
            ),
            "Value": float(raw.get("value", 0.0)),
            **_hsj_common_props(raw),
        }


# Backwards-compatible alias if older code imports HopSkipJumpAttackMetric
HopSkipJumpAttackMetric = HopSkipJumpAccuracyDropMetric


class HopSkipJumpASRMetric(BaseMetric):
    """Attack Success Rate (%) after HSJ (higher is worse)."""

    def __init__(self):
        super().__init__("attack_success_rate", "score_attack_success_rate")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)
        val = float(m.get("asr_pct", m.get("attack_success_rate_pct", 0.0)))
        return {"value": val, **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Attack Success Rate: percentage of attacked samples whose prediction changes "
                "compared to the clean prediction. Higher indicates lower robustness."
            ),
            "Value": float(raw.get("value", 0.0)),
            **_hsj_common_props(raw),
        }


class HopSkipJumpAdversarialAccuracyMetric(BaseMetric):
    """Adversarial accuracy (%) on evaluation subset (higher is better)."""

    def __init__(self):
        super().__init__("adversarial_accuracy", "score_adversarial_accuracy")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)
        adv_acc = float(m.get("adv_accuracy", 0.0))
        # core returns adv_accuracy as [0,1]; expose as %
        return {"value": adv_acc * 100.0, **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": "Accuracy (%) on adversarial examples. Higher is more robust.",
            "Value": float(raw.get("value", 0.0)),
            **_hsj_common_props(raw),
        }


class HopSkipJumpEmpiricalRobustnessL2Metric(BaseMetric):
    """Empirical robustness proxy (L2): mean L2 on successful attacks (higher is better)."""

    def __init__(self):
        super().__init__("empirical_robustness_l2", "score_empirical_robustness_l2")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)
        return {"value": float(m.get("er_l2_success", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Empirical robustness proxy (L2): mean L2 perturbation among successful adversarial examples. "
                "Higher means larger changes are required to fool the model."
            ),
            "Value": float(raw.get("value", 0.0)),
            **_hsj_common_props(raw),
        }


class HopSkipJumpEmpiricalRobustnessLinfMetric(BaseMetric):
    """Empirical robustness proxy (Linf): mean Linf on successful attacks (higher is better)."""

    def __init__(self):
        super().__init__("empirical_robustness_linf", "score_empirical_robustness_linf")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)
        return {"value": float(m.get("er_linf_success", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Empirical robustness proxy (Linf): mean Linf perturbation among successful adversarial examples. "
                "Higher indicates stronger robustness."
            ),
            "Value": float(raw.get("value", 0.0)),
            **_hsj_common_props(raw),
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
        return {"value": float(m.get("verification_error", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        params = raw.get("params", {}) if isinstance(raw.get("params"), dict) else {}
        return {
            "Metric Description": (
                "Clique Method robustness verification error for tree-based models. "
                "Lower indicates stronger verified robustness."
            ),
            "Value": float(raw.get("value", 0.0)),
            "Robustness Bound": float(raw.get("robustness_bound", 0.0)),
            "Verification Error": float(raw.get("verification_error", 0.0)),
            "Sample Size": int(raw.get("sample_size", 0.0)),
            "Params": params,
            "Metric": raw.get("metric"),
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
        return {"value": float(m.get("clever_score_mean", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        params = raw.get("params", {}) if isinstance(raw.get("params"), dict) else {}
        return {
            "Metric Description": (
                "CLEVER score (mean). Higher values indicate greater robustness. "
                "Requires an ART classifier with gradients (e.g., PyTorchClassifier)."
            ),
            "Value": float(raw.get("value", 0.0)),
            "CLEVER Mean": float(raw.get("clever_score_mean", 0.0)),
            "CLEVER Std": float(raw.get("clever_score_std", 0.0)),
            "N Eval": int(raw.get("n_eval", 0.0)),
            "Params": params,
            "Metric": raw.get("metric"),
        }
    


# ============================================================
# METRIC CLASSES
# ============================================================

class FastGradientAttackMetric(BaseMetric):
    def __init__(self):
        super().__init__("er_fast_gradient_attack", "score_fast_gradient_attack")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_fgm(ctx)
        return {"value": float(m["accuracy_drop_pct"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return raw


class CarliniWagnerAttackMetric(BaseMetric):
    def __init__(self):
        super().__init__("er_carlini_wagner_attack", "score_carlini_wagner_attack")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_cw(ctx)
        return {"value": float(m["accuracy_drop_pct"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return raw


class DeepFoolAttackMetric(BaseMetric):
    def __init__(self):
        super().__init__("er_deepfool_attack", "score_deepfool_attack")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_df(ctx)
        return {"value": float(m["accuracy_drop_pct"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return raw


class LossSensitivityMetric(BaseMetric):
    def __init__(self):
        super().__init__("loss_sensitivity", "score_loss_sensitivity")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_loss(ctx)
        return {"value": float(m["loss_sensitivity"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return raw


class ConfidenceScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__("confidence_score", "score_confidence_score")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_confidence(ctx)
        return {"value": float(m["confidence_score"]), **m}

    def build_properties(self, raw: dict) -> dict:
        return raw
    


class PopulationStabilityIndexMetric(BaseMetric):
    def __init__(self):
        super().__init__("psi", "score_psi")

    def compute(self, ctx: EvaluationContext) -> dict:
        train_values = ctx.extras.get("psi_train_distribution")
        current_values = ctx.extras.get("psi_current_distribution")

        if train_values is None or current_values is None:
            raise RuntimeError("PSI requires train and current distributions in ctx.extras.")

        m = core.psi_metrics(
            train_values=train_values,
            current_values=current_values,
            n_bins=ctx.extras.get("psi_bins", 10),
        )

        return {"value": m["psi"], **m}
    
class RobustnessRatioHSJMetric(BaseMetric):
    def __init__(self):
        super().__init__("robustness_ratio_hsj", "score_robustness_ratio_hsj")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_hsj(ctx)

        clean = float(m["clean_accuracy"])
        adv = float(m["adv_accuracy"])

        ratio = adv / clean if clean > 0 else 0.0

        return {
            "value": ratio,
            "clean_accuracy": clean,
            "adv_accuracy": adv,
        }
    

class EffectiveRobustnessMetric(BaseMetric):
    def __init__(self):
        super().__init__("effective_robustness", "score_effective_robustness")

    def compute(self, ctx: EvaluationContext) -> dict:
        acc1 = ctx.extras.get("acc_scenario_1")
        acc2 = ctx.extras.get("acc_scenario_2")
        beta = ctx.extras.get("beta", 1.0)

        if acc1 is None or acc2 is None:
            raise RuntimeError("EffectiveRobustness requires acc_scenario_1 and acc_scenario_2.")

        m = core.effective_robustness(
            acc_scenario_1=acc1,
            acc_scenario_2=acc2,
            beta=beta,
        )

        return {"value": m["effective_robustness"], **m}


_RGR_KEY = "robustness_rgr_metrics"
_RGR_PARAMS_KEY = "robustness_rgr_params"
_RGR_ERROR_KEY = "robustness_rgr_error"

_ECE_KEY = "calibration_ece_metrics"
_ECE_PARAMS_KEY = "calibration_ece_params"
_ECE_ERROR_KEY = "calibration_ece_error"

def _get_or_compute_rgr(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_RGR_KEY)
    if isinstance(cached, dict):
        return cached

    if _RGR_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_RGR_ERROR_KEY]))

    params = ctx.extras.get(_RGR_PARAMS_KEY, {}) or {}

    try:
        metrics = core.rgr_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            feature_index=int(params.get("feature_index", 0)),
            perturbation=float(params.get("perturbation", 0.01)),
            method=params.get("method", "additive"),
            ranking_method=params.get("ranking_method", "kendall"),
        )
    except Exception as exc:
        ctx.extras[_RGR_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_RGR_KEY] = metrics
    return metrics

def _get_or_compute_ece(ctx: EvaluationContext) -> dict:
    cached = ctx.extras.get(_ECE_KEY)
    if isinstance(cached, dict):
        return cached

    if _ECE_ERROR_KEY in ctx.extras:
        raise RuntimeError(str(ctx.extras[_ECE_ERROR_KEY]))

    params = ctx.extras.get(_ECE_PARAMS_KEY, {}) or {}

    try:
        metrics = core.ece_metrics(
            model=ctx.model,
            X_test=ctx.X_test,
            y_test=ctx.y_test,
            n_bins=int(params.get("n_bins", 10)),
        )
    except Exception as exc:
        ctx.extras[_ECE_ERROR_KEY] = str(exc)
        raise

    ctx.extras[_ECE_KEY] = metrics
    return metrics


class RankGlobalRobustnessMetric(BaseMetric):
    """
    Rank-based Global Robustness (RGR).
    Measures ranking stability under single-feature perturbation.
    Higher is better (1 = identical ranking).
    """

    def __init__(self):
        super().__init__("rgr", "score_rgr")

    def compute(self, ctx: EvaluationContext) -> dict:
        m = _get_or_compute_rgr(ctx)
        return {"value": float(m.get("rgr_score", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Rank Global Robustness (RGR): stability of prediction ranking "
                "after perturbing one feature. 1 indicates identical ranking."
            ),
            "Value": float(raw.get("value", 0.0)),
            "Feature Index": raw.get("feature_index"),
            "Perturbation": raw.get("perturbation"),
            "Ranking Method": raw.get("ranking_method"),
            "Metric": raw.get("metric"),
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
        return {"value": float(m.get("ece", 0.0)), **m}

    def build_properties(self, raw: dict) -> dict:
        return {
            "Metric Description": (
                "Expected Calibration Error (ECE): weighted average difference "
                "between predicted confidence and empirical accuracy across bins."
            ),
            "Value": float(raw.get("value", 0.0)),
            "Number of Bins": raw.get("n_bins"),
            "Metric": raw.get("metric"),
        }