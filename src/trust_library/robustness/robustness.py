from __future__ import annotations

from typing import Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext

from . import robustness_metrics_core as core
from .metrics import (
    # HopSkipJumpAttackMetric,
    # FastGradientAttackMetric,
    # CarliniWagnerAttackMetric,
    # DeepFoolAttackMetric,
    # EnsembleRobustnessMetric,

    IndividualAttackResultsMetric,
    AccuracyDropMetric,
    ASRMetric,
    AdversarialAccuracyMetric,
    AdversarialAccuracyCorrectOnlyMetric,
    RobustnessRatioMetric,
    EmpiricalRobustnessL2Metric,
    EmpiricalRobustnessLinfMetric,

    CliqueMethodMetric,
    CleverScoreMetric,

    LossSensitivityMetric,
    ConfidenceScoreMetric,
    # PopulationStabilityIndexMetric,
    ExpectedCalibrationErrorMetric,
)


class RobustnessPillar(Pillar):
    """Robustness pillar based on adversarial robustness metrics (ART + derived)."""

    @property
    def pillar_key(self) -> str:
        return "robustness"
    
    def get_metrics(self) -> List[Any]:
        return [
            IndividualAttackResultsMetric(),         # individual_attack_results
            # HSJ grouped metric
            # HopSkipJumpAttackMetric(),
            # FastGradientAttackMetric(),                # fgsm_success
            # CarliniWagnerAttackMetric(),               # cw_success
            # DeepFoolAttackMetric(),                    # deepfool_success
            # EnsembleRobustnessMetric(),               # ensemble_robustness

            AccuracyDropMetric(),                     # accuracy_drop
            ASRMetric(),                              # asr
            AdversarialAccuracyMetric(),              # adv_accuracy, adv_accuracy_correct_only
            AdversarialAccuracyCorrectOnlyMetric(),   # adv_accuracy_correct_only
            RobustnessRatioMetric(),                  # robustness_ratio
            EmpiricalRobustnessL2Metric(),            # er_l2_success (success-only)
            EmpiricalRobustnessLinfMetric(),          # er_linf_success (success-only)

            # ART metrics
            CliqueMethodMetric(),                      # verification_error (tree-only)
            CleverScoreMetric(),                       # clever_score_mean (requires gradients)
            # Other robustness metrics
            LossSensitivityMetric(),                   # loss_sensitivity
            ConfidenceScoreMetric(),                  # confidence_score
            # PopulationStabilityIndexMetric(),          # population_stability_index
            ExpectedCalibrationErrorMetric(),         # expected_calibration_error
        ]

    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        # Optional parameters under mappings.robustness.params
        params = (config or {}).get("params")
        if not isinstance(params, dict):
            params = {}

        # -------------------------------
        # HSJ (black-box) params
        # -------------------------------
        n_samples = int(params.get("n_samples"))
        seed = int(params.get("seed"))

        max_iter = int(params.get("max_iter"))
        max_eval = int(params.get("max_eval"))
        init_eval = int(params.get("init_eval"))
        init_size = int(params.get("init_size"))
        norm = params.get("norm")
        hsj_beta = float(params.get("hsj_beta"))
        fgm_epsilon = float(params.get("fgm_epsilon"))
        ece_n_bins = int(params.get("ece_n_bins"))

        context.extras["robustness_params"] = {
            "n_samples": n_samples,
            "seed": seed,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "init_eval": init_eval,
            "init_size": init_size,
            "norm": norm,
            "beta": hsj_beta,
            "eps": fgm_epsilon,
            "n_bins": ece_n_bins,
        }

        # -------------------------------
        # Clique Method params (tree-only)
        # (lazy compute in metric)
        # -------------------------------
        clique_params = params.get("clique") # NOT in "params" by default
        if not isinstance(clique_params, dict):
            clique_params = {}

        context.extras["robustness_clique_params"] = {
            "n_samples": int(clique_params.get("n_samples")),
            "seed": int(clique_params.get("seed")),
            "eps_init": float(clique_params.get("eps_init")),
            "norm": float(clique_params.get("norm")),
            "nb_search_steps": int(clique_params.get("nb_search_steps")),
            "max_clique": int(clique_params.get("max_clique")),
            "max_level": int(clique_params.get("max_level")),
        }

        # -------------------------------
        # CLEVER params (requires gradients)
        # (lazy compute in metric)
        # -------------------------------
        clever_params = params.get("clever") # NOT in "params" by default
        if not isinstance(clever_params, dict):
            clever_params = {}

        context.extras["robustness_clever_params"] = {
            "n_samples": int(clever_params.get("n_samples")),
            "seed": int(clever_params.get("seed")),
            "nb_batches": int(clever_params.get("nb_batches")),
            "batch_size": int(clever_params.get("batch_size")),
            "radius": float(clever_params.get("radius")),
            "norm": int(clever_params.get("norm")),
        }

        # --------------------------------------------
        # Best-effort eager compute (HSJ only)
        # --------------------------------------------
        # try:
        #     metrics = core.hopskipjump_metrics(
        #         model=context.model,
        #         X_test=context.X_test,
        #         y_test=context.y_test,
        #         X_train=getattr(context, "X_train", None),
        #         n_samples=n_samples,
        #         seed=seed,
        #         max_iter=max_iter,
        #         max_eval=max_eval,
        #         init_eval=init_eval,
        #         init_size=init_size,
        #         norm=norm,
        #     )
        #     context.extras["robustness_art_metrics"] = metrics
        # except Exception as exc:
        #     context.extras["robustness_error"] = str(exc)