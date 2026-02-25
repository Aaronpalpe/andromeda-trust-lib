from __future__ import annotations

from typing import Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext

from . import robustness_metrics_core as core
from .metrics import (
    HopSkipJumpAccuracyDropMetric,
    HopSkipJumpASRMetric,
    HopSkipJumpAdversarialAccuracyMetric,
    HopSkipJumpEmpiricalRobustnessL2Metric,
    HopSkipJumpEmpiricalRobustnessLinfMetric,
    CliqueMethodMetric,
    CleverScoreMetric,
)


class RobustnessPillar(Pillar):
    """Robustness pillar based on adversarial robustness metrics (ART + derived)."""

    @property
    def pillar_key(self) -> str:
        return "robustness"
    
    def get_metrics(self) -> List[Any]:
        return [
            # HSJ-derived
            HopSkipJumpAccuracyDropMetric(),           # accuracy_drop_pct
            HopSkipJumpASRMetric(),                    # asr_pct
            HopSkipJumpAdversarialAccuracyMetric(),    # adv_accuracy
            HopSkipJumpEmpiricalRobustnessL2Metric(),  # er_l2_success
            HopSkipJumpEmpiricalRobustnessLinfMetric(),# er_linf_success

            # ART metrics
            CliqueMethodMetric(),                      # verification_error (tree-only)
            CleverScoreMetric(),                       # clever_score_mean (requires gradients)
        ]

    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        # Optional parameters under mappings.robustness.params
        params = (config or {}).get("params", {})
        if not isinstance(params, dict):
            params = {}

        # -------------------------------
        # HSJ (black-box) params
        # -------------------------------
        n_samples = int(params.get("n_samples", 30))
        seed = int(params.get("seed", 42))

        max_iter = int(params.get("max_iter", 10))
        max_eval = int(params.get("max_eval", 1000))
        init_eval = int(params.get("init_eval", 10))
        init_size = int(params.get("init_size", 10))
        norm = params.get("norm", 2)

        context.extras["robustness_params"] = {
            "n_samples": n_samples,
            "seed": seed,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "init_eval": init_eval,
            "init_size": init_size,
            "norm": norm,
        }

        # -------------------------------
        # Clique Method params (tree-only)
        # (lazy compute in metric)
        # -------------------------------
        clique_params = params.get("clique", {})
        if not isinstance(clique_params, dict):
            clique_params = {}

        context.extras["robustness_clique_params"] = {
            "n_samples": int(clique_params.get("n_samples", 200)),
            "seed": int(clique_params.get("seed", seed)),
            "eps_init": float(clique_params.get("eps_init", 0.1)),
            "norm": float(clique_params.get("norm", float("inf"))),
            "nb_search_steps": int(clique_params.get("nb_search_steps", 10)),
            "max_clique": int(clique_params.get("max_clique", 2)),
            "max_level": int(clique_params.get("max_level", 2)),
        }

        # -------------------------------
        # CLEVER params (requires gradients)
        # (lazy compute in metric)
        # -------------------------------
        clever_params = params.get("clever", {})
        if not isinstance(clever_params, dict):
            clever_params = {}

        context.extras["robustness_clever_params"] = {
            "n_samples": int(clever_params.get("n_samples", 5)),
            "seed": int(clever_params.get("seed", seed)),
            "nb_batches": int(clever_params.get("nb_batches", 10)),
            "batch_size": int(clever_params.get("batch_size", 32)),
            "radius": float(clever_params.get("radius", 0.5)),
            "norm": int(clever_params.get("norm", 2)),
        }

        # --------------------------------------------
        # Best-effort eager compute (HSJ only)
        # --------------------------------------------
        try:
            metrics = core.compute_hopskipjump_metrics(
                model=context.model,
                X_test=context.X_test,
                y_test=context.y_test,
                X_train=getattr(context, "X_train", None),
                n_samples=n_samples,
                seed=seed,
                max_iter=max_iter,
                max_eval=max_eval,
                init_eval=init_eval,
                init_size=init_size,
                norm=norm,
            )
            context.extras["robustness_art_metrics"] = metrics
        except Exception as exc:
            context.extras["robustness_error"] = str(exc)