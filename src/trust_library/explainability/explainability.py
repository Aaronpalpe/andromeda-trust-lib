from __future__ import annotations

from multiprocessing import context
from typing import Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext

import numpy as np
from sklearn.inspection import partial_dependence
import pandas as pd

from .metrics import (
    SparsityMetric,
    FeatureEntropyMetric,
    TopKConcentrationMetric,
    AlgorithmClassMetric,
    CorrelatedFeaturesMetric,
    ModelSizeMetric,
    FeatureRelevanceMetric,
    # PerformanceDifferenceMetric,
    NumberOfRulesMetric,
    AverageRuleLengthMetric,
    RuleStatsMetric,
    TreeDepthMetric,
    InteractionStrengthMetric,
    FaithfulnessMetric,
    MonotonicityMetric,
    # ShapleyCorrMetric,
    # ROARMetric,
    InfidelityMetric,
    AlphaImportanceScoreMetric,
    XAIEaseScoreMetric,
    PositionParityMetric,
    RankAlignmentMetric,
    SpreadRatioMetric,
    SpreadDivergenceMetric,
    # FluctuationRatioMetric,
    # RankConsistencyMetric,
    # ImportanceStabilityMetric,
    # MSEDegradationMetric,
    # SurrogateFidelityMetric,
    # SurrogateFeatureStabilityMetric,
    WeightedAverageDepthMetric,
    WeightedAverageExplainabilityScoreMetric,
    WeightedTreeGiniMetric,
    TreeDepthVarianceMetric,
    TreeNumberOfRulesMetric,
    TreeNumberOfFeaturesMetric,
    XAIConsistencyScoreMetric,
)
from . import explainability_metrics_core as core


class ExplainabilityPillar(Pillar):
    """Explainability pillar based on SHAP-derived metrics."""

    @property
    def pillar_key(self) -> str:
        return "explainability"

    def prepare(self, context: EvaluationContext, config: dict[str, Any]) -> None:
        # Optional parameters (kept under mappings.explainability.params)
        params = (config or {}).get("params", {})

        n_samples = int(params.get("n_samples", 50))
        shap_threshold = float(params.get("shap_threshold", 1e-3))
        top_k = int(params.get("top_k", 5))
        seed = int(params.get("seed", 42))

        # Store params for lazy computation in metrics (and for reproducibility)
        context.extras["explainability_params"] = {
            "n_samples": n_samples,
            "shap_threshold": shap_threshold,
            "top_k": top_k,
            "seed": seed,
        }

        # Best-effort eager compute (but never let it crash the whole evaluation)
        try:
            metrics = core.shap_based_metrics(
                model=context.model,
                X=context.X_test,
                n_samples=n_samples,
                shap_threshold=shap_threshold,
                top_k=top_k,
                seed=seed,
            )
            context.extras["explainability_shap_metrics"] = metrics

            # Guardamos local_importances y base_values para Faithfulness / Monotonicity
            local_imps = np.asarray(metrics["local_importances"])
            context.extras["feature_weights"] = local_imps
            context.extras["base_values"] = metrics["base_values"]

            # ====================================================================
            # 1. Preparar Feature Names
            # ====================================================================
            if hasattr(context.X_test, "columns"):
                feature_names = context.X_test.columns.tolist()
                X_df = context.X_test
            else:
                feature_names = [f"x{i}" for i in range(context.X_test.shape[1])]
                X_df = pd.DataFrame(context.X_test, columns=feature_names)

            # ====================================================================
            # 2. Global Importances (Media absoluta de los SHAP values)
            # ====================================================================
            global_imps_array = np.abs(local_imps).mean(axis=0)
            global_importances = {feat: float(imp) for feat, imp in zip(feature_names, global_imps_array)}
            
            # ====================================================================
            # 3. Conditional Importances (Agrupado por la clase predicha)
            # ====================================================================
            y_pred = context.model.predict(context.X_test)
            # Si se usó submuestreo en shap_based_metrics, alineamos y_pred
            if len(y_pred) > len(local_imps):
                # Opcional: si tu shap_based_metrics hace subsampling, asegúrate 
                # de predecir solo sobre la muestra o tomar los primeros n_samples.
                y_pred = y_pred[:len(local_imps)]

            conditional_importances = {}
            for cls in np.unique(y_pred):
                idx = np.where(y_pred == cls)[0]
                if len(idx) > 0:
                    cls_imps_array = np.abs(local_imps[idx]).mean(axis=0)
                    conditional_importances[str(cls)] = {
                        feat: float(imp) for feat, imp in zip(feature_names, cls_imps_array)
                    }

            # ====================================================================
            # 4. Partial Dependencies Plots (PDP)
            # ====================================================================
            pdp_averages = {}

            # Calculamos PDP solo para el Top K de features para ahorrar cómputo
            top_features = sorted(global_importances.keys(), key=lambda k: global_importances[k], reverse=True)[:top_k]

            for feat in top_features:
                feat_idx = feature_names.index(feat)
                # kind="both" extrae tanto la media global (average) como las curvas individuales (ICE)
                pdp_res = partial_dependence(context.model, X_df, features=[feat_idx], kind="both")
                
                # Dependiendo de si es multiclase, scikit-learn puede devolver un array extra. 
                # Tomamos el índice [0] que suele corresponder a la clase positiva / principal.
                pdp_averages[feat] = pdp_res["average"][0].tolist()

            # ====================================================================
            # 5. Inyectar en context.extras
            # ====================================================================
            context.extras["global_importances"] = global_importances
            context.extras["conditional_importances"] = conditional_importances
            context.extras["pdp_averages"] = pdp_averages

        except Exception as exc:
            context.extras["explainability_error"] = str(exc)
            # Leave computation to metric wrappers (they will surface the error safely)

    def get_metrics(self) -> List[Any]:
        return [
            SparsityMetric(),
            FeatureEntropyMetric(),
            TopKConcentrationMetric(),
            AlgorithmClassMetric(),
            CorrelatedFeaturesMetric(),
            ModelSizeMetric(),
            FeatureRelevanceMetric(),
            # PerformanceDifferenceMetric(),
            NumberOfRulesMetric(),
            AverageRuleLengthMetric(),
            RuleStatsMetric(),
            TreeDepthMetric(),
            InteractionStrengthMetric(),
            FaithfulnessMetric(),
            MonotonicityMetric(),
            # ShapleyCorrMetric(),
            # ROARMetric(),
            InfidelityMetric(),
            AlphaImportanceScoreMetric(),
            XAIEaseScoreMetric(),
            PositionParityMetric(),
            RankAlignmentMetric(),
            SpreadRatioMetric(),
            SpreadDivergenceMetric(),
            # FluctuationRatioMetric(),
            # RankConsistencyMetric(),
            # ImportanceStabilityMetric(),
            # MSEDegradationMetric(),
            # SurrogateFidelityMetric(),
            # SurrogateFeatureStabilityMetric(),
            WeightedAverageDepthMetric(),
            WeightedAverageExplainabilityScoreMetric(),
            WeightedTreeGiniMetric(),
            TreeDepthVarianceMetric(),
            TreeNumberOfRulesMetric(),
            TreeNumberOfFeaturesMetric(),
            XAIConsistencyScoreMetric(),
        ]
