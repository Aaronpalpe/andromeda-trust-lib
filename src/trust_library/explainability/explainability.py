from __future__ import annotations

from multiprocessing import context
from typing import Any, List

from trust_library.pillar import Pillar
from trust_library.utils import EvaluationContext

import numpy as np
from sklearn.inspection import partial_dependence
import pandas as pd
import time

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
    TreeDepthMetric,
    # InteractionStrengthMetric,
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
        '''
        Context extra will provide:
        - explainability_params: the parameters used for explainability computations (n_samples, shap_threshold, top_k, seed, threshold_outlier, penalty_outlier, high_cor)
        - explainability_shap_metrics: the raw SHAP-based metrics computed
            - feature_weights: the local importances for each sample and feature (shape: n_samples x n_features)
            - base_values: the base values (mean of each feature) used for SHAP computations (shape: n_features)
        - global_importances: the mean absolute importance of each feature across all samples (shape: n_features)
        - conditional_importances: the mean absolute importance of each feature for each class (shape: n_classes x n_features). For rank alignment metric.
        - pdp_averages: the average prediction of a subset for each feature (shape: n_features)
        '''
        # Optional parameters (kept under mappings.explainability.params)
        params = (config or {}).get("params", {})

        n_samples = int(params.get("n_samples", 50))
        shap_threshold = float(params.get("shap_threshold", 1e-3))
        top_k = int(params.get("top_k", 5))
        seed = int(params.get("seed", 42))
        threshold_outlier = float(params.get("threshold_outlier_feature_relevance", 0.03))
        penalty_outlier = float(params.get("penalty_outlier_feature_relevance", 0))
        high_cor = float(params.get("high_cor_correlated_features", 0.95))


        # Store params for lazy computation in metrics (and for reproducibility)
        context.extras["explainability_params"] = {
            "n_samples": n_samples,
            "shap_threshold": shap_threshold,
            "top_k": top_k,
            "seed": seed,
            "threshold_outlier": threshold_outlier,
            "penalty_outlier": penalty_outlier,
            "high_cor": high_cor,
        }

        # Best-effort eager compute (but never let it crash the whole evaluation)
        t0 = time.time()
        # Local impotances: features importance for each individual prediction (shape: n_samples x n_features)
        local_imps = None
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
            local_imps = np.asarray(metrics["local_importances"])
            context.extras["feature_weights"] = local_imps
            global_imps_array = np.asarray(metrics["global_imps_array"])
            context.extras["base_values"] = metrics["base_values"]
        except Exception as shap_exc:
            context.extras["explainability_error"] = str(shap_exc)

        t1 = time.time()
        print(f"Explainability preparation took {t1 - t0:.2f} seconds")

        t0 = time.time()
        # Continue even if SHAP failed
        try:
            if hasattr(context.X_test, "columns"):
                feature_names = context.X_test.columns.tolist()
                X_df = context.X_test
            else:
                feature_names = [f"x{i}" for i in range(context.X_test.shape[1])]
                X_df = pd.DataFrame(context.X_test, columns=feature_names)

            # Global Importances: mean absolute importance of each feature across all samples (shape: n_features)
            if local_imps is not None:
                global_importances = {feat: float(imp) for feat, imp in zip(feature_names, global_imps_array)}
                context.extras["global_importances"] = global_importances
            else:
                global_importances = {}

            # Conditional Importances: mean absolute importance of each feature for each class (shape: n_classes x n_features)
            if local_imps is not None:
                y_pred = context.model.predict(context.X_test)
                if len(y_pred) > len(local_imps):
                    y_pred = y_pred[:len(local_imps)]
                conditional_importances = {}
                for cls in np.unique(y_pred):
                    idx = np.where(y_pred == cls)[0]
                    if len(idx) > 0:
                        cls_imps_array = np.abs(local_imps[idx]).mean(axis=0)
                        conditional_importances[str(cls)] = {
                            feat: float(imp) for feat, imp in zip(feature_names, cls_imps_array)
                        }
                context.extras["conditional_importances"] = conditional_importances
            else:
                context.extras["conditional_importances"] = {}

            # Partial Dependencies: average prediction for each feature (shape: n_features)
            pdp_averages = {}
            pdp_std = {}
            # # Only compute for top_k features to save time (and only if we have local importances to rank them, otherwise just take the first top_k)
            # if global_importances:
            #     top_features = sorted(global_importances.keys(), key=lambda k: global_importances[k], reverse=True)[:top_k]
            # else:
            #     top_features = feature_names[:top_k]

            # Calculate partial dependence for all features (but with low grid_resolution to save time), since we will use the std of the partial dependence for the XAI consistency metric, which requires all features to be computed.
            top_features = feature_names

            try:
                X_df_float = X_df.astype(float)
                # If the dataset is too large, sample a subset for partial dependence to save time
                X_df_float = X_df_float.sample(n=100, random_state=seed) if len(X_df_float) > 100 else X_df_float
                for feat in top_features:
                    try:
                        feat_idx = feature_names.index(feat)
                        try:
                            # Again, grid_resolution=10 to save time, and kind="average" to get the average prediction for each feature value (instead of the full dependence which can be more expensive)
                            pdp_res = partial_dependence(context.model, X_df_float, features=[feat_idx], kind="average", grid_resolution=10)
                            pdp_averages[feat] = pdp_res["average"][0].tolist()
                            pdp_std[feat] = np.std(pdp_res['average'])
                        except:
                            raise ValueError(f"partial_dependence with kind='average' failed for feature '{feat}'")      
                    except:
                        pass
            except:
                pass

            context.extras["pdp_averages"] = pdp_averages
            context.extras["pdp_std"] = pdp_std
            t1 = time.time()
            print(f"Additional explainability computations took {t1 - t0:.2f} seconds")

        except Exception as exc:
            context.extras["explainability_error"] = str(exc)

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

            # InteractionStrengthMetric(),
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
            
            FaithfulnessMetric(),
            MonotonicityMetric(),
            # ShapleyCorrMetric(),
            # ROARMetric(),
            InfidelityMetric(),

            NumberOfRulesMetric(),
            AverageRuleLengthMetric(),
            TreeDepthMetric(),
            WeightedAverageDepthMetric(),
            WeightedAverageExplainabilityScoreMetric(),
            WeightedTreeGiniMetric(),
            TreeDepthVarianceMetric(),
            TreeNumberOfFeaturesMetric(),
            XAIConsistencyScoreMetric(),
        ]
