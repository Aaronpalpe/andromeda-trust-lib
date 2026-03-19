# core.py

from __future__ import annotations
import json
from typing import List

import os
import random
import numpy as np

from trust_library import utils
from trust_library.fairness import FairnessPillar
from trust_library.accountability import AccountabilityPillar
from trust_library.privacy import PrivacyPillar
from trust_library.sustainability import SustainabilityPillar
from trust_library.explainability import ExplainabilityPillar
from trust_library.robustness import RobustnessPillar
from importlib import resources

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PILLARS = {
    "fairness": FairnessPillar(),
    "accountability": AccountabilityPillar(),
    "privacy": PrivacyPillar(),
    "sustainability": SustainabilityPillar(),
    "explainability": ExplainabilityPillar(),
    "robustness": RobustnessPillar(),
}


class TrustEvaluator:

    def __init__(
        self,
        model,
        train_data,
        test_data,
        factsheet,
        config_path: str | None = None,
        output_path: str = "trust_evaluation_result.json",
    ):
        self.model       = model
        self.train_data  = train_data
        self.test_data   = test_data
        self.factsheet   = factsheet
        self.output_path = output_path
        self.config      = self._load_config(config_path)
        self.context     = self._build_context()
        self.result: dict | None = None
        self.set_global_seed(42) # Config hay 2 NEW

    def set_global_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, pillars: list[str]= None, show_nan: bool = False) -> dict:
        """Full evaluation: all pillars + trust score + explanation."""
        if pillars is None:
            pillars = list(_PILLARS.keys())
        self._validate_pillars(pillars)
        pillar_results = self._compute_pillar_scores(pillars) # Diccionario compuesto por cada pilar, con su score agregado, sus métricas:score y metricas:propiedades.

        pillar_scores = {name: data["score"] for name, data in pillar_results.items()} # Pillar:score
        trust_score   = self._compute_trust_score(pillar_scores) # Trust global
        explanation   = self._build_score_explanation(pillar_results, show_nan=show_nan) # Fórmula

        clean_metrics = {}
        clean_properties = {}

        for pillar, data in pillar_results.items():

            metrics = {
                m: v for m, v in data["metrics"].items()
                if show_nan or not self._is_nan(v)
            }

            properties = {
                m: p for m, p in data["properties"].items()
                if m in metrics
            }
            clean_metrics[pillar] = metrics
            clean_properties[pillar] = properties

        self.result = {
            "trust_score":  trust_score,
            "pillar_score": pillar_scores,
            "details":      clean_metrics,
            "properties":   clean_properties,
            "explanation":  explanation,
        }

        self._save_result()
        return self.result

    # def evaluate_pillars(self, pillars: list[str]) -> dict:
    #     """Partial evaluation: only the requested pillars."""
    #     self._validate_pillars(pillars)
    #     return self._compute_pillar_scores(pillars)

    # def run_analysis(self) -> dict: #¿Se puede quitar? Creo que ya está representado ocn los 2 de arriba
    #     """Runs analyse() on every pillar and returns raw pillar results."""
    #     return {
    #         name: pillar.analyse(
    #             self.context,
    #             self.config.get("mappings", {}).get(name),
    #         )
    #         for name, pillar in _PILLARS.items()
    #     }
    
    def plot_results(self) -> None:
        """Display pillar scores and metric-level charts using Plotly."""
        if self.result is None:
            raise RuntimeError("No results available. Run evaluate() first.")

        pillar_scores = self.result["pillar_score"]
        details = self.result["details"]

        # Colores fijos por pilar
        pillar_colors = {
            "fairness": "#1f77b4",
            "accountability": "#b92323",
            "privacy": "#CDCA22",
            "sustainability": "#329729",
            "explainability": "#c37a1b",
            "robustness": "#8c564b",
        }

        # ─────────────────────────────────────
        # Pillar bar chart
        # ─────────────────────────────────────
        df_pillars = pd.DataFrame(
            list(pillar_scores.items()),
            columns=["Pillar", "Score"]
        )

        df_pillars["Color"] = df_pillars["Pillar"].map(pillar_colors)

        fig_pillars = px.bar(
            df_pillars,
            x="Pillar",
            y="Score",
            text="Score",
            range_y=[0, 5],
            title="Trust Pillar Scores",
        )

        fig_pillars.update_traces(
            marker_color=df_pillars["Color"],
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig_pillars.show()

        # ─────────────────────────────────────
        # Metric charts per pillar
        # ─────────────────────────────────────
        for pillar, metrics in details.items():

            df_metrics = pd.DataFrame(
                list(metrics.items()),
                columns=["Metric", "Score"]
            ).sort_values("Score")

            fig = px.bar(
                df_metrics,
                x="Score",
                y="Metric",
                orientation="h",
                text="Score",
                color="Score",
                color_continuous_scale="Viridis",
                range_x=[0, 5],
                title=f"{pillar.capitalize()} Metrics"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.show()
                
    @staticmethod
    def compare_all_bars(results: dict[str, dict]) -> None:
        """
        Compare trust score, pillars, and metrics of multiple models
        using separate bar charts.
        """

        # ────────────────
        # Trust Score
        # ────────────────
        trust_rows = []
        for model_name, result in results.items():
            trust_rows.append({
                "Model": model_name,
                "Trust Score": result["trust_score"]
            })
        df_trust = pd.DataFrame(trust_rows)

        fig_trust = px.bar(
            df_trust,
            x="Model",
            y="Trust Score",
            color="Model",
            text="Trust Score",
            range_y=[0, 5],
            title="Trust Score Comparison"
        )
        fig_trust.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_trust.show()

        # ────────────────
        # Pillars
        # ────────────────
        pillar_rows = []
        for model_name, result in results.items():
            for pillar, score in result["pillar_score"].items():
                pillar_rows.append({
                    "Model": model_name,
                    "Pillar": pillar.capitalize(),
                    "Score": score
                })
        df_pillars = pd.DataFrame(pillar_rows)

        fig_pillars = px.bar(
            df_pillars,
            x="Pillar",
            y="Score",
            color="Model",
            barmode="group",
            text="Score",
            range_y=[0, 5],
            title="Pillar Score Comparison"
        )
        fig_pillars.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_pillars.show()

        # ────────────────
        # Metrics per pillar
        # ────────────────
        pillars = set()
        for result in results.values():
            pillars.update(result["details"].keys())

        for pillar in sorted(pillars):
            metric_rows = []
            for model_name, result in results.items():
                metrics = result["details"].get(pillar, {})
                for metric, score in metrics.items():
                    if score is None:
                        continue
                    metric_rows.append({
                        "Model": model_name,
                        "Metric": metric,
                        "Score": score
                    })
            if not metric_rows:
                continue
            df_metrics = pd.DataFrame(metric_rows)

            fig_metrics = px.bar(
                df_metrics,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                text="Score",
                range_y=[0, 5],
                title=f"{pillar.capitalize()} Metrics Comparison"
            )
            fig_metrics.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_metrics.show()

    def plot_radar(self) -> None:
        """Display radar chart for pillar scores."""
        if self.result is None:
            raise RuntimeError("No results available. Run evaluate() first.")

        pillar_scores = self.result["pillar_score"]

        categories = list(pillar_scores.keys())
        values = list(pillar_scores.values())

        # cerrar el polígono
        categories += [categories[0]]
        values += [values[0]]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Model Score"
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            title="Trust Pillar Radar Chart",
            showlegend=False
        )

        fig_radar.show()

    @staticmethod
    def compare_radar(results: dict[str, dict]) -> None:
        """
        Compare multiple models in a radar chart.

        Parameters
        ----------
        results : dict
            Dictionary mapping model names to evaluation results.

            Example:
            {
                "RandomForest": result1,
                "XGBoost": result2
            }
        """

        fig = go.Figure()

        for model_name, result in results.items():

            pillar_scores = result["pillar_score"]

            categories = list(pillar_scores.keys())
            values = list(pillar_scores.values())

            # cerrar polígono
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=model_name,
                opacity=0.4
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            title="Trust Pillar Comparison",
            showlegend=True
        )

        fig.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_config(self, config_path: str | None) -> dict:
        if config_path is None:
            with resources.files("trust_library").joinpath("configs.json").open("r", encoding="utf-8") as f:
                return json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_context(self) -> utils.EvaluationContext:
        target = self.factsheet["general"]["target_column"]["value"]

        if target not in self.train_data.columns or target not in self.test_data.columns:
            raise ValueError(f"Target column '{target}' not found in the datasets.")

        X_train = self.train_data.drop(columns=[target])
        y_train = self.train_data[target].values.flatten()
        X_test  = self.test_data.drop(columns=[target])
        y_test  = self.test_data[target].values.flatten()

        try:
            y_pred_train = self._predict(X_train)
            y_pred_test  = self._predict(X_test)
            y_prob_train = self._predict_proba(X_train)
            y_prob_test  = self._predict_proba(X_test)
        except Exception as e:
            raise RuntimeError(f"Error during model prediction: {e}")

        return utils.EvaluationContext(
            model=self.model,
            train_data=self.train_data,
            test_data=self.test_data,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_prob_train=y_prob_train,
            y_prob_test=y_prob_test,
            factsheet=self.factsheet,
        )

    def _predict(self, X):
        result = self.model.predict(X)
        return result.flatten() if hasattr(result, "flatten") else result

    def _predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def _compute_pillar_scores(self, pillars: list[str] | None = None) -> dict:
        items = (
            ((name, _PILLARS[name]) for name in pillars)
            if pillars
            else _PILLARS.items()
        )
        results = {}
        for name, pillar in items:
            print(f"Computing {name.capitalize()} metrics...")
            aggregated_score, result_obj = pillar.score(self.context, self.config)
            results[name] = {
                "score":      aggregated_score,
                "metrics":    result_obj.score,
                "properties": result_obj.properties,
            }
        return results

    def _compute_trust_score(self, pillar_scores: dict) -> float:
        return utils.calculate_weighted_score(
            pillar_scores,
            self.config.get("pillars", {}),
        )

    def _build_score_explanation(self, pillar_results: dict, show_nan: bool = False) -> dict:
        explanation      = {}
        pillar_weights   = self.config.get("pillars", {})
        trust_formula_parts = []
        trust_value      = 0

        for pillar_name, data in pillar_results.items():
            p_weight = pillar_weights.get(pillar_name, 1)
            p_score  = data["score"]
            trust_value += p_weight * p_score
            trust_formula_parts.append(f"{p_weight}*{pillar_name.capitalize()}({p_score})")

            metric_weights = self.config.get("weights", {}).get(pillar_name, {})
            metric_parts   = [
                f"{metric_weights.get(metric_name, 1)}*{metric_name}({metric_value})"
                for metric_name, metric_value in data["metrics"].items()
                if not self._is_nan(metric_value) or show_nan
            ]

            explanation[pillar_name] = {
                "formula": " + ".join(metric_parts),
                "score":   p_score,
            }

        explanation["trust_score"] = {
            "formula":     " + ".join(trust_formula_parts),
            "final_score": trust_value,
        }
        return explanation

    def _save_result(self) -> None:
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(utils.to_json_safe(self.result), f, indent=4, ensure_ascii=False)

    def _validate_pillars(self, pillars: list[str]) -> None:
        unknown = [p for p in pillars if p not in _PILLARS]
        if unknown:
            raise ValueError(f"Unknown pillar(s): {unknown}. Available: {list(_PILLARS.keys())}")
        
    def _is_nan(self, value):
        if value is None:
            return True
        if isinstance(value, float):
            return math.isnan(value)
        if isinstance(value, str):
            return value.lower() == "nan"
        return False
    