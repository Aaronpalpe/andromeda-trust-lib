# core.py

from __future__ import annotations
import json
from typing import List

from trust_library import utils
from trust_library.fairness import FairnessPillar
from trust_library.accountability import AccountabilityPillar
from trust_library.privacy import PrivacyPillar
from trust_library.sustainability import SustainabilityPillar
from trust_library.explainability import ExplainabilityPillar
from trust_library.robustness import RobustnessPillar
from importlib import resources



# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if config_path is None:
        # Cargar config interna del paquete
        with resources.files("trust_library").joinpath("configs.json").open("r") as f:
            config = json.load(f)
    else:
        # Cargar config externa proporcionada por usuario
        with open(config_path, "r") as f:
            config = json.load(f)
    return config
# ─────────────────────────────────────────────────────────────────────────────
# Context preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_context(model, train_data, test_data, factsheet) -> utils.EvaluationContext:
    """Extracts features/labels and generates predictions to build the evaluation context."""
    target = factsheet["general"]["target_column"]["value"]

    if target not in train_data.columns or target not in test_data.columns:
        raise ValueError(f"Target column '{target}' not found in the datasets.")

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target].values.flatten()
    X_test  = test_data.drop(columns=[target])
    y_test  = test_data[target].values.flatten()

    try:
        y_pred_train = _predict(model, X_train)
        y_pred_test  = _predict(model, X_test)
        y_prob_train = _predict_proba(model, X_train)
        y_prob_test  = _predict_proba(model, X_test)
    except Exception as e:
        raise RuntimeError(f"Error during model prediction: {e}")

    return utils.EvaluationContext(
        model=model,
        train_data=train_data,
        test_data=test_data,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_prob_train=y_prob_train,
        y_prob_test=y_prob_test,
        factsheet=factsheet,
    )


def _predict(model, X):
    result = model.predict(X)
    return result.flatten() if hasattr(result, "flatten") else result


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pillar registry
# ─────────────────────────────────────────────────────────────────────────────

_PILLARS = {
    "fairness": FairnessPillar(),
    "accountability": AccountabilityPillar(),
    "privacy": PrivacyPillar(),
    "sustainability": SustainabilityPillar(),
    "explainability": ExplainabilityPillar(),
    "robustness": RobustnessPillar()
}


# ─────────────────────────────────────────────────────────────────────────────
# Pillar execution
# ─────────────────────────────────────────────────────────────────────────────

def run_pillars(context: utils.EvaluationContext, config: dict) -> dict:
    """Executes analyse() for each pillar and returns a Result per pillar."""
    results = {}
    for name, pillar in _PILLARS.items():
        print(f"Computing {name.capitalize()} metrics...")
        results[name] = pillar.analyse(
            context,
            config.get("mappings", {}).get(name),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pillar_scores(context: utils.EvaluationContext, config: dict, pillars: List[str] = None) -> dict:
    """Delegates aggregated score computation to each pillar.
    
    If `pillars` is provided, computes only for those pillar names (strings).
    If `pillars` is None, computes for all pillars.
    """

    if pillars is None:
        selected_items = _PILLARS.items()
    else:
        unknown = [p for p in pillars if p not in _PILLARS]
        if unknown:
            raise ValueError(
                f"Unknown pillar(s): {unknown}. Available: {list(_PILLARS.keys())}"
            )
        selected_items = ((name, _PILLARS[name]) for name in pillars)

    results = {}
    for name, pillar in selected_items:
        aggregated_score, result_obj = pillar.score(context, config)
        results[name] = {
            "score": aggregated_score,
            "metrics": result_obj.score,
            "properties": result_obj.properties,
        }

    return results


def compute_trust_score(pillar_scores: dict, pillar_weights: dict) -> float:
    return utils.calculate_weighted_score(pillar_scores, pillar_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def save_result(result: dict, path: str = "trust_evaluation_result.json") -> None:
    with open(path, "w") as f:
        json.dump(utils.to_json_safe(result), f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    train_data,
    test_data,
    factsheet,
    config_path: str | None = None,
    output_path: str = "trust_evaluation_result.json",
) -> dict:
    
    config  = load_config(config_path)
    context = build_context(model, train_data, test_data, factsheet)

    pillar_results = compute_pillar_scores(context, config)

    # Extract only aggretated scores
    pillar_scores = {
        name: data["score"]
        for name, data in pillar_results.items()
    }

    trust_score = compute_trust_score(
        pillar_scores,
        config.get("pillars", {})
    )

    score_explanation = build_score_explanation(pillar_results, config) # NEW

    output = {
        "trust_score": trust_score,
        "pillar_score": pillar_scores,
        "details": {
            name: data["metrics"]
            for name, data in pillar_results.items()
        },
        "properties": {
            name: data["properties"]
            for name, data in pillar_results.items()
        },
        "explanation": score_explanation, # NEW
    }

    save_result(output, output_path)
    return output



def build_score_explanation(pillar_results: dict, config: dict) -> dict:
    
    explanation = {}
    pillar_weights = config.get("pillars", {})

    trust_formula_parts = []
    trust_value = 0

    for pillar_name, data in pillar_results.items():
        p_weight = pillar_weights.get(pillar_name, 1)
        p_score  = data["score"]

        trust_value += p_weight * p_score

        trust_formula_parts.append(
            f"{p_weight}*{pillar_name.capitalize()}({p_score})"
        )

        # === Métricas internas del pilar ===
        metric_weights = config.get("weights", {}).get(pillar_name, {})
        metric_parts = []

        for metric_name, metric_value in data["metrics"].items():
            m_weight = metric_weights.get(metric_name, 1)
            metric_parts.append(
                f"{m_weight}*{metric_name}({metric_value})"
            )

        explanation[pillar_name] = {
            "formula": " + ".join(metric_parts),
            "score": p_score,
        }

    explanation["trust_score"] = {
        "formula": " + ".join(trust_formula_parts),
        "final_score": trust_value
    }

    return explanation

# ─────────────────────────────────────────────────────────────────────────────
# Optional facade - preserves compatibility with existing code
# ─────────────────────────────────────────────────────────────────────────────

class TrustEvaluator:
    """
    Thin wrapper around `evaluate()` for users already relying on the class-based API.
    Contains no internal logic: delegates everything to the module-level functions.
    """

    def __init__(self, model, train_data, test_data, factsheet):
        self.model       = model
        self.train_data  = train_data
        self.test_data   = test_data
        self.factsheet   = factsheet
        self.result: dict | None = None

    def compute(self) -> dict:
        self.result = evaluate(
            self.model, self.train_data, self.test_data,
            self.factsheet
        )
        return self.result
    
    def compute_pillars(self, pillars: list[str]) -> dict:
        config  = load_config(None)
        context = build_context(
            self.model,
            self.train_data,
            self.test_data,
            self.factsheet
        )

        results = {}

        for name in pillars:
            if name not in _PILLARS:
                raise ValueError(f"Pillar '{name}' is not registered.")

            pillar = _PILLARS[name]
            aggregated_score, result_obj = pillar.score(context, config)

            results[name] = {
                "score": aggregated_score,
                "metrics": result_obj.score,
                "properties": result_obj.properties,
            }

        return results