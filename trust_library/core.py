# core.py

from __future__ import annotations
import json
import os

from trust_library import utils
from trust_library.fairness import FairnessPillar
from trust_library.accountability import AccountabilityPillar
from trust_library.privacy import PrivacyPillar
from trust_library.sustainability import SustainabilityPillar




# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found at: {config_path}")
    with open(config_path) as f:
        return json.load(f)


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

def compute_pillar_scores(context: utils.EvaluationContext, config: dict) -> dict:
    """Delegates aggregated score computation to each pillar."""

    results = {}

    for name, pillar in _PILLARS.items():
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
    config_path: str = "trust_library/configs.json",
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
    }

    save_result(output, output_path)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Optional facade — preserves compatibility with existing code
# ─────────────────────────────────────────────────────────────────────────────

class TrustEvaluator:
    """
    Thin wrapper around `evaluate()` for users already relying on the class-based API.
    Contains no internal logic: delegates everything to the module-level functions.
    """

    def __init__(self, model, train_data, test_data, factsheet,
                 config_path="trust_library/configs.json"):
        self.model       = model
        self.train_data  = train_data
        self.test_data   = test_data
        self.factsheet   = factsheet
        self.config_path = config_path
        self.result: dict | None = None

    def compute(self) -> dict:
        self.result = evaluate(
            self.model, self.train_data, self.test_data,
            self.factsheet, self.config_path,
        )
        return self.result