import collections
from typing import Any, Optional
import numpy as np

from dataclasses import dataclass, field
import pandas as pd

# Estructura de resultados
Result = collections.namedtuple('result', 'score properties')

def calculate_weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute a weighted average of the scores"""
    weighted_scores = []
    valid_weights = []

    for metric, score in scores.items():
        weight = weights.get(metric, 0)
        # Solo sumamos si el score es un número válido
        if score is not None and not np.isnan(score):
            weighted_scores.append(score * weight)
            valid_weights.append(weight)
    
    sum_weights = np.sum(valid_weights)
    
    if sum_weights == 0:
        return 0
    
    return round(np.sum(weighted_scores) / sum_weights, 1)

def to_json_safe(obj: Any) -> Any:
    """Convert object to a JSON-serializable format, handling common non-serializable types."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)

    return obj


def calculate_score(value: float, thresholds: list[float]) -> int:
    """
    Compute a score from 1 to 5 based on the given value and thresholds.
    Detects whether the thresholds are for ascending (accuracy) or descending 
    (error) metrics and calculates the score accordingly.
    """
    if not thresholds or len(thresholds) == 0:
        raise ValueError("Thresholds must be a non empty list.")
        
    value = abs(value) # Trabajamos siempre con valor absoluto
    
    # Caso: Error (Menor es mejor). Ej: [0.075, 0.05, 0.01, 0]
    # Caso: Accuracy (Mayor es mejor). Ej: [0.8, 0.9, 0.95,0.99]
    idx = np.digitize(value, thresholds, right=False)
    score = idx + 1
        
    # Asegurar que el score nunca exceda 5 ni baje de 1
    return int(np.clip(score, 1, 5))


def load_fairness_config(factsheet: dict) -> tuple:
    '''
    Extracts the fairness configuration from the factsheet, ensuring that protected_feature and target_column are present.
    Returns a tuple of (protected_feature : String, protected_values : List, target_column : String, favorable_outcomes : List).
    '''
    fairness_section = factsheet.get("fairness", {})
    general_section  = factsheet.get("general", {})

    def _get(section, field, default):
        return section.get(field, {}).get("value") or default

    protected_feature = _get(fairness_section, "protected_feature", "")

    raw = fairness_section.get("protected_values", {}).get("value")
    if raw is None:
        protected_values = []
    elif isinstance(raw, list):
        protected_values = raw
    else:
        protected_values = [raw]

    favorable_outcomes = _get(fairness_section, "favorable_outcomes", []) or []
    target_column      = _get(general_section, "target_column", "")

    if not protected_feature or not target_column:
        raise ValueError(
            f"Configuración incompleta: falta 'protected_feature' o 'target_column'.\n"
            f"  protected_feature = '{protected_feature}'\n"
            f"  target_column     = '{target_column}'"
        )

    return protected_feature, protected_values, target_column, favorable_outcomes


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationContext
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationContext:
    model: Any
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred_train: np.ndarray | pd.DataFrame
    y_pred_test: np.ndarray | pd.DataFrame
    y_prob_train: np.ndarray | pd.DataFrame | None
    y_prob_test: np.ndarray | pd.DataFrame | None
    factsheet: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def group_mask(self) -> np.ndarray:
        ''' Returns a boolean mask indicating which samples in the test set belong to the protected group, based on the factsheet configuration. '''
        prot, vals, _, _ = load_fairness_config(self.factsheet)
        return self.X_test[prot].isin(vals).to_numpy()

    @property
    def protected_feature(self) -> str:
        ''' Returns the name of the protected feature as specified in the factsheet. '''
        return load_fairness_config(self.factsheet)[0]

    @property
    def protected_values(self) -> list[Any]:
        ''' Returns the list of protected values for the protected feature, as specified in the factsheet. '''
        return load_fairness_config(self.factsheet)[1]

    @property
    def target_column(self) -> str:
        ''' Returns the name of the target column as specified in the factsheet. '''
        return load_fairness_config(self.factsheet)[2]

    @property
    def favorable_outcomes(self) -> list[Any]:
        ''' Returns the list of favorable outcomes for the target variable, as specified in the factsheet. '''
        return load_fairness_config(self.factsheet)[3]

    @property
    def y_prob_positive(self) -> np.ndarray | None:
        ''' Returns the predicted probabilities for the positive class, if available. '''
        if self.y_prob_test is None:
            return None
        return np.asarray(self.y_prob_test)[:, 1]