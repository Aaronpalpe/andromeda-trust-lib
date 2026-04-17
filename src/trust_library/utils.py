import collections
from typing import Any, Optional
import numpy as np

from dataclasses import dataclass, field
import pandas as pd

from trust_library.base_metric import BaseMetric

# Result structure
Result = collections.namedtuple('result', 'score properties')


# =============================================================================
# CONVERSION HELPERS FOR DATAFRAME/NUMPY
# =============================================================================

def to_numpy(data: Any) -> np.ndarray:
    """
    Converts any data type to numpy array.
    Supports: pd.DataFrame, pd.Series, list, np.ndarray
    """
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    return np.asarray(data)


def to_dataframe(data: Any, columns: list = None) -> pd.DataFrame:
    """
    Converts any data type to pandas DataFrame.
    Supports: np.ndarray, pd.DataFrame, list
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, np.ndarray):
        if columns is None:
            columns = [f"x{i}" for i in range(data.shape[1] if data.ndim > 1 else 1)]
        return pd.DataFrame(data, columns=columns)
    if isinstance(data, (list, tuple)):
        return pd.DataFrame(data, columns=columns)
    return pd.DataFrame(data)


def to_series(data: Any, name: str = None) -> pd.Series:
    """
    Converts any data type to pandas Series.
    """
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, name=name)


def ensure_1d(data: Any) -> np.ndarray:
    """
    Ensures data is a 1D array (useful for y_train, y_test).
    """
    arr = to_numpy(data)
    if arr is None:
        return None
    return arr.flatten() if arr.ndim > 1 else arr


# =============================================================================
# VALUE FORMATTING (MAX 2 DECIMALS)
# =============================================================================

def format_value(value: Any, decimals: int = 2) -> Any:
    """
    Formats a numeric value to a maximum of N decimals.
    Returns the original value if not numeric.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), decimals)
    return value


def format_dict(d: dict, decimals: int = 2) -> dict:
    """
    Recursively formats all numeric values in a dict to N decimals.
    """
    if not isinstance(d, dict):
        return format_value(d, decimals)

    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = format_dict(v, decimals)
        elif isinstance(v, list):
            result[k] = [format_dict(item, decimals) if isinstance(item, dict)
                         else format_value(item, decimals) for item in v]
        else:
            result[k] = format_value(v, decimals)
    return result


# =============================================================================
# SCORING: THRESHOLDS AND MIN-MAX
# =============================================================================

def calculate_weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute a weighted average of the scores"""
    weighted_scores = []
    valid_weights = []

    for metric, score in scores.items():
        weight = weights.get(metric, 0)
        # Only sum if score is a valid number
        if score is not None and not np.isnan(score):
            weighted_scores.append(score * weight)
            valid_weights.append(weight)

    sum_weights = np.sum(valid_weights)

    if sum_weights == 0:
        return 0

    return round(np.sum(weighted_scores) / sum_weights, 2)


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
        return float(obj) #round(float(obj), 2)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj) # round(obj, 2)

    return obj


def calculate_score(value: float, thresholds: list[float]) -> int:
    """
    Compute a score from 1 to 5 based on the given value and thresholds.
    Detects whether the thresholds are for ascending (accuracy) or descending
    (error) metrics and calculates the score accordingly.
    """
    if not thresholds or len(thresholds) == 0:
        raise ValueError("Thresholds must be a non empty list.")

    value = abs(value) # Always work with absolute value

    # Case: Error (Lower is better). Ex: [0.075, 0.05, 0.01, 0]
    # Case: Accuracy (Higher is better). Ex: [0.8, 0.9, 0.95,0.99]
    idx = np.digitize(value, thresholds, right=False)
    score = idx + 1

    # Ensure that the score never exceeds 5 and is not less than 1
    return int(np.clip(score, 1, 5))


def calculate_score_normalized(
    value: float,
    min_val: float,
    max_val: float,
    higher_is_better: bool = True,
    min_score: float = 1.0,
    max_score: float = 5.0
) -> float:
    """
    Calculates a score using min-max normalization instead of thresholds.

    Args:
        value: The metric value to convert to score
        min_val: Expected minimum value of the metric
        max_val: Expected maximum value of the metric
        higher_is_better: If True, high values = high score. If False, low values = high score.
        min_score: Minimum score (default 1.0)
        max_score: Maximum score (default 5.0)

    Returns:
        Normalized score between min_score and max_score

    Example:
        # Accuracy: min=60% (score 1), max=100% (score 5)
        >>> calculate_score_normalized(0.79, 0.60, 1.00, higher_is_better=True)
        2.9

        # Error rate: min=0% (score 5), max=40% (score 1)
        >>> calculate_score_normalized(0.10, 0.00, 0.40, higher_is_better=False)
        4.0
    """
    if value is None or np.isnan(value):
        return np.nan

    # Avoid division by zero
    if max_val == min_val:
        return (min_score + max_score) / 2

    # Normalize to range [0, 1]
    normalized = (value - min_val) / (max_val - min_val)

    # Clip to range [0, 1]
    normalized = np.clip(normalized, 0, 1)

    # If lower is better, invert
    if not higher_is_better:
        normalized = 1 - normalized

    # Scale to score range
    score = min_score + normalized * (max_score - min_score)

    return round(float(score), 2)


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
            f"Incomplete configuration: missing 'protected_feature' or 'target_column'.\n"
            f"  protected_feature = '{protected_feature}'\n"
            f"  target_column     = '{target_column}'"
        )

    return protected_feature, protected_values, target_column, favorable_outcomes


def _coerce_label_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_unique_labels(*arrays: Any) -> list[Any]:
    labels: list[Any] = []
    for array in arrays:
        if array is None:
            continue
        flat = np.asarray(array, dtype=object).reshape(-1)
        for value in flat.tolist():
            coerced = _coerce_label_value(value)
            if pd.isna(coerced):
                continue
            if coerced not in labels:
                labels.append(coerced)
    return labels


def _infer_n_classes_from_probabilities(*probability_arrays: Any) -> int | None:
    for probability_array in probability_arrays:
        if probability_array is None:
            continue
        probabilities = np.asarray(probability_array)
        if probabilities.ndim == 1:
            return 2
        if probabilities.ndim >= 2 and probabilities.shape[1] > 0:
            return max(2, int(probabilities.shape[1]))
    return None


def infer_problem_type(
    *,
    model: Any,
    y_train: Any,
    y_test: Any,
    y_prob_train: Any = None,
    y_prob_test: Any = None,
) -> tuple[bool, BaseMetric.ProblemType | None, list[Any]]:
    """
    Infer whether the evaluation looks like a classification problem and, if so,
    whether it is binary or multiclass.

    This first version intentionally uses conservative heuristics centered on the
    runtime capabilities exposed by the model (`predict_proba`, `classes_`) and
    on the observed label/probability outputs.
    """
    class_labels: list[Any] = []

    if hasattr(model, "classes_"):
        try:
            classes = np.asarray(getattr(model, "classes_"), dtype=object).reshape(-1)
            class_labels = [_coerce_label_value(value) for value in classes.tolist()]
        except Exception:
            class_labels = []

    if not class_labels:
        class_labels = _collect_unique_labels(y_train, y_test)

    has_probability_outputs = y_prob_train is not None or y_prob_test is not None
    is_classification = bool(
        has_probability_outputs
        or hasattr(model, "predict_proba")
        or hasattr(model, "classes_")
    )

    if not is_classification:
        return False, None, class_labels

    n_classes = len(class_labels)
    if n_classes == 0:
        inferred_n_classes = _infer_n_classes_from_probabilities(y_prob_train, y_prob_test)
        n_classes = inferred_n_classes or 0

    if n_classes > 2:
        return True, BaseMetric.ProblemType.MULTICLASS, class_labels

    return True, BaseMetric.ProblemType.BINARY, class_labels


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
    is_classification: bool = False
    problem_type: BaseMetric.ProblemType | None = None
    class_labels: list[Any] = field(default_factory=list)
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
        if self.problem_type != BaseMetric.ProblemType.BINARY:
            return None

        probabilities = np.asarray(self.y_prob_test)
        if probabilities.ndim == 1:
            return probabilities
        if probabilities.ndim >= 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return None

    @property
    def is_binary_classification(self) -> bool:
        return self.problem_type == BaseMetric.ProblemType.BINARY

    @property
    def is_multiclass_classification(self) -> bool:
        return self.problem_type == BaseMetric.ProblemType.MULTICLASS
