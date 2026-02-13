import collections
import numpy as np

# Estructura de resultados
Result = collections.namedtuple('result', 'score properties')

def calculate_weighted_score(scores, weights):
    """Calcula el score ponderado"""
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

def to_json_safe(obj):
    """Convierte recursivamente objetos numpy/pandas a JSON."""
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


def calculate_score(value, thresholds):
    """
    Calcula el score 1-5 de forma robusta, detectando si los umbrales
    son ascendentes (accuracy) o descendentes (errores).
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