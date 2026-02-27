import json
from pathlib import Path

from trust_library.utils import to_json_safe

DEFAULT_CONFIG_FILE_PATH = Path(__file__).parent / "configs.json"

def load_config_default(config_path=DEFAULT_CONFIG_FILE_PATH):
    """Carga el archivo de configuración por defecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(config_path="configs.json"):
    """Carga el archivo de configuración."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_config(config, config_path="configs.json"):
    """Guarda el archivo de configuración."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(config), f, indent=4, ensure_ascii=False)

def set_pillar_weight(config, pillar_name, new_weight):
    """
    Modifica el peso de un pilar.
    """
    if pillar_name not in config["pillars"]:
        raise ValueError(f"Pilar '{pillar_name}' no existe.")

    total = sum(config["pillars"].values()) - config["pillars"][pillar_name] + new_weight
    if total != 1:
        raise ValueError("La suma de pesos debe ser 1. Cambio no aplicado.")
    else:
        config["pillars"][pillar_name] = float(new_weight)
    return config

def set_metric_weight(config, pillar_name, metric_name, new_weight):
    """
    Modifica el peso de una métrica dentro de un pilar.
    """
    if pillar_name not in config["weights"]:
        raise ValueError(f"Pilar '{pillar_name}' no existe.")

    if metric_name not in config["weights"][pillar_name]:
        raise ValueError(f"Métrica '{metric_name}' no existe en '{pillar_name}'.")

    total = sum(config["weights"][pillar_name].values()) - config["weights"][pillar_name][metric_name] + new_weight
    if total != 1:
        raise ValueError("La suma de pesos dentro del pilar debe ser 1. Cambio no aplicado.")
    else:
        config["weights"][pillar_name][metric_name] = float(new_weight)
    return config

def set_metric_thresholds(config, pillar_name, score_metric_name, new_thresholds):
    """
    Modifica los thresholds de una métrica concreta.

    new_thresholds debe ser una lista ordenada de longitud 4.
    """
    if len(new_thresholds) != 4:
        raise ValueError("new_thresholds debe tener longitud 4.")
    try:
        config["mappings"][pillar_name][score_metric_name]["thresholds"]["value"] = list(new_thresholds)
    except KeyError:
        raise ValueError(
            f"No se encontraron thresholds para {score_metric_name} en {pillar_name}"
        )

    return config

def set_metric_mapping_values(config, pillar_name, score_metric_name, new_mapping_dict):
    """
    Modifica un mapping categórico (por ejemplo normalization o missing_data).

    new_mapping_dict debe ser un diccionario con formato {categoría: score} de longitud 5.
    """
    if len(new_mapping_dict) != 5:
        raise ValueError("new_mapping_dict debe tener longitud 5.")
    try:
        config["mappings"][pillar_name][score_metric_name]["mappings"]["value"] = dict(new_mapping_dict)
    except KeyError:
        raise ValueError(
            f"No se encontraron mappings para {score_metric_name} en {pillar_name}"
        )

    return config