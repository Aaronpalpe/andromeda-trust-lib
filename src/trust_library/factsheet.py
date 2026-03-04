import json
from pathlib import Path

from trust_library.utils import to_json_safe

import copy

DEFAULT_FACTSHEET_FILE_PATH = Path(__file__).parent / "factsheet.json"

def load_factsheet_default(path=DEFAULT_FACTSHEET_FILE_PATH):
    """Carga el archivo de factsheet por defecto."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_factsheet(path="factsheet.json"):
    """Carga el archivo de factsheet."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_factsheet(factsheet, path="factsheet.json"):
    """Guarda el archivo de factsheet."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(factsheet), f, indent=4, ensure_ascii=False)

def create_factsheet_interactive(output_path="factsheet.json"):
    """
    Crea una factsheet interactiva basada en la plantilla oficial.
    Solo permite modificar los campos 'value', preservando la estructura.
    """

    template = load_factsheet_default()
    factsheet = {}

    for section, fields in template.items():
        print(f"\n=== {section.upper()} ===")
        factsheet[section] = {}

        for field, meta in fields.items():
            print(f"\n{field}")
            print(f"  {meta.get('description', '')}")

            current_value = meta.get("value", None)
            if current_value is not None:
                print(f"  Valor actual: {current_value}")

            user_input = input("Nuevo valor (Enter para dejar vacío): ").strip()

            # Intento de parseo automático para listas o números
            if user_input:
                try:
                    parsed_value = json.loads(user_input)
                except json.JSONDecodeError:
                    parsed_value = user_input
            else:
                parsed_value = None

            factsheet[section][field] = {
                "description": meta.get("description", ""),
                "value": parsed_value
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(factsheet, f, indent=4, ensure_ascii=False)

    print(f"\nFactsheet guardada en {output_path}")
    return factsheet


def create_factsheet(initial_values=None):
    """
    Crea una factsheet a partir de la plantilla oficial.

    Args:
        initial_values (dict): valores a sobreescribir en la plantilla

    Returns:
        dict: factsheet válida
    """
    factsheet = copy.deepcopy(load_factsheet_default())

    if initial_values:
        _deep_update(factsheet, initial_values)

    return factsheet

def _deep_update(base, updates):
    for section, fields in updates.items():
        if section in base:
            for field, value in fields.items():
                if field in base[section]:
                    base[section][field]["value"] = value


# def validate_factsheet_structure(factsheet):
#     template = get_factsheet_template()
#     errors = []

#     for section in template:
#         if section not in factsheet:
#             errors.append(f"Missing section: {section}")

#     if errors:
#         raise ValueError("Invalid factsheet:\n" + "\n".join(errors))
