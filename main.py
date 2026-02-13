import pandas as pd
import pickle
from trust_library.core import TrustEvaluator
from trust_library.factsheet import (
    load_factsheet_default,
    load_factsheet,
    save_factsheet, 
    create_factsheet_interactive,
    create_factsheet,
)

import json

from trust_library.utils import to_json_safe


if __name__ == "__main__":
    # factsheet = create_factsheet_interactive()
    # print(load_factsheet_default()) 
    # fs = create_factsheet({
    # "general": {
    #     "target_column": "Target"
    # },
    # "fairness": {
    #     "protected_feature": "Group",
    #     "protected_values": [1],      # Valor que identifica al grupo desprotegido
    #     "favorable_outcomes": [1]     # Valor que identifica el éxito
    # }
    # })
    # save_factsheet(fs, "factsheet.json")

    factsheet = load_factsheet("factsheet.json")

    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    train_loaded = pd.read_csv("train.csv")
    test_loaded = pd.read_csv("test.csv")

    evaluator = TrustEvaluator(
        model=loaded_model,
        train_data=train_loaded,
        test_data=test_loaded,
        factsheet=factsheet,
        config_path="trust_library/configs.json"
    )

    results = evaluator.compute()

    # print("\n=== RESULTADOS FINAL ===")
    # print(f"Global Trust Score: {results['trust_score']}")
    # print(f"Pillar Scores:\n{json.dumps(results['pillar_score'], indent=4)}")
    # print("\nDetalles (Scores brutos):\n" + json.dumps(results['details'], indent=4))

    # print(
    #     "\nPropiedades calculadas:\n" +
    #     json.dumps(to_json_safe(results['properties']), indent=4)
    # )
