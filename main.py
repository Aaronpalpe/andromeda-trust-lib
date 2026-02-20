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
import plotly.express as px

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

    factsheet = load_factsheet_default() #load_factsheet("factsheet.json")

    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    train_loaded = pd.read_csv("train.csv")
    test_loaded = pd.read_csv("test.csv")

    evaluator = TrustEvaluator(
        model=loaded_model,
        train_data=train_loaded,
        test_data=test_loaded,
        factsheet=factsheet
    )

    #results = evaluator.compute()

    # print("\n=== RESULTADOS FINAL ===")
    # print(f"Global Trust Score: {results['trust_score']}")
    # print(f"Pillar Scores:\n{json.dumps(results['pillar_score'], indent=4)}")
    # print("\nDetalles (Scores brutos):\n" + json.dumps(results['details'], indent=4))

    # print(
    #     "\nPropiedades calculadas:\n" +
    #     json.dumps(to_json_safe(results['properties']), indent=4)
    # )


    # with open("trust_evaluation_result.json", "r") as f:
    #     trust_results = json.load(f)
    # pillar_scores = trust_results["pillar_score"]

    # df_pillars = pd.DataFrame(
    #     list(pillar_scores.items()),
    #     columns=["Pillar", "Score"]
    # )

    # plt.figure(figsize=(8, 5))
    # sns.barplot(data=df_pillars, x="Pillar", y="Score")
    # plt.title("Trust Pillar Scores")
    # plt.ylim(0, 5)
    # plt.tight_layout()
    # plt.show()

    # details = trust_results["details"]

    # for pillar, metrics in details.items():
    #     df = pd.DataFrame(
    #         list(metrics.items()),
    #         columns=["Metric", "Score"]
    #     )

    #     plt.figure(figsize=(10, 6))
    #     sns.barplot(data=df, x="Score", y="Metric")
    #     plt.title(f"{pillar.capitalize()} Metrics")
    #     plt.xlim(0, 5)
    #     plt.tight_layout()
    #     plt.show()

    # === Cargar datos ===
    with open("trust_evaluation_result.json", "r") as f:
        trust_results = json.load(f)

    pillar_scores = trust_results["pillar_score"]
    details = trust_results["details"]

    # === Gráfico de barras de pilares ===
    df_pillars = pd.DataFrame(list(pillar_scores.items()), columns=["Pillar", "Score"])
    fig_pillars = px.bar(
        df_pillars,
        x="Pillar",
        y="Score",
        text="Score",
        color="Score",
        color_continuous_scale="Viridis",
        range_y=[0, 5],
        title="Trust Pillar Scores"
    )
    fig_pillars.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_pillars.show()  # Esto abre un navegador con el gráfico interactivo

    # === Gráficos por métricas individuales de cada pilar ===
    for pillar, metrics in details.items():
        df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
        df_metrics = df_metrics.sort_values("Score")  # Para que las barras horizontales estén ordenadas
        fig = px.bar(
            df_metrics,
            x="Score",
            y="Metric",
            orientation="h",
            text="Score",
            color="Score",
            color_continuous_scale="Cividis",
            range_x=[0, 5],
            title=f"{pillar.capitalize()} Metrics"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.show()  # Abre cada gráfico en una ventana o pestaña del navegador

