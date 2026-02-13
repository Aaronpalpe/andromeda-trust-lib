import time
import logging
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from .utils import Result, calculate_score

# === TRACKER HELPERS ===

def run_with_tracker(model, train_data, project_name="ML_Project"):
    """
    Ejecuta una función de entrenamiento con CodeCarbon
    y devuelve el último registro del CSV.
    """
    # import pickle

    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)

    # print(type(model))
    # print(model)
    tracker = EmissionsTracker(
        project_name=project_name,
        log_level=logging.ERROR,    # o INFO para más detalles
        output_file="emissions.csv",
        output_dir=".",
        save_to_file=True,
        tracking_mode='process',    # 'process' para medir solo este proceso
        #pue=None,                  # Usar PUE detectado automáticamente
        #wue=None,                  # Usar WUE detectado automáticamente
        #country_iso_code="ESP"     # Si offline, region específica
        measure_power_secs=15       # Medir consumo cada 15 segundos, por defecto
    )

    X = train_data.iloc[:, :-1]  # todas las columnas menos la última
    y = train_data.iloc[:, -1]   # última columna

    tracker.start()
    start = time.time()

    model.fit(X, y)

    duration = time.time() - start
    tracker.stop()

    df = pd.read_csv("emissions.csv")
    last_run = df.iloc[-1]      # última fila
    last_run["duration_real"] = duration
    return last_run

# === MAIN ANALYSE ===

def analyse(model, train_data, test_data, factsheet, config):
    """
    Analiza la sostenibilidad del modelo usando CodeCarbon y calcula las métricas.
    """

    def get_thresh(key):
        return config.get(key, {}).get("thresholds", {}).get("value", [])

    # Thresholds
    th_emissions = get_thresh("score_emissions")
    th_energy = get_thresh("score_energy")
    th_ci = get_thresh("score_carbon_intensity")
    # th_eff = get_thresh("score_energy_efficiency")

    # Run training
    run = run_with_tracker(model, train_data)

    output = {
        "energy_consumed": energy_score(run, th_energy),
        "carbon_intensity": carbon_intensity_score(run, th_ci),
        "emissions": emissions_score(run, train_data, th_emissions),
        # "energy_efficiency": efficiency_score(run, th_eff), #NUEVA
    }

    scores = {k: v.score for k, v in output.items()}
    props = {k: v.properties for k, v in output.items()}

    return Result(score=scores, properties=props)

# === METRIC FUNCTIONS ===

def emissions_score(run, train_data, thresholds):
    try:
        val = run["emissions"]  # kgCO2
        score = calculate_score(val, thresholds)

        #duration_h = run['duration_real'] / 3600.0  # Tiempo en horas
        pue = run.get("pue", "auto")
        # wue puede ser NaN si no está configurado, usamos 0 o valor por defecto
        wue = run.get('wue', 0) if not pd.isna(run.get('wue')) else 0.0

        props = {
            "Metric Description": "Total CO2 emissions during training (Energy consumed * Carbon Intensity).",
            "Emissions (kgCO2)": f"{val:.6f}",
            "Other metadata": {
                "Duration (s)": f"{run['duration_real']:.2f}",
                "Dataset Size": f"{len(train_data)} samples", #  ({train_data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB)
                "PUE": f"{pue}",
                "WUE": f"{wue}",
            },
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def energy_score(run, thresholds):
    try:
        val = run["energy_consumed"]  # kWh
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": "Total energy consumed (CPU + GPU + RAM).",
            "Energy Consumed (kWh)": f"{val:.6f}",
            "CPU Energy (kWh)": f"{run.get('cpu_energy', 0):.6f}",
            "GPU Energy (kWh)": f"{run.get('gpu_energy', 0):.6f}",
            "RAM Energy (kWh)": f"{run.get('ram_energy', 0):.6f}",
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def carbon_intensity_score(run, thresholds):
    try:
        energy = run["energy_consumed"]
        emissions = run["emissions"]

        ci = emissions / energy if energy > 0 else 0.0
        score = calculate_score(ci, thresholds)

        props = {
            "Metric Description": "Carbon intensity (kgCO2/kWh). Based on the type of energy used and location.",
            "Carbon Intensity": f"{ci:.4f}",
            "Country": run.get("country_name", "Unknown"),
            #"PUE": run.get("pue", "auto"),
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


# def efficiency_score(run, thresholds):
#     try:
#         duration_h = run["duration_real"] / 3600
#         energy = run["energy_consumed"]

#         eff = energy / duration_h if duration_h > 0 else np.inf
#         score = calculate_score(eff, thresholds)

#         props = {
#             "Metric Description": "Energy per training hour (kWh/h).",
#             "Energy Efficiency": f"{eff:.4f} kWh/h",
#             "Training Time (h)": f"{duration_h:.4f}",
#         }

#         return Result(score, props)

#     except Exception as e:
#         return Result(np.nan, {"Error": str(e)})
