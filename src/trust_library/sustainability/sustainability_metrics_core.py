from __future__ import annotations

import time
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker


# ==========================================================
# TRAINING TRACKER (EXECUTED ONLY ONCE PER PILLAR)
# ==========================================================

def track_training_run(model, train_data, project_name="ML_Project") -> Dict[str, Any]:
    """
    Executes model training under CodeCarbon tracking
    and returns structured run metadata.
    """

    # import pickle

    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)

    # print(type(model))
    # print(model)
    tracker = EmissionsTracker(
        project_name=project_name,
        log_level=logging.ERROR,    # o INFO form more details
        output_file="emissions.csv",
        output_dir=".",
        save_to_file=True,
        tracking_mode='process',
        #pue=None,                  # PUE automaticaly detected
        #wue=None,                  # WUE automaticaly detected
        #country_iso_code="ESP"     # Specify country region if known, otherwise auto-detected
        measure_power_secs=15       # Measure power every 15 seconds (default)
    )

    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]

    tracker.start()
    start = time.time()

    model.fit(X, y)

    duration = time.time() - start
    tracker.stop()

    df = pd.read_csv("emissions.csv")
    last_run = df.iloc[-1]

    return {
        "energy_consumed": float(last_run.get("energy_consumed", 0.0)),
        "cpu_energy": float(last_run.get("cpu_energy", 0.0)),
        "gpu_energy": float(last_run.get("gpu_energy", 0.0)),
        "ram_energy": float(last_run.get("ram_energy", 0.0)),
        "emissions": float(last_run.get("emissions", 0.0)),
        "duration": float(duration),
        "country": last_run.get("country_name", "Unknown"),
        "pue": last_run.get("pue", "auto"),
        "wue": last_run.get("wue", 0.0),
    }


# ==========================================================
# METRIC COMPUTATIONS (PURE FUNCTIONS)
# ==========================================================

def compute_energy_consumption(run_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": run_data["energy_consumed"],
        "cpu_energy": run_data["cpu_energy"],
        "gpu_energy": run_data["gpu_energy"],
        "ram_energy": run_data["ram_energy"],
    }


def compute_emissions(run_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": run_data["emissions"],
        "duration": run_data["duration"],
        "pue": run_data["pue"],
        "wue": run_data["wue"],
    }


def compute_carbon_intensity(run_data: Dict[str, Any]) -> Dict[str, Any]:
    energy = run_data["energy_consumed"]
    emissions = run_data["emissions"]

    ci = emissions / energy if energy > 0 else 0.0

    return {
        "value": float(ci),
        "country": run_data["country"],
    }


# def compute_energy_efficiency(run_data: Dict[str, Any]) -> Dict[str, Any]:
#     duration_hours = run_data["duration"] / 3600.0
#     energy = run_data["energy_consumed"]

#     efficiency = energy / duration_hours if duration_hours > 0 else np.inf

#     return {
#         "value": float(efficiency),
#         "duration_hours": float(duration_hours),
#     }