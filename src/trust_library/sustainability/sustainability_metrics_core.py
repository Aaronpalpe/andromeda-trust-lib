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

    Parameters
    ----------
    model: object
        The machine learning model to be trained (must implement fit()).
    train_data: pd.DataFrame
        Training data as a pandas DataFrame (features + target).
    project_name: str
        Name of the project for CodeCarbon tracking.

    Returns
    -------
    Dict[str, Any]: A dictionary containing:
        - energy_consumed: Total energy consumed during training (kWh).
        - cpu_energy: Energy consumed by CPU (kWh).
        - gpu_energy: Energy consumed by GPU (kWh).
        - ram_energy: Energy consumed by RAM (kWh).
        - emissions: Total CO2 emissions during training (kgCO2).
        - duration: Training duration in seconds.
        - country: Detected country of execution.
        - pue: Power Usage Effectiveness (PUE) value.
        - wue: Water Usage Effectiveness (WUE) value.
    """

    # import pickle

    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)

    # print(type(model))
    # print(model)
    tracker = EmissionsTracker(
        project_name=project_name,
        log_level=logging.ERROR,    # o INFO for more details
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

def energy_consumption(energy_conumed: float, cpu_energy: float, gpu_energy: float, ram_energy: float) -> Dict[str, Any]:
    '''
    Returns energy consumption metrics.
    
    Parameters
    ----------
    energy_consumed: float
        Total energy consumed during training (kWh).
    cpu_energy: float
        Energy consumed by CPU (kWh).
    gpu_energy: float
        Energy consumed by GPU (kWh).
    ram_energy: float
        Energy consumed by RAM (kWh).

    Returns
    -------
    Dict[str, Any]: A dictionary containing:
        - value: Total energy consumed during training (kWh).
        - cpu_energy: Energy consumed by CPU (kWh).
        - gpu_energy: Energy consumed by GPU (kWh).
        - ram_energy: Energy consumed by RAM (kWh).
    '''
    return {
        "value": energy_conumed,
        "cpu_energy": cpu_energy,
        "gpu_energy": gpu_energy,
        "ram_energy": ram_energy,
    }


def emissions(emissions: float, energy_consumed: float, duration: float, pue: float, wue: float) -> Dict[str, Any]:
    '''
    Returns emissions metrics.
    
    Parameters
    ----------
    emissions: float
        Total CO2 emissions during training (kgCO2).
    energy_consumed: float
        Total energy consumed during training (kWh).
    duration: float
        Training duration in seconds.
    pue: float
        Power Usage Effectiveness (PUE) value.
    wue: float
        Water Usage Effectiveness (WUE) value.

    Returns
    -------
    Dict[str, Any]: A dictionary containing:
        - value: Total CO2 emissions during training (kgCO2).
        - duration: Training duration in seconds.
        - pue: Power Usage Effectiveness (PUE) value.
        - wue: Water Usage Effectiveness (WUE) value.
    '''

    return {
        "value": emissions,
        "energy_consumed": energy_consumed,
        "duration": duration,
        "pue": pue,
        "wue": wue,
    }


def carbon_intensity(energy_consumed: float, emissions: float, country: str) -> Dict[str, Any]:
    '''
    Returns carbon intensity metrics.

    Parameters
    ----------
    energy_consumed: float
        Total energy consumed during training (kWh).
    emissions: float
        Total CO2 emissions during training (kgCO2).
    country: str
        Country of execution.

    Returns
    -------
    Dict[str, Any]: A dictionary containing:
        - value: Carbon intensity (kgCO2/kWh).
        - country: Country of execution.
    '''
    ci = emissions / energy_consumed if energy_consumed > 0 else 0.0

    if np.isnan(ci):
        raise ValueError("Carbon intensity computation resulted in NaN. Check if emissions and energy_consumed are valid.")
    return {
        "value": float(ci),
        "country": country,
    }
