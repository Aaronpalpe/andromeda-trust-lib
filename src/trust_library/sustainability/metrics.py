from __future__ import annotations

from typing import Dict, Any
from trust_library.base_metric import BaseMetric
from trust_library.utils import calculate_score
from . import sustainability_metrics_core as core


# ==========================================================
# Energy Consumption
# ==========================================================

class EnergyConsumptionMetric(BaseMetric):

    def __init__(self):
        super().__init__("energy_consumed", "score_energy")

    def compute(self, ctx):
        # run_data = ctx.extras["run_data"]
        energy_consumed = ctx.factsheet.get("sustainability", {}).get("energy_consumed", {}).get("value", 0.0)
        cpu_energy = ctx.factsheet.get("sustainability", {}).get("cpu_energy", {}).get("value", 0.0)
        gpu_energy = ctx.factsheet.get("sustainability", {}).get("gpu_energy", {}).get("value", 0.0)
        ram_energy = ctx.factsheet.get("sustainability", {}).get("ram_energy", {}).get("value", 0.0)

        return core.energy_consumption(energy_consumed, cpu_energy, gpu_energy, ram_energy)

    def build_properties(self, raw):
        return {
            "Metric Description": "Total energy consumed during training (CPU + GPU + RAM).",
            "Energy Consumed (kWh)": f"{raw['value']:.6f}",
            "CPU Energy (kWh)": f"{raw['cpu_energy']:.6f}",
            "GPU Energy (kWh)": f"{raw['gpu_energy']:.6f}",
            "RAM Energy (kWh)": f"{raw['ram_energy']:.6f}",
        }


# ==========================================================
# Emissions
# ==========================================================

class EmissionsMetric(BaseMetric):

    def __init__(self):
        super().__init__("emissions", "score_emissions")

    def compute(self, ctx):
        # run_data = ctx.extras["run_data"]
        emissions = ctx.factsheet.get("sustainability", {}).get("emissions", {}).get("value", 0.0)
        duration = ctx.factsheet.get("sustainability", {}).get("duration", {}).get("value", 0.0)
        pue = ctx.factsheet.get("sustainability", {}).get("pue", {}).get("value", 0.0)
        wue = ctx.factsheet.get("sustainability", {}).get("wue", {}).get("value", 0.0)

        return core.emissions(emissions, duration, pue, wue)

    def build_properties(self, raw):
        return {
            "Metric Description": "Total CO2 emissions during training  (Energy consumed * Carbon Intensity).",
            "Emissions (kgCO2)": f"{raw['value']:.6f}",
            "Training Duration (s)": f"{raw['duration']:.2f}",
            "PUE": raw["pue"],
            "WUE": raw["wue"],
        }


# ==========================================================
# Carbon Intensity
# ==========================================================

class CarbonIntensityMetric(BaseMetric):

    def __init__(self):
        super().__init__("carbon_intensity", "score_carbon_intensity")

    def compute(self, ctx):
        # run_data = ctx.extras["run_data"]
        energy_consumed = ctx.factsheet.get("sustainability", {}).get("energy_consumed", {}).get("value", 0.0)
        emissions = ctx.factsheet.get("sustainability", {}).get("emissions", {}).get("value", 0.0)
        country = ctx.factsheet.get("sustainability", {}).get("country", {}).get("value", "Unknown")

        return core.carbon_intensity(energy_consumed, emissions, country)

    def build_properties(self, raw):
        return {
            "Metric Description": "Carbon intensity (kgCO2 per kWh). Based on the type of energy used and location.",
            "Carbon Intensity": f"{raw['value']:.6f}",
            "Country": raw["country"],
        }