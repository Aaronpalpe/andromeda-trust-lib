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

    def compute(self, ctx, run_data):
        return core.compute_energy_consumption(run_data)

    def compute_score(self, raw, config):
        thresholds = config.get(self.score_config_key, {}).get("thresholds", {}).get("value", [])
        return calculate_score(raw["value"], thresholds)

    def build_properties(self, raw):
        return {
            "Metric Description": "Total energy consumed during training.",
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

    def compute(self, ctx, run_data):
        return core.compute_emissions(run_data)

    def compute_score(self, raw, config):
        thresholds = config.get(self.score_config_key, {}).get("thresholds", {}).get("value", [])
        return calculate_score(raw["value"], thresholds)

    def build_properties(self, raw):
        return {
            "Metric Description": "Total CO2 emissions during training.",
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

    def compute(self, ctx, run_data):
        return core.compute_carbon_intensity(run_data)

    def compute_score(self, raw, config):
        thresholds = config.get(self.score_config_key, {}).get("thresholds", {}).get("value", [])
        return calculate_score(raw["value"], thresholds)

    def build_properties(self, raw):
        return {
            "Metric Description": "Carbon intensity (kgCO2 per kWh).",
            "Carbon Intensity": f"{raw['value']:.6f}",
            "Country": raw["country"],
        }


# ==========================================================
# Optional Energy Efficiency
# ==========================================================

class EnergyEfficiencyMetric(BaseMetric):

    def __init__(self):
        super().__init__("energy_efficiency", "score_energy_efficiency")

    def compute(self, ctx, run_data):
        return core.compute_energy_efficiency(run_data)

    def compute_score(self, raw, config):
        thresholds = config.get(self.score_config_key, {}).get("thresholds", {}).get("value", [])
        return calculate_score(raw["value"], thresholds)

    def build_properties(self, raw):
        return {
            "Metric Description": "Energy per training hour.",
            "Energy Efficiency (kWh/h)": f"{raw['value']:.6f}",
            "Training Time (h)": f"{raw['duration_hours']:.4f}",
        }