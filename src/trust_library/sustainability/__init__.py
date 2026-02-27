"""
Sustainability Module
=====================
Modular sustainability analysis package.

Structure
---------
sustainability_metrics_core.py  - pure mathematical implementations
SustainabilityPillar
"""

from .sustainability import SustainabilityPillar

# ─────────────────────────────────────────────────────────────
# Tracking
# ─────────────────────────────────────────────────────────────
from .sustainability_metrics_core import (
    track_training_run,
)

# ─────────────────────────────────────────────────────────────
# Sustainability Metrics
# ─────────────────────────────────────────────────────────────
from .sustainability_metrics_core import (
    compute_energy_consumption,
    compute_emissions,
    compute_carbon_intensity,
    #compute_energy_efficiency,
)

__all__ = [
    # Pillar
    "SustainabilityPillar",

    # Tracking
    "track_training_run",

    # Metrics
    "compute_energy_consumption",
    "compute_emissions",
    "compute_carbon_intensity",
    #"compute_energy_efficiency",
]