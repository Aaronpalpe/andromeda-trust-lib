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
# from .sustainability_metrics_core import (
#     track_training_run,
# )

# ─────────────────────────────────────────────────────────────
# Sustainability Metrics
# ─────────────────────────────────────────────────────────────
from .sustainability_metrics_core import (
    energy_consumption,
    emissions,
    carbon_intensity,
)

__all__ = [
    # Pillar
    "SustainabilityPillar",

    # # Tracking
    # "track_training_run",

    # Metrics
    "energy_consumption",
    "emissions",
    "carbon_intensity",
]