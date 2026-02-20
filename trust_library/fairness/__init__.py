"""
Fairness Module
===================
Modular fairness analysis package.

Structure
---------
fairness_metrics_core.py  — pure mathematical implementations (no fairness library deps)
FairnessPillar
"""

from .fairness import FairnessPillar
from . import fairness_metrics_core

__all__ = [
    "FairnessPillar",
    "fairness_metrics_core",
]