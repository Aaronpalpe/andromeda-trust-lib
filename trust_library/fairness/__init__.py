"""
fairness_refactored
===================
Modular fairness analysis package.

Structure
---------
metrics_core.py  — pure mathematical implementations (no fairness library deps)
adapters.py      — optional thin wrappers for AIF360 and HolisticAI
config.py        — factsheet parsing and EvaluationContext dataclass
analyse.py       — orchestration layer: scoring, result aggregation
"""

from .fairness import analyse
from . import metrics_core
from . import adapters

__all__ = [
    "analyse",
    "metrics_core",
    "adapters",
]