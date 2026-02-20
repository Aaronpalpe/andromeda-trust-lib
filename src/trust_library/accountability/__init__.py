"""
Accountability Module
===================
Modular accountability analysis package.

Structure
---------
accountability_metrics_core.py  — pure mathematical implementations (no fairness library deps)
AccountabilityPillar
"""

from .accountability import AccountabilityPillar
#from . import accountability_metrics_core

__all__ = [
    "AccountabilityPillar",
    #"accountability_metrics_core",
]