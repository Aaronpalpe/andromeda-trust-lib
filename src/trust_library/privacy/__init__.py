"""
Privacy Module
===================
Modular privacy analysis package.

Structure
---------
privacy_metrics_core.py  — pure mathematical implementations
"""

from .privacy import PrivacyPillar
from . import privacy_metrics_core

__all__ = [
    "PrivacyPillar",
    "privacy_metrics_core",
]