"""
adapters.py
===================

External fairness metric implementations using:

    - AIF360 (when available)
    - HolisticAI (optional)

All functions operate on raw numpy arrays.
"""

import numpy as np
import pandas as pd

# TODO