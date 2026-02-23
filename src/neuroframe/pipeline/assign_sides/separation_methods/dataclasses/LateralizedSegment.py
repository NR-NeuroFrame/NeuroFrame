# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class LateralizedSegment:
    left: np.ndarray
    right: np.ndarray
    separation_method: str
