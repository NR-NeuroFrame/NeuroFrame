# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ClusterData:
    labeled_array: np.ndarray
    sizes: np.ndarray
    num_features: int
