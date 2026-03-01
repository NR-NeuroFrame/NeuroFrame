# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .dataclasses import LateralizedSegment



# ================================================================
# 1. Section: Functions
# ================================================================
def failed_separation(volume: np.ndarray) -> LateralizedSegment:
    # 1. Get the center of the full thing
    center = np.mean(np.argwhere(volume), axis=0)
    midline_x = volume.shape[2] // 2

    # 2. Assigns the volume to the closest and the other as None
    if(center[2] > midline_x): return LateralizedSegment(np.zeros_like(volume), volume, "Failed")
    else: return LateralizedSegment(volume, np.zeros_like(volume), "Failed")
