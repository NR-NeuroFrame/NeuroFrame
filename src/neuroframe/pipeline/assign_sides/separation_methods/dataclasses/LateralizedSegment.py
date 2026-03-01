# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass, field
from typing import Optional



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class LateralizedSegment:
    left: np.ndarray
    right: np.ndarray
    separation_method: str
    id: Optional[int] = field(default=None, init=False)  # set later

    @property
    def volume(self) -> np.ndarray:
        volume = self.left + self.right
        return np.where(volume > 0, 1, 0)

    @property
    def left_ratio(self):
        l_nr_voxels = np.sum(np.where(self.left > 0))
        nr_voxels = np.sum(np.where(self.volume > 0))
        return l_nr_voxels / nr_voxels

    @property
    def right_ratio(self):
        r_nr_voxels = np.sum(np.where(self.right > 0))
        nr_voxels = np.sum(np.where(self.volume > 0))
        return r_nr_voxels / nr_voxels
