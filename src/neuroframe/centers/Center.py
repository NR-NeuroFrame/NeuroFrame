# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Center:
    id: int
    left_center = np.ndarray | tuple
    right_center = np.ndarray | tuple

    @property
    def average_center(self) -> tuple[float, float, float]:
        mirror_right_center = self.right_center
        mirror_right_center[2] *= -1 # flips over x

        return np.mean([mirror_right_center, self.left_center], axis=0)
