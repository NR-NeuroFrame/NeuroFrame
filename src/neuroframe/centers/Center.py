# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from ..logger import logger



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Center:
    id: int
    left_center: np.ndarray
    right_center: np.ndarray
    converted: bool = False

    @property
    def average_center(self) -> np.ndarray[float, float, float]:
        mirror_right_center = self.right_center.copy()
        mirror_right_center[2] *= -1 # flips over x

        return np.mean([mirror_right_center, self.left_center], axis=0)

    @property
    def std_center(self) -> np.ndarray[float, float, float]:
        mirror_right_center = self.right_center.copy()
        mirror_right_center[2] *= -1 # flips over x

        return np.std([mirror_right_center, self.left_center], axis=0)

    def convert_center_to_bl(self, bl_space: np.ndarray):
        if not self.converted:
            idx_left = tuple(np.round(self.left_center).astype(int))
            idx_right = tuple(np.round(self.right_center).astype(int))
            self.left_center = bl_space[idx_left]
            self.right_center = bl_space[idx_right]
            self.converted = True
        else:
            logger.warning(f"Already converted the segment {self.id} to BL space")
