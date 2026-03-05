# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Volume:
    id: int
    left_volume_vx: int
    right_volume_vx: int

    def left_volume_mm(self, voxel_size: float) -> float:
        sx, sy, sz = voxel_size
        return self.left_volume_vx * sx * sy * sz

    def right_volume_mm(self, voxel_size: float) -> float:
        sx, sy, sz = voxel_size
        return self.right_volume_vx * sx * sy * sz

    def average_volume_mm(self, voxel_size: float) -> float:
        left_volume = self.left_volume_mm(voxel_size)
        right_volume = self.right_volume_mm(voxel_size)

        return np.mean([left_volume, right_volume])

    def std_volume_mm(self, voxel_size: float) -> float:
        left_volume = self.left_volume_mm(voxel_size)
        right_volume = self.right_volume_mm(voxel_size)

        return np.std([left_volume, right_volume])
