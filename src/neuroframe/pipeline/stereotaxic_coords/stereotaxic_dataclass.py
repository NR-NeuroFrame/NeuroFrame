# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd
import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass



# ================================================================
# 1. Section: Config Dataclass
# ================================================================
@dataclass
class StereotaxicConfig:
    reference_df: pd.DataFrame
    skull_points: tuple
    group_folder: str | None = None
    is_parallelized: bool = True
    file_name: str = "stereotaxic_coordinates"
    mode: str = "full_mean"



# ================================================================
# 2. Section: Centroid Data Dataclass
# ================================================================
Vec3 = tuple[float, float, float] | NDArray[np.floating]

@dataclass(frozen=True)
class LRPair:
    left: Vec3 | float
    right: Vec3 | float

    def as_array(self) -> NDArray[np.float64]:
        return np.stack([np.asarray(self.left, dtype=float),
                         np.asarray(self.right, dtype=float)], axis=0)

    def stats(self, *, ddof: int = 1) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        x = self.as_array()
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=ddof)
        ste = std / np.sqrt(x.shape[0])
        return mean, std, ste

@dataclass(frozen=True)
class SegmentCentroidData:
    separation_method: str
    voxel_size: float

    centroid_voxel: LRPair
    volume_voxel: LRPair

    @property
    def centroid_um(self) -> LRPair:
        return LRPair(
            self.centroid_voxel.left * self.voxel_size,
            self.centroid_voxel.right * self.voxel_size,
        )

    @property
    def volume_um(self) -> LRPair:
        return LRPair(
            self.volume_voxel.left * (self.voxel_size ** 3),
            self.volume_voxel.right * (self.voxel_size ** 3),
        )

    @property
    def data_dict(self) -> dict:
        data = {
            "Data μm": self._generate_um_dict(),
            "Data voxel": self._generate_voxel_dict()
        }
        return data

    def _generate_um_dict(self) -> dict:
        return {
            'Separation Method': self.separation_method,
            'xyz (um) - L': self.centroid_um[0],
            'xyz (um) - R': self.centroid_um[1],
            'mean xyz (um)': self.centroid_um.stats()[0],
            'std xyz (um)': self.centroid_um.stats()[1],
            'ste xyz (um)': self.centroid_um.stats()[2],
            'volume (um³) - L': self.volume_um[0],
            'volume (um³) - R': self.volume_um[1],
            'mean volume (um³)': self.volume_um.stats()[0],
            'std volume (um³)': self.volume_um.stats()[1],
            'ste volume (um³)': self.volume_um.stats()[2],
        }

    def _generate_voxel_dict(self) -> dict:
        return {
            'Separation Method': self.separation_method,
            'xyz (voxel) - L': self.centroid_voxel[0],
            'xyz (voxel) - R': self.centroid_voxel[1],
            'mean xyz (voxel)': self.centroid_voxel.stats()[0],
            'std xyz (voxel)': self.centroid_voxel.stats()[1],
            'ste xyz (voxel)': self.centroid_voxel.stats()[2],
            'volume (#voxel) - L': self.volume_voxel[0],
            'volume (#voxel) - R': self.volume_voxel[1],
            'mean volume (#voxel)': self.volume_voxel.stats()[0],
            'std volume (#voxel)': self.volume_voxel.stats()[1],
            'ste volume (#voxel)': self.volume_voxel.stats()[2],
        }
