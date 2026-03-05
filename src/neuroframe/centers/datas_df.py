# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from ..logger import logger
from ..mouse import Mouse
from .DataDF import DataDF

LEFT_COLS_NAMES = ["id", "x", "y", "z", "volume"]
RIGHT_COLS_NAMES = ["id", "x", "y", "z", "volume"]
AVERAGE_COLS_NAMES = ["id", "x", "y", "z", "volume", "x_std", "y_std", "z_std", "volume_std"]



# ================================================================
# 1. Section: Functions
# ================================================================
def build_center_df(
    mouse: Mouse,
    centers: np.ndarray,
    volumes: np.ndarray,
    info_df: pd.DataFrame
) -> DataDF:

    # 1. Init the data_df
    data_dfs = init_data_df()

    # 2. Adds all the data to the data_df
    for idx, center in enumerate(centers):
        volume = volumes[idx]

        if(volume.id != center.id):
            logger.warning(
                "The Volumes array and Centers array does not match."
                f"When calling center {center.id} got volume {volume.center}"
            )

        data_dfs.left_df.loc[len(data_dfs.left_df), LEFT_COLS_NAMES] = [
            center.id,
            np.round(center.left_center[2], 2),
            np.round(center.left_center[1], 2),
            np.round(center.left_center[0], 2),
            np.round(volume.left_volume_mm(mouse.voxel_size), 2)
        ]

        data_dfs.right_df.loc[len(data_dfs.right_df), RIGHT_COLS_NAMES] = [
            center.id,
            np.round(center.right_center[2], 2),
            np.round(center.right_center[1], 2),
            np.round(center.right_center[0], 2),
            np.round(volume.right_volume_mm(mouse.voxel_size), 2)
        ]

        data_dfs.average_df.loc[len(data_dfs.average_df), AVERAGE_COLS_NAMES] = [
            center.id,
            np.round(center.average_center[2], 2),
            np.round(center.average_center[1], 2),
            np.round(center.average_center[0], 2),
            np.round(volume.average_volume_mm(mouse.voxel_size), 2),
            np.round(center.std_center[2], 2),
            np.round(center.std_center[1], 2),
            np.round(center.std_center[0], 2),
            np.round(volume.std_volume_mm(mouse.voxel_size), 2),
        ]

    # 3. Merges with info_df
    data_dfs.left_df = info_df.merge(data_dfs.left_df, on='id', how='left')
    data_dfs.right_df = info_df.merge(data_dfs.right_df, on='id', how='left')
    data_dfs.average_df = info_df.merge(data_dfs.average_df, on='id', how='left')

    return data_dfs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def init_data_df() -> DataDF:
    left_df = pd.DataFrame(columns=LEFT_COLS_NAMES)
    right_df = pd.DataFrame(columns=RIGHT_COLS_NAMES)
    average_df = pd.DataFrame(columns=AVERAGE_COLS_NAMES)

    return DataDF(left_df, right_df, average_df)
