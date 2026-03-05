# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from ..logger import logger
from ..mouse import Mouse

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 1. Init the data_df
    left_df, right_df, average_df = init_data_df()

    # 2. Adds all the data to the data_df
    for idx, center in enumerate(centers):
        volume = volumes[idx]

        if(volume.id != center.id):
            logger.warning(
                "The Volumes array and Centers array does not match."
                f"When calling center {center.id} got volume {volume.center}"
            )

        left_df.loc[len(left_df), LEFT_COLS_NAMES] = [
            center.id,
            center.left_center[2],
            center.left_center[1],
            center.left_center[0],
            volume.left_volume_mm(mouse.voxel_size)
        ]

        right_df.loc[len(right_df), RIGHT_COLS_NAMES] = [
            center.id,
            center.right_center[2],
            center.right_center[1],
            center.right_center[0],
            volume.right_volume_mm(mouse.voxel_size)
        ]

        average_df.loc[len(average_df), AVERAGE_COLS_NAMES] = [
            center.id,
            center.average_center[2],
            center.average_center[1],
            center.average_center[0],
            volume.average_volume_mm(mouse.voxel_size)
        ]

    # 3. Merges with info_df
    left_df = info_df.merge(left_df, on='id', how='left')
    right_df = info_df.merge(right_df, on='id', how='left')
    average_df = info_df.merge(average_df, on='id', how='left')

    return left_df, right_df, average_df


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def init_data_df() -> tuple:
    left_df = pd.DataFrame(columns=LEFT_COLS_NAMES)
    right_df = pd.DataFrame(columns=RIGHT_COLS_NAMES)
    average_df = pd.DataFrame(columns=AVERAGE_COLS_NAMES)

    return left_df, right_df, average_df
