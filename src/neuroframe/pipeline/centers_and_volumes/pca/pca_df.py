# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from ....mouse import Mouse
from ..dataclasses import DataDF

COLS_NAMES = ["id", "1_x", "1_y", "1_z", "2_x", "2_y", "2_z", "3_x", "3_y", "3_z"]



# ================================================================
# 1. Section: Functions
# ================================================================
def buid_pca_df(mouse: Mouse, pcas: np.ndarray, info_df: pd.DataFrame) -> DataDF:
    data_dfs = init_pca_df()

    for idx, pca in enumerate(pcas):
        data_dfs.left_df.loc[len(data_dfs.left_df), COLS_NAMES] = [
            pca.id,
            np.round(pca.left_pca[0, 2], 2),
            np.round(pca.left_pca[0, 1], 2),
            np.round(pca.left_pca[0, 0], 2),
            np.round(pca.left_pca[1, 2], 2),
            np.round(pca.left_pca[1, 1], 2),
            np.round(pca.left_pca[1, 0], 2),
            np.round(pca.left_pca[2, 2], 2),
            np.round(pca.left_pca[2, 1], 2),
            np.round(pca.left_pca[2, 0], 2),
        ]

        data_dfs.right_df.loc[len(data_dfs.right_df), COLS_NAMES] = [
            pca.id,
            np.round(pca.right_pca[0, 2], 2),
            np.round(pca.right_pca[0, 1], 2),
            np.round(pca.right_pca[0, 0], 2),
            np.round(pca.right_pca[1, 2], 2),
            np.round(pca.right_pca[1, 1], 2),
            np.round(pca.right_pca[1, 0], 2),
            np.round(pca.right_pca[2, 2], 2),
            np.round(pca.right_pca[2, 1], 2),
            np.round(pca.right_pca[2, 0], 2),
        ]

    data_dfs.left_df = info_df.merge(data_dfs.left_df, on='id', how='left')
    data_dfs.right_df = info_df.merge(data_dfs.right_df, on='id', how='left')

    return data_dfs



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def init_pca_df() -> DataDF:
    left_df = pd.DataFrame(columns=COLS_NAMES)
    right_df = pd.DataFrame(columns=COLS_NAMES)

    return DataDF(left_df, right_df, None)
