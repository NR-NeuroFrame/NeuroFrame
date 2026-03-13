# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from dataclasses import dataclass

COLS_NAMES = ["id", "1_x", "1_y", "1_z", "2_x", "2_y", "2_z", "3_x", "3_y", "3_z"]



# ================================================================
# 1. Section: Storage DataClass
# ================================================================
@dataclass
class PCADF:
    left_df: pd.DataFrame
    right_df: pd.DataFrame



# ================================================================
# 1. Section: Building the DFs
# ================================================================
def buid_pca_df(pcas: np.ndarray, info_df: pd.DataFrame) -> PCADF:
    pca_dfs = init_pca_df()

    for idx, pca in enumerate(pcas):
        pca_dfs.left_df.loc[len(pca_dfs.left_df), COLS_NAMES] = [
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

        pca_dfs.right_df.loc[len(pca_dfs.right_df), COLS_NAMES] = [
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

    pca_dfs.left_df = info_df.merge(pca_dfs.left_df, on='id', how='left')
    pca_dfs.right_df = info_df.merge(pca_dfs.right_df, on='id', how='left')

    return pca_dfs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def init_pca_df() -> PCADF:
    left_df = pd.DataFrame(columns=COLS_NAMES)
    right_df = pd.DataFrame(columns=COLS_NAMES)

    return PCADF(left_df, right_df)
