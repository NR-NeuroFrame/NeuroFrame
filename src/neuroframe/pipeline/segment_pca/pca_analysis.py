# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm

from ...mouse import Mouse
from .computation import get_segment_pca
from .pca_df import buid_pca_df
from .save_pca import save_mouse_pca



# ================================================================
# 1. Section: Functions
# ================================================================
def get_segments_pca(
    mouse: Mouse,
    info_df: pd.DataFrame,
) -> Path:
    # 0. Extract the data
    segmentations = mouse.segmentation.data
    segments_labels = mouse.segmentation.labels
    segments_lateralized = mouse.hemisphere.data

    # 1. Loop over all the data
    pcas = []
    for seg_lab in tqdm(segments_labels, desc="Calculating PCA", unit="PCA"):
        # 1.1 Lateralize the segment
        seg_lat = np.where(segmentations == seg_lab, segments_lateralized, 0)
        seg_left = np.where(seg_lat == 1, 1, 0)
        seg_right = np.where(seg_lat == 2, 1, 0)

        # 1.2 Compute and store the PCA
        seg_pca = get_segment_pca(seg_lab, seg_left, seg_right)
        pcas.append(seg_pca)
    pcas = np.array(pcas)

    # 2. Store in a file
    pca_df = buid_pca_df(mouse, pcas, info_df)
    pca_path = save_mouse_pca(mouse, pca_df)

    return pca_path
