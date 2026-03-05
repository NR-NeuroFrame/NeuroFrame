# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd
import numpy as np

from tqdm import tqdm

from ..mouse import Mouse
from .DataDF import DataDF
from .mean_center import get_mean_centers
from .inner_center import get_inner_centers
from .segment_volume import get_segment_volumes
from .datas_df import build_center_df
from .save_csv import save_mouse_results



# ================================================================
# 1. Section: Functions
# ================================================================
def get_segments_centers(mouse: Mouse, info_df: pd.DataFrame, mode: str) -> DataDF:
    # 1. Extract the data
    segmentations = mouse.segmentation.data
    segments_labels = mouse.segmentation.labels
    segments_lateralized = mouse.hemisphere.data
    segments_bl = mouse.field_bl.data
    segments_nedt = mouse.segmentation_nedt.data

    # 2. Loop over every segment
    centers = []
    volumes = []
    for seg_lab in tqdm(segments_labels, desc="Calculating centers", unit="seg"):
        # 2.1 Get the segment data
        seg_lat = np.where(segmentations == seg_lab, segments_lateralized, 0)
        seg_left = np.where(seg_lat == 1, 1, 0)
        seg_right = np.where(seg_lat == 2, 1, 0)

        # 2.2 Get the correct centers
        if(mode.lower() == "mean"):
            seg_centers = get_mean_centers(seg_lab, seg_left, seg_right)
        elif(mode.lower() == "inner"):
            seg_left = np.where(seg_lat == 1, segments_nedt, 0)
            seg_right = np.where(seg_lat == 2, segments_nedt, 0)
            seg_centers = get_inner_centers(seg_lab, seg_left, seg_right)

        # 2.3 Convert to bl-mm coordinates
        seg_centers.convert_center_to_bl(segments_bl)

        # 2.4 Get the volumes
        seg_volumes = get_segment_volumes(seg_lab, seg_left, seg_right)

        # 2.5. Store everything back into a list
        centers.append(seg_centers)
        volumes.append(seg_volumes)

    centers = np.array(centers)
    volumes = np.array(volumes)

    # 3. Builds the DF
    data_dfs = build_center_df(mouse, centers, volumes, info_df)

    # 4. Saves the files in the mouse folder
    save_path = save_mouse_results(mouse, data_dfs, mode)
    print(f"The mouse results where saved at {save_path}")

    return data_dfs
