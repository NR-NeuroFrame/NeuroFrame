# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd
import numpy as np

from tqdm import tqdm

from ...mouse import Mouse
from .dataclasses import DataDF
from .datas_df import build_center_df
from .save_csv import save_mouse_results, save_mouse_pca
from .volumes import get_segment_volumes
from .pca import get_segment_pca, buid_pca_df
from .centers import (
    get_inner_centers,
    get_mean_centers,
    get_shape_centers
)



# ================================================================
# 1. Section: Functions
# ================================================================
def get_segments_data(
    mouse: Mouse,
    info_df: pd.DataFrame,
    mode: str,
    template_mouse: Mouse | None = None
) -> DataDF:
    # 1. Extract the data
    segmentations = mouse.segmentation.data
    segments_labels = mouse.segmentation.labels
    segments_lateralized = mouse.hemisphere.data
    segments_bl = mouse.field_bl.data
    segments_nedt = mouse.segmentation_nedt.data

    # 2. Loop over every segment
    centers = []
    volumes = []
    pcas = []
    shape_pca = []
    for seg_lab in tqdm(segments_labels, desc="Calculating centers", unit="seg"):
        # 2.1 Get the segment data
        seg_lat = np.where(segmentations == seg_lab, segments_lateralized, 0)
        seg_left = np.where(seg_lat == 1, 1, 0)
        seg_right = np.where(seg_lat == 2, 1, 0)

        # 2.2 Get the correct centers
        if(mode.lower() == "mean"):
            seg_centers = get_mean_centers(seg_lab, seg_left, seg_right)
        elif(mode.lower() == "inner"):
            seg_left_nedt = np.where(seg_lat == 1, segments_nedt, 0)
            seg_right_nedt = np.where(seg_lat == 2, segments_nedt, 0)
            seg_centers = get_inner_centers(seg_lab, seg_left_nedt, seg_right_nedt)
        elif(mode.lower() == "wt_shape"):
            seg_left_nedt = np.where(seg_lat == 1, segments_nedt, 0)
            seg_right_nedt = np.where(seg_lat == 2, segments_nedt, 0)
            seg_centers, wt_seg_pca = get_shape_centers(seg_lab, seg_left_nedt, seg_right_nedt, template_mouse)
            shape_pca.append(wt_seg_pca)

        # 2.3 Convert to bl-mm coordinates
        seg_centers.convert_center_to_bl(segments_bl)

        # 2.4 Get the volumes
        seg_volumes = get_segment_volumes(seg_lab, seg_left, seg_right)

        # 2.5 Get the pca for each segment
        seg_pca = get_segment_pca(seg_lab, seg_left, seg_right)

        # 2.5. Store everything back into a list
        centers.append(seg_centers)
        volumes.append(seg_volumes)
        pcas.append(seg_pca)

    centers = np.array(centers)
    volumes = np.array(volumes)
    pcas = np.array(pcas)
    shape_pca = np.array(shape_pca)

    # 3. Builds the DF
    data_dfs = build_center_df(mouse, centers, volumes, info_df)

    # 4. Saves the files in the mouse folder
    save_path = save_mouse_results(mouse, data_dfs, mode)
    print(f"The mouse results where saved at {save_path}")

    # 5. Save the pcas
    pca_df = buid_pca_df(mouse, pcas, info_df)
    pca_path = save_mouse_pca(mouse, pca_df, mode="pca")
    print(f"The segments pca where saved at {pca_path}")

    # 6. Save the shape-pcas if available
    if(len(shape_pca) > 0):
        shape_pca_df = buid_pca_df(mouse, shape_pca, info_df)
        shape_pca_path = save_mouse_pca(mouse, shape_pca_df, mode="shape_pca")
        print(f"The segments wt shape pca where saved at {shape_pca_path}")

    return data_dfs
