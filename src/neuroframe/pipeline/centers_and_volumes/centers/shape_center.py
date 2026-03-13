# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ....registrator import Registrator
from ....mouse import Mouse
from ...segment_pca import (
    PCASummary,
    get_segment_pca
)
from ..dataclasses import Center
from .inner_center import get_inner_centers

SEGMENT_REGISTRATOR = Registrator(
    method="rigid",
    multiple_resolutions=True
)



# ================================================================
# 1. Section: Functions
# ================================================================
def get_shape_centers(
    seg_lab: int,
    seg_left_nedt: np.ndarray,
    seg_right_nedt: np.ndarray,
    template_mouse: Mouse
) -> tuple:
    # 0. If segment not present in WT just skip this (make all 0,0,0 ?)
    if seg_lab not in template_mouse.segmentation.labels:
        return (
            Center.empty(seg_lab),
            PCASummary.empty(seg_lab)
        )

    # 1. Get the WT Laterals
    wt_seg = np.where(template_mouse.segmentation.data == seg_lab, template_mouse.hemisphere.data, 0)
    wt_left = np.where(wt_seg == 1, 1, 0)
    wt_right = np.where(wt_seg == 2, 1, 0)

    seg_left = np.where(seg_left_nedt > 0, 1, 0)
    seg_right = np.where(seg_right_nedt > 0, 1, 0)

    # 2. Do the rigid registration to the mci shape
    wt_trs_left, left_transform = SEGMENT_REGISTRATOR.register(
        seg_left, wt_left
    )
    wt_trs_right, right_transform = SEGMENT_REGISTRATOR.register(
        seg_right, wt_right
    )

    # 3. Get the WT NEDT
    wt_left_nedt = np.where(wt_seg == 1, template_mouse.segmentation_nedt.data, 0)
    wt_right_nedt = np.where(wt_seg == 2, template_mouse.segmentation_nedt.data, 0)

    # 4. Apply it to the nedt
    wt_left_trs_nedt = SEGMENT_REGISTRATOR.apply_transform(wt_left_nedt, left_transform)
    wt_right_trs_nedt = SEGMENT_REGISTRATOR.apply_transform(wt_right_nedt, right_transform)

    # 5. Get the centers
    seg_center = get_inner_centers(seg_lab, wt_left_trs_nedt, wt_right_trs_nedt)

    # 6. Get the PCA of the transformed
    pca_data = get_segment_pca(seg_lab, wt_trs_left, wt_trs_right)

    return seg_center, pca_data
