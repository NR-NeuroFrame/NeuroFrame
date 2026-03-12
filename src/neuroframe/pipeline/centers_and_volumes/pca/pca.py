# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .pca_computation import get_volume_pca_components

from ..dataclasses import PCASummary


# ================================================================
# 1. Section: Functions
# ================================================================
def get_segment_pca(seg_lab, seg_left, seg_right) -> PCASummary:
    left_pca = get_volume_pca_components(seg_left)
    right_pca = get_volume_pca_components(seg_right)

    return PCASummary(
        id=seg_lab,
        left_pca=left_pca,
        right_pca=right_pca,
    )
