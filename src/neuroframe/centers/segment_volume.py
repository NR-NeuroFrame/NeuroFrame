# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .Volume import Volume


# ================================================================
# 1. Section: Functions
# ================================================================
def get_segment_volumes(seg_lab: int, seg_left: np.ndarray, seg_right: np.ndarray) -> Volume:
    segment_volume = Volume(
        id=seg_lab,
        left_volume_vx=np.sum(seg_left),
        right_volume_vx=np.sum(seg_right)
    )

    return segment_volume
