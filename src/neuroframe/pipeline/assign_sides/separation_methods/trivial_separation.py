# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ...utils import separate_volume



# ================================================================
# 1. Section: Functions
# ================================================================
def trivial_separation(volume: np.ndarray) -> tuple[tuple, bool]:
    left, right = separate_volume(volume)
    midline_x = volume.shape[2] // 2

    # 2. Check if any of the hemispheres is empty, by counting the non-zero elements
    hemisphere_not_empty = (np.count_nonzero(left) != 0) and (np.count_nonzero(right) != 0)

    # 3. Check if the midline cuts any segment
    is_cut = (np.count_nonzero(volume[:, :, midline_x]) != 0)

    # 4. If the separation is clean, just use midline for trivial separation
    is_trivial = hemisphere_not_empty and not is_cut
    return (left, right), is_trivial
