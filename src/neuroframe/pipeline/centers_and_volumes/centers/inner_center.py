# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from ..dataclasses import Center



# ================================================================
# 1. Section: Functions
# ================================================================
def get_inner_centers(seg_lab: int, seg_left: np.ndarray, seg_right: np.ndarray) -> Center:
    # 1. Compute the inner center
    left_center = compute_safe_inner(seg_left)
    right_center = compute_safe_inner(seg_right)

    # 2. Deals with poor separations
    if left_center is None and right_center is not None:
        left_center = right_center.copy()
    elif right_center is None and left_center is not None:
        right_center = left_center.copy()

    # 3. Sore it in a dataclass
    seg_centers = Center(
        id=seg_lab,
        left_center=np.round(left_center, 0),
        right_center=np.round(right_center, 0)
    )

    return seg_centers

# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def compute_safe_inner(volume: np.ndarray) -> np.ndarray:
    idx = np.unravel_index(np.argmax(volume), volume.shape)

    return np.array(idx)
