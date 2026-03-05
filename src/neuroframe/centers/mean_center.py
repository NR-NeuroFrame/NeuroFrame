# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd
import numpy as np

from .Center import Center



# ================================================================
# 1. Section: Functions
# ================================================================
def get_mean_centers(seg_lab: int, seg_left: np.ndarray, seg_right: np.ndarray) -> pd.DataFrame:
    # 1. Compute the mean center
    left_center = compute_safe_mean(seg_left)
    right_center = compute_safe_mean(seg_right)

    # 2. Deals with poor separations
    if left_center is None and right_center is not None:
        left_center = right_center.copy()
    elif right_center is None and left_center is not None:
        right_center = left_center.copy()

    # 3. Sore it in a dataclass
    seg_centers = Center(
        id=seg_lab,
        left_center=left_center,
        right_center=right_center
    )

    return seg_centers


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def compute_safe_mean(seg: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(seg)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)
