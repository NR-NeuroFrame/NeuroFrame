# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from skimage.filters import threshold_otsu



# ================================================================
# 1. Section: Relevant Clustering
# ================================================================
def get_relevant_clusters_otsu(features: np.ndarray) -> np.ndarray:
    # 1. Gets the threshold and returns the clusters above it
    thr = threshold_otsu(features)
    relevant_features = np.where(features > thr)[0]

    return relevant_features



# ================================================================
# 2. Section: Left and Right Assignment
# ================================================================
def assign_side(labeled_array: np.ndarray, get_center: bool = False) -> tuple:
    first_side = np.where(labeled_array == 1, 1, 0)
    second_side = np.where(labeled_array == 2, 1, 0)

    first_center = np.mean(np.argwhere(first_side), axis=0)
    second_center = np.mean(np.argwhere(second_side), axis=0)

    if first_center[2] > second_center[2]:
        left = first_side
        left_center = first_center
        right = second_side
        right_center = second_center
    else:
        left = second_side
        left_center = second_center
        right = first_side
        right_center = first_center

    if(get_center): return (left, right), (left_center, right_center)
    return left, right
