# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.ndimage import label

from .ClusterData import ClusterData


# ================================================================
# 1. Section: Functions
# ================================================================
def get_grouping(volume: np.ndarray) -> ClusterData:
    labeled_array, num_features = label(volume)
    labeled_array, features_sizes = reorder_labels_array(labeled_array)
    cluster_data = ClusterData(labeled_array, features_sizes, num_features)

    return cluster_data


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Make labels by size
# ──────────────────────────────────────────────────────
def reorder_labels_array(labeled_array: np.ndarray) -> tuple:
    # Extract sizes without background count
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0

    # Get the sorted positions
    sorted_old_labels = np.argsort(sizes)[::-1]

    # Remap the labels by size starting in voxel 1
    new_label_map = np.zeros_like(sizes, dtype=int)
    for new_lbl, old_lbl in enumerate(sorted_old_labels[:-1], start=1):
        new_label_map[old_lbl] = new_lbl

    # Assign to variables
    sorted_labels = new_label_map[labeled_array]
    sizes = np.bincount(sorted_labels.ravel())[1:]

    return sorted_labels, sizes
