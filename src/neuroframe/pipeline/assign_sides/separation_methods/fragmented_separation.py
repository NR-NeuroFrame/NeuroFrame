# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .dataclasses import LateralizedSegment, MethodOutput
from .grouping import (
    get_relevant_clusters_otsu,
    assign_side,
    ClusterData
)



# ================================================================
# 1. Section: Functions
# ================================================================
def fragmented_grouping_separation(cluster_data: ClusterData) -> MethodOutput:
    # 1. Extract the data
    labeled_array = cluster_data.labeled_array
    cluster_sizes = cluster_data.sizes
    left, right = assign_side(labeled_array)

    # 2. Checks if we should do fragmentation
    relevant_clusters = get_relevant_clusters_otsu(cluster_sizes)
    has_enough_relevant_clusters = len(relevant_clusters) >= 2

    # 3. Merge Fragments if should (after the two biggest)
    if(has_enough_relevant_clusters): left, right = merge_fragments(labeled_array)

    # 4. Makes sure it is still a mask
    left = np.where(left > 0, 1, 0)
    right = np.where(right > 0, 1, 0)

    # 5. Builds the methods update
    output = MethodOutput(
        lateralized_segment=LateralizedSegment(left, right, "Naive Separation (Fragmented)"),
        condition=has_enough_relevant_clusters
    )
    return output


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def merge_fragments(labeled_array: np.ndarray) -> tuple:
    (left_volume, right_volume), (left_center, right_center) = assign_side(labeled_array, True)

    for i in range(3, np.max(labeled_array) + 1):
        piece = np.where(labeled_array == i, 1, 0)
        piece_center = np.mean(np.argwhere(piece), axis=0)

        # 3.1 Checks to which center is closer to
        is_closer_to_left = np.linalg.norm(piece_center - left_center) < np.linalg.norm(piece_center - right_center)

        # 3.2 Update the volumes accordingly
        if is_closer_to_left:
            left_volume, left_center = update_side(left_volume, piece, left_center)
        else:
            right_volume, right_center = update_side(right_volume, piece, right_center)

    return left_volume, right_volume

def update_side(
    main_volume: np.ndarray,
    new_piece: np.ndarray,
    current_center: np.ndarray
) -> tuple:
    # 1. Adds it to the volume and recalculates the center
    main_volume += new_piece
    current_center = np.mean(np.argwhere(main_volume), axis=0)

    return main_volume, current_center
