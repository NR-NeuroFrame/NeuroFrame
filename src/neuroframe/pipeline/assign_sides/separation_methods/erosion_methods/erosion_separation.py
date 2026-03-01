# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from skimage.morphology import opening

from ..dataclasses import LateralizedSegment, MethodOutput
from .initiate_parameters import get_selem_shape, get_starting_opening_size
from .similarity import get_volume_similarity
from ..grouping import (
    get_relevant_clusters_otsu,
    get_grouping,
    assign_side,
    ClusterData
)

SIMILARITY_THRESHOLD: int = 90



# ================================================================
# 1. Section: Functions
# ================================================================
def destroying_bridges_separation(volume: np.ndarray, method: str) -> MethodOutput:
    # 1. Tryies multiple erosion (progressive intensity) to make sure we remove bridge
    has_found_separation, cluster_data = loop_opening(volume, method)

    if has_found_separation:
        left, right = assign_side(cluster_data.labeled_array)
    else:
        left = volume.copy()
        right = volume.copy()

    # 2. Builds the method output
    output = MethodOutput(
        lateralized_segment=LateralizedSegment(left, right, f"Opening Separation ({method})"),
        condition=has_found_separation
    )
    return output



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def loop_opening(volume: np.ndarray, method: str) -> tuple:
    # 1. Initialize variables
    eroded_volume = volume.copy()
    similarity_value = 100
    found_separation = False
    opening_size = get_starting_opening_size(method)

    # 2. Loop until either less than 90% of similar volume is kept or the hemispheres are separable
    while similarity_value >= SIMILARITY_THRESHOLD and opening_size <= 20 and not found_separation:
        # 2.1 Perform the erosion
        eroded_volume, cluster_data = perform_morphological_opening(volume, opening_size, method)

        # 2.2. Get metrics needed to understand to continue
        similarity_value = get_volume_similarity(volume, eroded_volume)
        relevant_features = get_relevant_clusters_otsu(cluster_data.sizes)
        found_separation = len(relevant_features) > 1

        # 2.3 Prepare next iteration
        opening_size += 1

    return found_separation, cluster_data


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def perform_morphological_opening(volume: np.ndarray, opening_size: int, method: str) -> np.ndarray:
    # 1. Define the structuring element based on the method
    selem = get_selem_shape(method, opening_size)

    # 2. Perform morphological opening
    eroded_volume = opening(volume, selem)
    if(np.count_nonzero(eroded_volume) == 0):
        return eroded_volume, ClusterData.empty(volume.shape)

    # 3. Perform grouping (label)
    eroded_clusters_data = get_grouping(eroded_volume)

    return eroded_volume, eroded_clusters_data
