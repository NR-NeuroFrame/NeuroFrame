# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ....logger import logger
from .dataclasses import LateralizedSegment, MethodOutput
from .grouping import (
    get_grouping,
    assign_side
)



# ================================================================
# 1. Section: Functions
# ================================================================
def naive_grouping_separation(volume: np.ndarray) -> MethodOutput:
    logger.debug("Naive Grouping applied")

    # 1. Get the biggest connected groups
    cluster_data = get_grouping(volume)
    cluster_sizes = cluster_data.sizes

    # 2. Assess if the clusteres are real clusters or just noise
    has_two_direct_features = (cluster_data.num_features == 2 and cluster_sizes[1]/cluster_sizes[0]*100 > 50)

    # 3. Assign them with respect to left and right
    if has_two_direct_features:
        left, right = assign_side(cluster_data.labeled_array)
    else:
        left = volume.copy()
        right = volume.copy()

    # 4. Builds the methods output
    output = MethodOutput(
        lateralized_segment=LateralizedSegment(left, right, "Naive Separation"),
        condition=has_two_direct_features,
        cluster_data=cluster_data
    )
    return output
