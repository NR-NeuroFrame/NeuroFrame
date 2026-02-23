# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .dataclasses import LateralizedSegment, MethodOutput
from .grouping import (
    get_grouping,
    assign_side
)



# ================================================================
# 1. Section: Functions
# ================================================================
def naive_grouping_separation(volume: np.ndarray) -> MethodOutput:
    # 1. Get the biggest connected groups
    cluster_data = get_grouping(volume)
    cluster_sizes = cluster_data.sizes

    # 2. Assign them with respect to left and right
    left, right = assign_side(cluster_data.labeled_array)

    # 3. Assess if the clusteres are real clusters or just noise
    has_two_direct_features = (cluster_data.num_features == 2 and cluster_sizes[1]/cluster_sizes[0]*100 > 50)

    # 4. Builds the methods output
    output = MethodOutput(
        lateralized_segment=LateralizedSegment(left, right, "Naive Separation"),
        condition=has_two_direct_features,
        cluster_data=cluster_data
    )
    return output
