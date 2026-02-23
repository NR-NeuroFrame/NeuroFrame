# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .grouping import (
    get_grouping,
    assign_side,
    ClusterData,
)



# ================================================================
# 1. Section: Functions
# ================================================================
def naive_grouping_separation(volume: np.ndarray) -> tuple[tuple, bool, ClusterData]:
    # 1. Get the biggest connected groups
    cluster_data = get_grouping(volume)
    cluster_sizes = cluster_data.sizes

    # 2. Assign them with respect to left and right
    left, right = assign_side(cluster_data.labeled_array)

    # 2. Assess if the clusteres are real clusters or just noise
    has_two_direct_features = (cluster_data.num_features == 2 and cluster_sizes[1]/cluster_sizes[0]*100 > 50)
    return (left, right), has_two_direct_features, cluster_data
