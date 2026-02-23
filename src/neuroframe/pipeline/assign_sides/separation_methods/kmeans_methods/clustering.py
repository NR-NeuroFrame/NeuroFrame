# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..grouping import assign_side
from .kmeans import perform_kmeans
from .kmeans_utils import(
    generate_initial_centers,
    build_hemispheres_from_clustering
)



# ================================================================
# 1. Section: KMeans Separation
# ================================================================
def clustering_separation(volume: np.ndarray, nr_centers: int = 30) -> np.ndarray:
    # 1. Generates a set of initial starting points based on the lateralized means
    random_centers = generate_initial_centers(volume, nr_centers=nr_centers)

    # 2. Loop until the centers obtained follow the lateralized condition
    labeled_array, is_center_found = try_kmeans_on_centers(volume, random_centers)

    # 3. Assigns sides, even if did not work
    left, right = assign_side(labeled_array)

    return (left, right), is_center_found


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Try multiple centers
# ──────────────────────────────────────────────────────
def try_kmeans_on_centers(volume: np.ndarray, random_centers: np.ndarray) -> tuple:
    # 1. Loop until the centers obtained follow the lateralized condition
    for centers in random_centers:
        cluster_centers, cluster_labels, is_centers_found = perform_kmeans(volume, centers)

        if(is_centers_found):
            labeled_array = build_hemispheres_from_clustering(volume, cluster_labels)
            return labeled_array, is_centers_found
    return labeled_array, is_centers_found
