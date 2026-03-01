# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from sklearn.cluster import KMeans
from ..grouping import check_lateralization_condition



# ================================================================
# 1. Section: Functions
# ================================================================
def perform_kmeans(volume: np.ndarray, centers: np.ndarray | list) -> tuple:
    # 1. Prepares the volume for kmeans
    kmeans = KMeans(n_clusters=2, init=centers, n_init=1, tol=1e-2)
    adapted_vol = np.argwhere(volume)

    # 2. Performs kmeans clustering on the adapted volume
    kmeans.fit(adapted_vol)

    # 3. Extract the data from the kmeans clustering
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # 4. Check lateralization condition
    is_correct_cluster_centers = check_lateralization_condition(cluster_centers)

    return cluster_centers, cluster_labels, is_correct_cluster_centers


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Check if was successeful
# ──────────────────────────────────────────────────────
