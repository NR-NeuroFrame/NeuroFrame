# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Assign sides after KMEANS
# ================================================================
def build_hemispheres_from_clustering(volume: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    # Re assign values (-1 -> 0, 0->2)
    cluster_labels[cluster_labels == 0] = 2
    cluster_labels[cluster_labels == -1] = 0

    # Reshape volume to match the cluster labels
    adapted_vol = np.argwhere(volume)

    # Recapture the original shape of the volume
    reconstruced_volume = np.full(volume.shape, -1, dtype=int)
    reconstruced_volume[adapted_vol[:,0], adapted_vol[:,1], adapted_vol[:,2]] = cluster_labels

    return reconstruced_volume



# ================================================================
# 2. Section: Generates Initial Centers
# ================================================================
def generate_initial_centers(volume: np.ndarray, nr_centers: int = 20, range_val: int = 15) -> np.ndarray:
    # Get artificial center of segment
    mean_point = np.mean(np.argwhere(volume), axis=0)

    # Create two starting points, one for the left and one for the right hemisphere (borders of the volume)
    start_left = np.array([mean_point[0], mean_point[1], 0])
    start_right = np.array([mean_point[0], mean_point[1], volume.shape[2]-1])
    centers = [np.array([start_left, start_right])]

    # Generate random points around the starting points to create more centers
    for i in range(nr_centers - 1):
        random_value_y = np.random.random()*(range_val*2) - range_val
        ranndom_value_z = np.random.random()*(range_val*2) - range_val

        random_left = start_left + np.array([ranndom_value_z, random_value_y, 0])
        random_right = start_right + np.array([ranndom_value_z, random_value_y, 0])

        centers += [np.array([random_left, random_right])]

    return np.array(centers)
