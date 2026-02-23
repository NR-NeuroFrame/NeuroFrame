# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm

from ...mouse import Mouse
from .separation_methods import (
    trivial_separation,
    naive_grouping_separation,
    fragmented_grouping_separation,
    destroying_bridges_separation,
    clustering_separation
)



# ================================================================
# 1. Section: Functions
# ================================================================
def separate_segments(mouse: Mouse):
    segments_labels = mouse.segmentation.labels
    segmentations = mouse.segmentation.data

    for seg_lab in tqdm(segments_labels, desc="Separating segments", unit="seg"):
        seg_vol = np.where(segmentations == seg_lab, 1, 0)

        left, right = separate_single_segment(seg_vol)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Apply for one segment
# ──────────────────────────────────────────────────────
def separate_single_segment(volume: np.ndarray) -> tuple:
    # 0. Makes sure any segment with 1 or 0 voxel does not go through all of it
    # TODO

    # 1. Try trivial separation first
    trivial_sides, is_trivial = trivial_separation(volume)
    if(is_trivial): return trivial_sides

    # 2. In case it fails it does naive separation (checks for connected clusters)
    naive_grouping, is_naive_groupable, cluster_data = naive_grouping_separation(volume)
    if(is_naive_groupable): return naive_grouping

    # 3. If naive grouping does not work, let's try to add the fragments to the main clusters
    fragmented_grouping, is_groupable = fragmented_grouping_separation(cluster_data)
    if(is_groupable): return fragmented_grouping

    # 4. Let's see if we can break bridges to get separable fragments (directional eroded)
    unbridged_z, is_separated_z = destroying_bridges_separation(volume, "z_directed")
    if(is_separated_z): return unbridged_z

    # 5. Let's see if we can break bridges to get separable fragments (ball eroded)
    unbridged_ball, is_separated_ball = destroying_bridges_separation(volume, "ball")
    if(is_separated_ball): return unbridged_ball

    # 6. Try Kmeans clustering as last resort
    clustered, is_center_found = clustering_separation(volume)
    if(is_center_found): return clustered

    # 7. If all fails, it will be assigned to the closest
