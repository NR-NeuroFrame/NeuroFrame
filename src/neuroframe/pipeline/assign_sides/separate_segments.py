# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

from ...mouse import Mouse
from .validate import validate_lateralization
from .separation_methods import (
    LateralizedSegment,
    trivial_separation,
    naive_grouping_separation,
    fragmented_grouping_separation,
    destroying_bridges_separation,
    clustering_separation,
    failed_separation
)

LEFT_TAG: int = 1
RIGHT_TAG: int = 2



# ================================================================
# 1. Section: Functions
# ================================================================
def separate_segments(mouse: Mouse) -> np.ndarray:
    # 1. Extracts the data
    segments_labels = mouse.segmentation.labels
    segmentations = mouse.segmentation.data

    # 2. Loops over eveyr different segment
    lateralized_volume = np.zeros_like(segmentations)
    for seg_lab in tqdm(segments_labels[:10], desc="Separating segments", unit="seg"):
        # 1. Extracts left vs right segments
        seg_vol = np.where(segmentations == seg_lab, 1, 0)
        lateralized_segment = separate_single_segment(seg_vol)

        # 2. Assigns side labels
        lateralized_segment.left *= LEFT_TAG
        lateralized_segment.right *= RIGHT_TAG

        # 3. Stores it
        segment = lateralized_segment.left + lateralized_segment.right
        lateralized_volume += segment

    # 3. Makes sure there are no overlaps
    validate_lateralization(lateralized_volume)

    # 4. Save the channel near the data
    mouse.hemisphere = lateralized_volume

    # 5. Also save lateralization description

    return lateralized_volume

# ──────────────────────────────────────────────────────
# 1.1 Subsection: Apply for one segment
# ──────────────────────────────────────────────────────
def separate_single_segment(volume: np.ndarray) -> LateralizedSegment:
    # 0. Makes sure any segment with 1 or 0 voxel does not go through all of it
    if(np.count_nonzero(volume) == 1): return failed_separation(volume)

    # 1. Try trivial separation first
    trivial_output = trivial_separation(volume)
    if(trivial_output.condition): return trivial_output.lateralized_segment

    # 2. In case it fails it does naive separation (checks for connected clusters)
    naive_ouput = naive_grouping_separation(volume)
    if(naive_ouput.condition): return naive_ouput.lateralized_segment

    # 3. If naive grouping does not work, let's try to add the fragments to the main clusters
    fragment_output = fragmented_grouping_separation(naive_ouput.cluster_data)
    if(fragment_output.condition): return fragment_output.lateralized_segment

    # 4. Let's see if we can break bridges to get separable fragments (directional eroded)
    z_erode_output = destroying_bridges_separation(volume, "z_directed")
    if(z_erode_output.condition): return z_erode_output.lateralized_segment

    # 5. Let's see if we can break bridges to get separable fragments (ball eroded)
    ball_erode_output = destroying_bridges_separation(volume, "ball")
    if(ball_erode_output.condition): return ball_erode_output.lateralized_segment

    # 6. Try Kmeans clustering as last resort
    clustering_output = clustering_separation(volume)
    if(clustering_output.condition): return clustering_output.lateralized_segment

    # 7. If all fails, it will be assigned to the closest
    return failed_separation(volume)
