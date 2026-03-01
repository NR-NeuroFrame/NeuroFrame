# ================================================================
# 0. Section: IMPORTS
# ================================================================
import warnings

import numpy as np

from tqdm import tqdm

from ...mouse import Mouse
from .validate import validate_lateralization
from .summary import build_lateralization_summary
from ...save import (
    save_channel,
    save_summary
)
from .SummaryLateralization import SummaryLateralization
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




warnings.filterwarnings("error", message="Mean of empty slice.*", category=RuntimeWarning)
np.seterr(invalid="raise", divide="raise")



# ================================================================
# 1. Section: Functions
# ================================================================
def separate_segments(mouse: Mouse) -> np.ndarray:
    # 1. Extracts the data
    segments_labels = mouse.segmentation.labels
    segmentations = mouse.segmentation.data

    # 2. Loops over eveyr different segment
    lateralized_volume = np.zeros_like(segmentations)
    summary_array = []
    for seg_lab in tqdm(segments_labels, desc="Separating segments", unit="seg"):
        # 1. Extracts left vs right segments
        seg_vol = np.where(segmentations == seg_lab, 1, 0)
        lateralized_segment = separate_single_segment(seg_vol)

        # 2. Assigns side labels
        lateralized_segment.left *= LEFT_TAG
        lateralized_segment.right *= RIGHT_TAG

        # 3. Stores it
        segment = lateralized_segment.left + lateralized_segment.right
        lateralized_volume += segment

        # 4. Prepares for summary
        summary_array.append(
            SummaryLateralization(
                id=seg_lab,
                separation_method=lateralized_segment.separation_method,
                left_ratio=lateralized_segment.left_ratio,
                right_ratio=lateralized_segment.right_ratio,
            )
        )
    summary_array = np.array(summary_array)

    # 3. Makes sure there are no overlaps
    validate_lateralization(lateralized_volume)

    # 4. Save the channel near the data
    hemisphere_path = save_channel(mouse, lateralized_volume, "hemisphere")

    # 5. Also save lateralization description
    summary_df = build_lateralization_summary(summary_array)
    info_path = save_summary(mouse, summary_df, "hemisphere")

    print(f" Channel saved at {hemisphere_path} wiht info saved at {info_path}")
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
