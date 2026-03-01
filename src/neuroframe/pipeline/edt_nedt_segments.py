# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from ..logger import logger
from ..mouse import Mouse
from ..save import save_channel
from ..mouse_data import SegmentationEDT, SegmentationNEDT

ROI_PAD: int = 1



# ================================================================
# 1. Section: Functions
# ================================================================
def edt_segments(mouse: Mouse) -> tuple[np.ndarray, np.ndarray]:
    # 0. Assert there is a lateralization
    if(mouse.hemisphere is None):
        logger.critical("Without lateralization it is impossible to continue")
        raise TypeError("No lateralization found, without lateralization it is impossible to continue")

    # 1. Extract the data
    segments_labels = mouse.segmentation.labels
    segmentations = mouse.segmentation.data
    lateralization = mouse.hemisphere.data

    # 2. Loops over each segment
    edt_space = np.zeros_like(segmentations)
    nedt_space = np.zeros_like(segmentations)
    for seg_lab in tqdm(segments_labels, desc="Computing EDT and NEDT", unit="seg"):
        # 2.1 Extract the segment lateral data
        seg_lat = np.where(segmentations == seg_lab, lateralization, 0)
        seg_left = np.where(seg_lat == 1, 1, 0)
        seg_right = np.where(seg_lat == 2, 1, 0)

        # 2.2 Compute the edt space
        left_edt = compute_edt(seg_left)
        right_edt = compute_edt(seg_right)

        # 2.3 Compute the nedt space
        left_nedt = compute_nedt(left_edt)
        right_nedt = compute_nedt(right_edt)

        # 2.4 Merge back the computations
        edt_space += left_edt
        edt_space += right_edt
        nedt_space += left_nedt
        nedt_space += right_nedt

    # 3. Save the channel near the data
    edt_path = save_channel(mouse, edt_space, "edt")
    mouse.add_path(edt_path, SegmentationEDT)
    nedt_path = save_channel(mouse, nedt_space, "nedt")
    mouse.add_path(nedt_path, SegmentationNEDT)
    print(f"EDT Channel saved at {edt_path} and NEDT at {nedt_path}")

    return edt_space, nedt_space


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def compute_edt(volume: np.ndarray) -> np.ndarray:
    # 1. Guarantees that the volume is not empty
    if not np.any(volume): return np.zeros_like(volume)

    # 2. Computes entries for ROI
    idx = np.argwhere(volume)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1

    # 3. Get's the ROI limits
    start = np.maximum(mins - ROI_PAD, 0)
    stop  = np.minimum(maxs + ROI_PAD, volume.shape)
    sl = tuple(slice(s, e) for s, e in zip(start, stop))

    # 4. Computes edt only in the ROI
    edt_roi = distance_transform_edt(volume[sl]).astype(np.float32)

    # 5. Remerges the ROI to the oiginal position
    edt_volume = np.zeros(volume.shape, dtype=np.float32)
    edt_volume[sl] = edt_roi

    return edt_volume

def compute_nedt(edt_space: np.ndarray) -> np.ndarray:
    # 1. Get's the max for min-max norm
    edt_max = float(np.max(edt_space))

    # 2. Deals with empty sides
    if edt_max <= 0: return np.zeros_like(edt_space, dtype=np.float32)

    return edt_space / edt_max
