# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd
import numpy as np

from dataclasses import dataclass
from time import time
from tqdm import tqdm
from multiprocessing import Pool

from neuroframe.pipeline.stereotaxic_coords.stereotaxic_dataclass import LRPair

from ..logger import logger
from ..mouse import Mouse
from .stereotaxic_coords import StereotaxicConfig
from ..utils import separate_volume

_CTX = None



# ================================================================
# 1. Section: Stereotaxic Coordinates Extraction
# ================================================================
def stereotaxic_coordinates(
    mouse: Mouse,
    config: StereotaxicConfig
) -> pd.DataFrame:

    # 1. Get the list of segments
    list_of_segments_ids = mouse.segmentation.labels

    # 2. Initialize data that is common for every segment (memory efficient)
    global_data = GlobalDataWorker(
        segmentation_data=mouse.segmentation.data,
        skull_points=config.skull_points,
        voxel_size=mouse.voxel_size[0],
        mode=config.mode
    )

    # 4. Iterate through each segment
    with Pool(initializer=_init_worker, initargs=(global_data)) as pool:
        results = list(tqdm(
            pool.imap_unordered(centroid_calculation, list_of_segments_ids),
            total=len(list_of_segments_ids))
        )

    # 5. Get the path where the data will be stored
    folder = mouse.folder
    if(config.group_folder is None): results_path = f'{folder}/{config.file_name}.csv'
    else: results_path = f"{config.group_folder}/{mouse.id.lower()}_{config.file_name}.csv"

    # 6. Initializes the dataframe

    pass



# ================================================================
# 2. Section: Multiprocess
# ================================================================
@dataclass
class GlobalDataWorker:
    segmentation_data: np.ndarray
    skull_points: tuple
    voxel_size: float
    mode: str

def _init_worker(global_data: GlobalDataWorker):
    global _CTX
    _CTX = global_data

@dataclass
class LRArrayPair:
    left_array: np.ndarray
    right_array: np.ndarray

@dataclass
class CentroidData:
    mean_center: LRPair | None
    inner_center: LRPair | None
    shape_center: LRPair | None

    inner_mask: np.ndarray | None

    segment_volume: LRPair



def centroid_calculation(segment_id: int) -> dict:
    # 1. Isolate the segment data
    segment_mask = np.where(_CTX.segmentation_data == segment_id, 1, 0)

    # 2. Separate between left and right (complex)
    lr_segment_pair = separate_segment(segment_mask)

    # 3. Compute centroid(s) center, volume and inner mask (CentroidData)
    centroid_data = get_centroid_data(lr_segment_pair)

    #   4.4. Compute statistics and space conversions


    # Create a dictionary to store the results
    rec = {'id': segment}

    try:
        # Compute the center of the segment. First check if it is separable, then compute the center accordingly
        rec = extract_coords((left_hemisphere, right_hemisphere), rec, ref_coords, voxel_size, mode, verbose)
    except Exception as e:
        if(verbose >= 2):
            print(f"    🚨 Error processing segment {segment}: {e}")
            print(f"    🚨 Running segment with higher verbosity for debugging")
            try:
                rec = extract_coords((left_hemisphere, right_hemisphere), rec, ref_coords, voxel_size, mode, verbose=10)
            except Exception as e:
                print(f"    🚨 Error processing segment {segment} with high verbosity: {e}")

    return rec

def separate_segment(segment_mask: np.ndarray):
    pass

def get_centroid_data(lr_segment_pair: LRArrayPair): # return a CenterData dataclass
    # 1. Compute Center
    #   1.1. Compute the mean centroid
    #   1.2. Compute the inner centroid (save the inner mask)
    #   1.3. Compute the WT-shape adjusted centroid
    # 2. Compute the Volume
    pass

def get_volume(lr_segment_pair: LRArrayPair):
    pass
