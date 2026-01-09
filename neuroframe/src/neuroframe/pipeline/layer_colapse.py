# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd
import numpy as np

from ..logger import logger
from ..mouse import Mouse
from ..assertions import assert_all_from_same_parent



# ================================================================
# 1. Section: Preparing Volume - Layer Collapsing
# ================================================================
def layer_colapsing(mouse: Mouse, data: pd.DataFrame) -> np.ndarray:
    segments = mouse.segmentation.data
    original_nr_segments = len(mouse.segmentation.labels)
    layer_indexs = []

    # Goes through every row in the processed data, if the name contains "Layer" it will store the index
    for entry in range(len(data)):
        logger.debug(f"Checking segment {data['id'].iloc[entry]} - {data['name'].iloc[entry]} Has layer? {'layer' in data['name'].iloc[entry].lower()}")

        # Initiate, continue or terminate a layer if conditions are met
        segments, layer_indexs = check_and_build_layer(segments, data, layer_indexs, entry)

    # Finish any layer that could be left open
    if(len(layer_indexs)> 0): segments, layer_indexs = terminate_layer(segments, data, layer_indexs)

    # Updates the mice only if the segments have changed
    labels = update_mouse_segments(mouse, segments, original_nr_segments)
    return labels


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Preparing Volume - Helpers
# ──────────────────────────────────────────────────────
def check_and_build_layer(segments: np.ndarray, data: pd.DataFrame, layer_indexs: list, entry: int) -> tuple:
    # Start if the layer_indexs is empty and the name contains "Layer"
    is_start_layer = len(layer_indexs) == 0 and 'layer' in data['name'].iloc[entry].lower()
    # Or if the layer_indexs is not empty and the parent_id of the current entry is the same as the parent_id of the first layer in the layer_indexs
    is_continue_layer = len(layer_indexs) > 0 and data['parent_id'].iloc[entry] == data['parent_id'].iloc[layer_indexs[0]] and 'layer' in data['name'].iloc[entry].lower()
    # Or if nothing checks, but layer index is not empty (finishing the layer)
    is_terminate_layer = len(layer_indexs) > 0 and not (is_start_layer or is_continue_layer)

    # Start a Layer or Continue a Layer
    if(is_start_layer or is_continue_layer): layer_indexs.append(entry)

    # Finish a Layer
    elif(is_terminate_layer): 
        # Update the segments (colpasing the layers)
        segments = terminate_layer(segments, data, layer_indexs)
        layer_indexs = initiate_new_layer(data, layer_indexs, entry)

    return segments, layer_indexs

def terminate_layer(segments: np.ndarray, data: pd.DataFrame, layer_indexs: list) -> None:
    logger.debug(f"All layer names in layer_indexs: {[data['name'].iloc[i] for i in layer_indexs]}")
    
    # Check if every layer has the same parent_id
    assert_all_from_same_parent(data, layer_indexs)

    # Get the new voxel value for the colpased layer
    parent_id = data['parent_id'].iloc[layer_indexs[0]].astype(int)

    # Remove evrything after the str layer in the layer name
    layer_name = data['name'].iloc[layer_indexs[0]]
    layer_name = layer_name.split('layer')[0].strip()

    # Updates the layer voxel values to the parent_id
    for index in layer_indexs: segments[segments == data['id'].iloc[index]] = parent_id

    logger.debug(f'Layer: {layer_name} - Parent: {parent_id}')
    return segments

def update_mouse_segments(mouse: Mouse, segments: np.ndarray, original_nr_labels: int) -> np.ndarray:
    # Get the number of segments before and after the colapsing
    new_nr_segments = len(np.unique(segments)) - 1  # Exclude background segment

    # Only updates the segments if the number of segments has changed
    if(original_nr_labels != new_nr_segments):
        logger.info("Reduced from ", original_nr_labels, "to", new_nr_segments, "segments")
        mouse.segmentation.data = segments
    else: logger.info("No layers found to colapse.")

    # Get the updated labels after colapsing (or no colapsing)
    labels = mouse.segmentation.labels
    logger.debug(f"Labels after collapsing: {labels}")
    return labels

def initiate_new_layer(data: pd.DataFrame, layer_indexs: list, entry: int) -> list:
    layer_indexs = []

    # In the case of the entry that activated the termination is part of another layer, it will start a new layer storage
    if('layer' in data['name'].iloc[entry].lower()): 
        logger.debug(f"            Initiating, right away, a new layer with entry {entry} - {data['name'].iloc[entry]}")
        layer_indexs.append(entry)

    return layer_indexs