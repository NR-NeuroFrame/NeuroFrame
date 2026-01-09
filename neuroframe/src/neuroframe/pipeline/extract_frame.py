# ================================================================
# 0. Section: Imports
# ================================================================
import time
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from skimage.morphology import ball, opening
from sklearn.cluster import KMeans

from ..mouse import Mouse
from ..utils import separate_volume, compute_inner_center



# ================================================================
# 1. Section: Stereotaxic Coordinates Extraction
# ================================================================
def stereotaxic_coordinates(mouse: Mouse, reference_df: pd.DataFrame, ref_coords: tuple, mode: str = 'full_inner', 
                            is_parallelized: bool = True, verbose: int = 1, **kwargs) -> pd.DataFrame:
    folder = mouse.get_folder()
    updated_csv_path = kwargs.get('updated_csv_path', f'{folder}/NF_ef_{mode}_coords_fixed.csv')
    if(verbose >= 1): print("EF_StereotaxicCoordinates: Starting the stereotaxic coordinate extraction process!")
    
    # Extracts the needed data
    labels = mouse.get_segmentations().get_labels()
    voxel_size = mouse.get_voxel_size()

    # Updates the mice segmentations to remove any layers that have pairing (layer1,2,3,4,5)
    labels = layer_colapsing(mouse, reference_df, verbose=verbose)

    # Updates the reference DataFrame to remove any entries that do not correspond to the segmentations
    reference_df = preprocess_reference_df(mouse, reference_df, verbose=verbose)

    # Separates the brain
    hemispheres = separate_volume(mouse.get_segmentations().get_data())

    # Calculates the coordinates of the segments in bregma-lambda space (parallelized or not)
    if(is_parallelized): results = parallelized_process(hemispheres, labels, ref_coords, voxel_size, mode, verbose=verbose)
    else: results = non_parallelized_process(mouse, hemispheres, labels, ref_coords, voxel_size, mode, verbose=verbose)

    # Create a DataFrame from the list of result dictionaries and merge the results into your original DataFrame
    res_df = pd.DataFrame(results)
    data = reference_df.merge(res_df, on='id', how='left')

    # Save the updated CSV file
    
    data.to_csv(updated_csv_path, index=False)

    return data





# ================================================================
# 1. Section: Preparing Volume - Layer Collapsing
# ================================================================
def layer_colapsing(mouse: Mouse, data: pd.DataFrame, verbose: int) -> np.ndarray:
    """
    Collapse layers in the segmentation data of a mouse instance.
    This function iterates through each row in the provided DataFrame and checks if the 'name'
    column contains the substring "layer". It uses helper functions to initiate, continue, or
    terminate layers based on the segmentation data. After processing all rows, it finalizes any
    unfinished layers and updates the mouse segments accordingly. The resulting labels are then
    returned as a NumPy array.

    Parameters:
        mice: Mice
            An instance containing mouse segmentation data and associated methods.
        data: pd.DataFrame
            A DataFrame containing processed data with segmentation information; each row should
            have at least 'id' and 'name' columns.
        verbose: int
            The verbosity level for printing debug information.
    Returns:
        np.ndarray
            An array of updated labels corresponding to the mouse segments.
    """
    if(verbose >= 2): print("    🔄 Collapsing layers...")
    segments = mouse.get_segmentations().get_data()
    original_nr_segments = len(mouse.get_segmentations().get_labels())
    layer_indexs = []

    # Goes through every row in the processed data, if the name contains "Layer" it will store the index
    for entry in range(len(data)):
        if(verbose >= 5): print(f"                Checking segment {data['id'].iloc[entry]} - {data['name'].iloc[entry]}")
        if(verbose >= 5): print(f"                Has layer? {'layer' in data['name'].iloc[entry].lower()}")

        # Initiate, continue or terminate a layer if conditions are met
        segments, layer_indexs = check_and_build_layer(segments, data, layer_indexs, entry, verbose=verbose)

    # Finish any layer that could be left open
    if(len(layer_indexs)> 0): segments, layer_indexs = terminate_layer(segments, data, layer_indexs, verbose=verbose)

    # Updates the mice only if the segments have changed
    labels = update_mouse_segments(mouse, segments, original_nr_segments, verbose)
    
    if(verbose >= 4): print(f"            Labels after collapsing: {labels}")
    if(verbose >= 2): print(f"    ✅ Collapsed layers, now {len(labels)} segments in total.\n")
    return labels



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Preparing Volume - Helpers
# ──────────────────────────────────────────────────────
def check_and_build_layer(segments: np.ndarray, data: pd.DataFrame, layer_indexs: list, entry: int, verbose: int) -> tuple:
    """
    Check whether to start a new layer, continue an existing layer, or terminate the current layer.

    This function examines the provided data and layer indices to determine if the current entry:
    - Should start a new layer when the list of layer indices is empty and the 'name' field (from data) contains
        the word "layer" (case insensitive).
    - Should continue the current layer when the list is not empty and the 'parent_id' of the current entry matches
        that of the first entry in layer_indexs.
    - Should terminate the current layer when none of the above conditions hold and layer_indexs is not empty.
        On termination, it updates the segments by calling the terminate_layer function and resets the layer index
        by calling the initiate_new_layer function.

    Parameters:
            segments (np.ndarray): An array representing the current segments, which may be updated upon terminating a layer.
            data (pd.DataFrame): A DataFrame containing entry data with fields such as 'name' and 'parent_id'.
            layer_indexs (list): A list of indices representing entries associated with the currently active layer.
            entry (int): The current index of the entry in the DataFrame to be processed.
            verbose (int): Verbosity level for logging or debugging (passed along to other functions if required).

    Returns:
            tuple: A tuple consisting of the updated segments (np.ndarray) and the updated list of layer indices (list).
    """
    # Start if the layer_indexs is empty and the name contains "Layer"
    is_start_layer = len(layer_indexs) == 0 and 'layer' in data['name'].iloc[entry].lower()
    # Or if the layer_indexs is not empty and the parent_id of the current entry is the same as the parent_id of the first layer in the layer_indexs
    is_continue_layer = len(layer_indexs) > 0 and data['parent_id'].iloc[entry] == data['parent_id'].iloc[layer_indexs[0]]
    # Or if nothing checks, but layer index is not empty (finishing the layer)
    is_terminate_layer = len(layer_indexs) > 0 and not (is_start_layer or is_continue_layer)

    # Start a Layer or Continue a Layer
    if(is_start_layer or is_continue_layer): layer_indexs.append(entry)

    # Finish a Layer
    elif(is_terminate_layer): 
        # Update the segments (colpasing the layers)
        segments = terminate_layer(segments, data, layer_indexs, verbose)
        layer_indexs = initiate_new_layer(data, layer_indexs, entry, verbose)

    return segments, layer_indexs
    
def terminate_layer(segments: np.ndarray, data: pd.DataFrame, layer_indexs: list, verbose: int) -> None:
    """
    Terminate the specified layer segments by updating their voxel values to the parent ID.
    This function verifies that all layers specified by the indices in `layer_indexs`
    share the same parent by calling `alert_layer_diff_parents`. It then retrieves the
    parent ID from the first layer and uses it to replace the voxel values in the
    `segments` array that correspond to each of the layer IDs at the given indices.
    Verbose logging is provided based on the value of `verbose` to trace key steps.

    Parameters:
        segments (np.ndarray): Array of voxel values where specific layer IDs are
            stored and updated.
        data (pd.DataFrame): DataFrame containing layer information with columns such as
            'id', 'parent_id', and 'name'.
        layer_indexs (list): List of indices referring to rows in `data` that represent
            the layers to be terminated.
        verbose (int): Verbosity level; higher values result in more detailed logging.
    Returns:
        np.ndarray: The updated `segments` array with the layer voxel values replaced by
        the parent ID.
    Raises:
        ValueError: This exception may be raised by `alert_layer_diff_parents` if the layers
            in `layer_indexs` do not share the same parent.
    """
    if(verbose >= 4): print(f"            All layer names in layer_indexs: {[data['name'].iloc[i] for i in layer_indexs]}")
    
    # Check if every layer has the same parent_id
    alert_layer_diff_parents(data, layer_indexs)

    # Get the new voxel value for the colpased layer
    parent_id = data['parent_id'].iloc[layer_indexs[0]].astype(int)

    # remove evrything after the str layer in the layer name
    layer_name = data['name'].iloc[layer_indexs[0]]
    layer_name = layer_name.split('layer')[0].strip()

    # Updates the layer voxel values to the parent_id
    for index in layer_indexs: segments[segments == data['id'].iloc[index]] = parent_id

    if(verbose >= 3): print(f'        🎵 Layer: {layer_name} - Parent: {parent_id}')
    return segments

def initiate_new_layer(data: pd.DataFrame, layer_indexs: list, entry: int, verbose: int) -> list:
    """
    Initiate a new layer based on a specified entry in the DataFrame.

    This function clears the given layer index list and checks whether the
    'name' column of the provided DataFrame at the specified entry contains the
    substring "layer" (case insensitive). If it does, the entry is added to the
    new layer index list.

    Parameters:
        data (pd.DataFrame): DataFrame that must contain a 'name' column.
        layer_indexs (list): List intended for storing the indices of layers (will be reset).
        entry (int): The index of the DataFrame row to check.

    Returns:
        list: A new list containing the entry if the condition is met; otherwise, an empty list.
    """
    layer_indexs = []

    # In the case of the entry that activated the termination is part of another layer, it will start a new layer storage
    if('layer' in data['name'].iloc[entry].lower()): 
        if(verbose >= 4): print(f"            Initiating, right away, a new layer with entry {entry} - {data['name'].iloc[entry]}")
        layer_indexs.append(entry)

    return layer_indexs

def update_mouse_segments(mouse: Mouse, segments: np.ndarray, original_nr_labels: int, verbose: int) -> np.ndarray:
    """
    Update the segmentation labels for a given Mice object based on the provided segments array.

    This function compares the number of segments currently stored in the Mice object's segmentation
    data with the number of unique segments found in the provided array (excluding the background
    segment). If the number of segments has changed, it updates the segmentation data in the Mice
    object. Otherwise, it logs that no collapsing of layers was performed.

    Parameters:
        mice (Mice): An instance of the Mice class that contains segmentation data.
        segments (np.ndarray): A numpy array representing the segmentation data.
                               It should include a background segment which is excluded from the count.
        verbose (int): Level of verbosity for logging:
                       - If verbose is greater than or equal to 3, detailed logging is output.

    Returns:
        np.ndarray: The updated array of segmentation labels after processing.
    """
    # Get the number of segments before and after the colapsing
    new_nr_segments = len(np.unique(segments)) - 1  # Exclude background segment

    # Only updates the segments if the number of segments has changed
    if(original_nr_labels != new_nr_segments):
        if(verbose >= 3): print("        📉 Reduced from ", original_nr_labels, "to", new_nr_segments, "segments")
        mouse.get_segmentations().set_data(segments)
    else: 
        if(verbose >= 3): print("        ❌ No layers found to colapse.")

    # Get the updated labels after colapsing (or no colapsing)
    labels = mouse.get_segmentations().get_labels()
    return labels

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 1.1.1 Sub-subsection: Alerts and Warnings
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def alert_layer_diff_parents(data: pd.DataFrame, layer_indexs: list) -> None:
    """
    Check if the layers specified by the indices in layer_indexs have different parent IDs.

    This function examines the parent_id values from the provided DataFrame using the given
    indices. If more than one unique parent_id is found, it issues a warning indicating that the
    layers do not share the same parent_id, which may cause issues during the collapsing process.

    Parameters:
        data (pd.DataFrame): DataFrame containing layer data with a 'parent_id' column.
        layer_indexs (list): List of indices corresponding to the layers to be checked.

    Returns:
        None

    Raises:
        Warning: A warning is issued if not all of the specified layers have the same parent_id.
    """
    parents_different = len(set(data['parent_id'].iloc[layer_indexs])) > 1
    if(parents_different): warnings.warn("WARNING: Not all layers have the same parent_id, this may cause issues in the colapsing process.", stacklevel=2)





# ================================================================
# 2. Section: Preprocessing for Reference DataFrame
# ================================================================
def preprocess_reference_df(mouse: Mouse, reference_df: pd.DataFrame, verbose: int = 1) -> pd.DataFrame:
    """
    Remove every entry from the reference DataFrame that does not correspond
    to any segmentation label defined in the provided mice instance.

    Parameters:
        mice (Mice): An instance of Mice used to retrieve segmentation labels.
        reference_df (pd.DataFrame): A DataFrame containing reference entries with 'id' values.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only entries with an 'id'
        present in the mice segmentation labels.
    """
    if(verbose >= 2): print("    🔄 Preprocessing the reference DataFrame...")
    labels = mouse.get_segmentations().get_labels()

    # Remove every entry from the reference DataFrame that does not correspond to any segmentation label
    reference_df = reference_df[reference_df['id'].isin(labels)]

    # Print any entri that was present in the labels but not in the reference DataFrame
    missing_labels = set(labels) - set(reference_df['id'])
    if len(missing_labels) > 0:
        print(f"        ❗ Missing labels in reference DataFrame: {missing_labels}")

    # Remove the data from columns called reed, blue and green
    if 'red' in reference_df.columns: reference_df = reference_df.drop(columns=['red'])
    if 'blue' in reference_df.columns: reference_df = reference_df.drop(columns=['blue'])
    if 'green' in reference_df.columns: reference_df = reference_df.drop(columns=['green'])

    if(verbose >= 4): print(f"            20 Entries of Dataframe: {reference_df.head(20)}")
    if(verbose >= 2): print(f"    ✅ Preprocessed reference DataFrame: {len(reference_df)} entries remaining.\n")
    return reference_df





# ================================================================
# 3. Section: Paralelized Processing of Center Coordinates
# ================================================================
def parallelized_process(hemispheres: np.ndarray, labels: np.ndarray, ref_coords: tuple, voxel_size: float, mode: str, verbose: int) -> list:
    """
    Parallelize processing of segments by computing their center coordinates.

    This function distributes the computation across multiple processes using a pool
    of workers. It processes each segment from the 'labels' array using the parameters
    provided, including hemisphere data, reference coordinates, voxel size, and mode,
    and returns a list with the computed results.

    Parameters:
        hemispheres (np.ndarray): Array representing hemisphere data.
        labels (np.ndarray): Array of segments (labels) to be processed.
        ref_coords (tuple): Reference coordinates used for center coordinate calculation.
        voxel_size (float): The size of the voxel used in the computation.
        mode (str): Processing mode that dictates how the computation is performed.
        verbose (int): Verbosity level for printing progress and timing information.

    Returns:
        list: A list of computed center coordinates for each segment.
    """
    if(verbose >= 2): print(f"    🔄 Starting PARALLELIZED processing of {len(labels)} segments...")
    args_list = [(segment, hemispheres, ref_coords, voxel_size, mode, verbose) for segment in labels]

    start_time = time.time()
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(center_coord_worker, args_list), total=len(args_list)))
    if(verbose >= 2): print(f"    ✅ Processed {len(labels)} segments in {time.time() - start_time:.2f} s.\n")

    return results

def non_parallelized_process(mouse: Mouse, hemispheres: np.ndarray, labels: np.ndarray, ref_coords: tuple, voxel_size: float, mode: str, verbose: int) -> list:
    """
    Process segments in a non-parallelized manner using the center_coord_worker.
    It processes each segment from the 'labels' array using the parameters
    provided, including hemisphere data, reference coordinates, voxel size, and mode,
    and returns a list with the computed results.

    Parameters:
        mice (Mice): The Mice object containing necessary data for processing.
        hemispheres (np.ndarray): An array indicating the hemispheres information.
        labels (np.ndarray): An array of segments to be processed.
        ref_coords (tuple): A tuple of reference coordinates.
        voxel_size (float): The voxel size to be used in the processing.
        mode (str): Mode identifier that controls specific processing behavior.
        verbose (int): Verbosity level that dictates the amount of logging information.

    Returns:
        list: A list of dictionaries, each containing the processing results for a segment.
    """
    if(verbose >= 2): print(f"    🔄 Starting NON-PARALLELIZED processing of {len(labels)} segments...")
    results = []

    start_time = time.time()
    for segment in labels:
        # Gets the dictionary with all coordinates of a given segment
        args_item = (segment, hemispheres, ref_coords, voxel_size, mode, verbose)
        result_dict: dict = center_coord_worker(args_item)
        results.append(result_dict)
        
    if(verbose >= 2): print(f"    ✅ Processed {len(labels)} segments in {time.time() - start_time:.2f} s.\n")
    return results

def center_coord_worker(args: tuple) -> dict:
    """
    Compute the center coordinates of a given segment across two hemispheres.
    This function creates binary masks for the left and right hemispheres based on
    the provided segment and then computes the center of the segment by calling
    an external function, extract_coords. The processing behavior can be modified
    using the mode parameter, and verbose output is optionally printed.

    Parameters:
        args (tuple): A tuple containing the following elements
            - segment: The identifier for the segment.
            - hemispheres: A tuple of two numpy arrays corresponding to the left and 
              right hemisphere data.
            - ref_coords: Reference coordinates used in the center extraction process.
            - voxel_size: The size of the voxel used for scaling the coordinates.
            - mode: The mode specifying how the center should be computed.
            - verbose: An integer controlling the verbosity of the output (e.g., debug information).

    Returns:
        dict: A dictionary containing the segment identifier ('id') and the computed
        center coordinates along with any additional data returned by extract_coords.
    """
    segment, hemispheres, ref_coords, voxel_size, mode, verbose = args
    if(verbose >= 5): print(f"                → Processing segment {segment}...")
    
    # Create binary mask for each hemisphere of the segment
    left_hemisphere = np.where(hemispheres[0] == segment, 1, 0)
    right_hemisphere = np.where(hemispheres[1] == segment, 1, 0)
    
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





# ================================================================
# 4. Section: Each Segement Center Coordinates Extraction
# ================================================================
def extract_coords(hemispheres: tuple, rec: dict, ref_coords: np.ndarray, voxel_size: float, mode: str, verbose: int) -> dict:
    # Extract the centroids in voxel and um coordinates
    centroids, volumes_sizes, separation_method = get_centroid(hemispheres, mode, verbose=verbose)
    ref_centroids, volumes_um = convert_to_ref(centroids, ref_coords, voxel_size, volumes_sizes, verbose=verbose)

    # Unpack the centroids
    left_centroid, right_centroid = centroids
    left_ref_centroid, right_ref_centroid = ref_centroids

    # Extract the statistics for the um coordinates
    mean_um, std_um, ste_um = extract_statistics(ref_centroids, verbose=verbose)

    # Reorder from [z, y, x] to [x, y, z]
    left_ref_centroid, right_ref_centroid, mean_um, std_um, ste_um = map(lambda coords: reorder_coords(coords, 3), [left_ref_centroid, 
                                                                                                                    right_ref_centroid, mean_um, std_um, ste_um])
    left_centroid, right_centroid = map(lambda coords: reorder_coords(coords, 0), [left_centroid, right_centroid])

    # Update the dictionary with the computed values
    rec.update({
        'Separation Method': separation_method,
        'xyz (um) - L': left_ref_centroid, 'xyz (um) - R': right_ref_centroid, 'mean xyz (um)': mean_um, 'ste xyz (um)': ste_um, 'std xyz (um)': std_um,
        'volume (um^3) - L': volumes_um[0], 'volume (um^3) - R': volumes_um[1],
        'xyz (voxel) - L': left_centroid, 'xyz (voxel) - R': right_centroid,
        'volume (voxel) - L': volumes_sizes[0], 'volume (voxel) - R': volumes_sizes[1]
    })

    return rec



# ──────────────────────────────────────────────────────
# 4.1 Subsection: Each Segement Center Coordinates Extraction - Centroid
# ──────────────────────────────────────────────────────
def get_centroid(hemispheres: np.ndarray, mode: str, verbose: int):
    if(verbose >= 6): print(f"                    Extracting the Centroid Coordinates")

    # Extract the hemispheres and other data
    left_hemisphere, right_hemisphere = hemispheres
    volume = left_hemisphere + right_hemisphere
    midline_x = left_hemisphere.shape[2] // 2

    # Check if any of the hemispheres is empty, by counting the non-zero elements
    hemisphere_not_empty = (np.count_nonzero(left_hemisphere) != 0) and (np.count_nonzero(right_hemisphere) != 0)

    # Check if the midline cuts any segment
    is_cut = (np.count_nonzero(volume[:, :, midline_x]) != 0)

    # Debugging output
    if(verbose >= 7): 
        print(f"                        ? Is the Hemisphere Empty? {not hemisphere_not_empty}")
        print(f"                        ? Is the Midline Cutting the Segments? {is_cut}")
        print(f"                        ? Inspecting separation: Left: {np.count_nonzero(left_hemisphere)} — Right: {np.count_nonzero(right_hemisphere)}")

    # Because the separation is clean, just use midline for trivial separation
    if hemisphere_not_empty and not is_cut: 
        if(verbose >= 8): print("                            😁 Trivial separated centroids")
        centroids, volumes_sizes = mode_centroid_calculation(hemispheres, mode)
        separation_method = 'Trivial'
    
    # This handles the other cases (separable, non separable, and complex separations)
    else:
        if(verbose >= 8): print("                            😅 Separation was not trivial, complex approach needed")
        centroids, volumes_sizes, separation_method = complex_separated_centroids(volume, mode, verbose)

    return centroids, volumes_sizes, separation_method

def get_centroid_tip(hemispheres: np.ndarray, mode: str, verbose: int, tip: str):
    # Extract the hemispheres and other data
    left_hemisphere, right_hemisphere = hemispheres
    volume = left_hemisphere + right_hemisphere

    # Because the separation is clean, just use midline for trivial separation
    if tip == 'Trivial': pass
    elif tip == 'Naive Separation' or tip == 'Naive Separation (Fragmented)':
        labeled_array, num_features = label(volume)
        labeled_array, features = reorder_labels_array(labeled_array)
        hemishpheres = rebuild_hemispheres(labeled_array, verbose)
    elif tip == 'Opening Separation':
        labeled_array = try_destroying_bridges(volume, verbose)
        hemishpheres, rebuild_hemispheres(labeled_array, verbose)
    elif tip == 'KMeans Clustering':
        labeled_array = try_clustering_hemispheres(volume, verbose)
        hemishpheres, rebuild_hemispheres(labeled_array, verbose)

    centroids, volumes_sizes = mode_centroid_calculation(hemispheres, mode)

    return centroids, volumes_sizes
    

def complex_separated_centroids(volume: np.ndarray, mode: str, verbose: int) -> tuple:
    # Assess if is true separable, if they are it rebuilds the hemispheres
    hemispheres, separation_method = evaluate_cluster_separability(volume, verbose=verbose)

    # if not separable, send it and classify it the same for the left and right
    if(not isinstance(hemispheres, tuple)): centroids, volumes_sizes = mode_centroid_calculation((hemispheres, hemispheres), mode)

    # If true separable then send both of the hemispheres to the compute_inner_center function
    else: centroids, volumes_sizes = mode_centroid_calculation(hemispheres, mode)

    return centroids, volumes_sizes, separation_method

def mode_centroid_calculation(hemispheres: tuple, mode: str) -> tuple:
    """
    Calculates the centroids and volume sizes of the left and right hemispheres
    based on the specified mode.

    Parameters:
        hemispheres (tuple): A tuple containing two NumPy arrays representing the left
            and right hemispheres.
        mode (str): A string specifying the calculation mode. Supported modes include
            - 'full_inner': Calculates the centroid using the inner center method.
            - 'full_mean': Calculates the centroid as the mean of the indices of non-zero elements.

    Returns:
        tuple: A tuple containing
            - centroids (tuple): A tuple with the centroids for the left and right hemispheres.
            - volumes_sizes (tuple): A tuple with the voxel counts (number of non-zero elements)
              for the left and right hemispheres.
    """
    left_hemisphere, right_hemisphere = hemispheres

    # Compute the center of the left and right hemisphere according to the mode
    if(mode == 'full_inner'): 
        left_centroid = compute_inner_center(left_hemisphere)
        right_centroid = compute_inner_center(right_hemisphere)
    elif(mode == 'full_mean'): 
        left_centroid = np.mean(np.argwhere(left_hemisphere), axis=0)
        right_centroid = np.mean(np.argwhere(right_hemisphere), axis=0)

    volumes_sizes = (np.count_nonzero(left_hemisphere), np.count_nonzero(right_hemisphere))
    centroids = (left_centroid, right_centroid)

    return centroids, volumes_sizes

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.1.1 Sub-subsection: Complex Separated Centroids (Helpers)
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def evaluate_cluster_separability(volume: np.ndarray, verbose: int):
    # Label the volume using the conectivity structure
    labeled_array, num_features = label(volume)
    labeled_array, features = reorder_labels_array(labeled_array)

    if(verbose >= 9): print(f"                                Number of features found when naive clustering: {num_features}")

    # Assess if the clusteres are real clusters or just noise
    has_two_direct_features = (num_features == 2 and features[1]/features[0]*100 > 50)

    if(has_two_direct_features):
        if(verbose >= 8): print(f"                            🤓 Only two features found and both with relevance → Complex but Separable")
        separation_method = 'Naive Separation'
        return rebuild_hemispheres(labeled_array, verbose), separation_method
    
    relevant_features = extract_relevant_features_otsu(features)
    if(verbose >= 9): print(f"                                The relevant features (above threshold) are: {relevant_features}")

    # If there are two  or morerelevant sizes, it means the hemispheres are separable
    # This means we need to merge the non relevant pieces to the closest hemisphere
    if(len(relevant_features) >= 2): 
        if(verbose >= 8): print(f"                            🤓 More than two relevant features found, rebuilding to the biggest 2 → Complex but Separable")
        separation_method = 'Naive Separation (Fragmented)'
        return rebuild_hemispheres(labeled_array, verbose), separation_method
    
    if(verbose >= 8): print(f"                            😅 Naive Clustering was not enough to find separation. Trying Destroying Possible Bridges")

    labeled_array = try_destroying_bridges(volume, verbose)

    if(len(np.unique(labeled_array)) > 2): 
        separation_method = 'Opening Separation'
        return rebuild_hemispheres(labeled_array, verbose), separation_method
    
    if(verbose >= 8): print(f"                            😅 Opening was not enough to find separation. Trying KMeans Clustering")
    labeled_array = try_clustering_hemispheres(volume, verbose)

    if(len(np.unique(labeled_array)) > 2):
        separation_method = 'KMeans Clustering'
        return rebuild_hemispheres(labeled_array, verbose), separation_method

    if(verbose >= 8): print(f"                            😭 Every clustering method failed to find separation! -> Not Separable")
    separation_method = 'Not separable'
    return volume, separation_method

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.1.2 Sub-subsection: Complex Separated Centroids - Clustering Approach
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def try_clustering_hemispheres(volume: np.ndarray, verbose:int, nr_centers: int = 30) -> np.ndarray:
    """
    This function attempts to segment a volume into hemispheres by generating multiple sets of initial
    cluster centers based on lateralized means and performing k-means clustering on each set.
    It verifies that the clustering satisfies the lateralized condition. If a valid clustering is found,
    the hemispheres are built from the clustering labels and returned. Otherwise, the original volume
    is returned.

    Parameters:
        volume (np.ndarray): A numpy array representing the volume to be segmented.
        nr_centers (int): The number of cluster centers to generate. Default value is 20.
    Returns:
        np.ndarray: An array with the segmented hemispheres if a valid lateralized clustering is found,
                    or the original volume if no valid clustering is achieved.
    """
    # Check if the volume is empty or only has one voxel
    if(np.count_nonzero(volume) <= 1): return volume

    # Generates a set of initial starting points based on the lateralized means
    random_centers = generate_initial_centers(volume, nr_centers=nr_centers)

    # Loop until the centers obtained follow the lateralized condition 
    for centers in random_centers:
        # Perform kmeans
        cluster_centers, cluster_labels, is_centers_found = perform_kmeans(volume, centers)

        if(is_centers_found): 
            if(verbose >= 8): print(f"                            🤓 KMeans clustering method found separation! → Most Complex but Separable")
            labeled_array = build_hemispheres_from_clustering(volume, cluster_labels)
            return labeled_array
        
    # If no lateralized centers are found, return the original volume (it is not separable)
    return volume

def build_hemispheres_from_clustering(volume: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """
    Build hemispheres from clustering by reassigning labels and reconstructing a volume.

    This function takes a 3D volume and an array of clustering labels, reassigns the labels 
    (switching -1 to 0 and 0 to 2), and then projects these labels back onto the original volume 
    shape. Points in the volume that do not correspond to any clustering label are set to -1.

    Parameters:
        volume (np.ndarray): A 3D NumPy array representing the input volume. Non-zero entries 
            indicate the points from which cluster labels are derived.
        cluster_labels (np.ndarray): A 1D NumPy array containing cluster labels corresponding 
            to the non-zero points in the volume.

    Returns:
        np.ndarray: A reconstructed 3D NumPy array with the same shape as the input volume, where 
            the labeled points have been updated according to the cluster label reassignment, and 
            all other points are set to -1.
    """
    # Re assign values (-1 -> 0, 0->2)
    cluster_labels[cluster_labels == 0] = 2
    cluster_labels[cluster_labels == -1] = 0

    # Reshape volume to match the cluster labels
    adapted_vol = np.argwhere(volume)

    # Recapture the original shape of the volume
    reconstruced_volume = np.full(volume.shape, -1, dtype=int)
    reconstruced_volume[adapted_vol[:,0], adapted_vol[:,1], adapted_vol[:,2]] = cluster_labels

    return reconstruced_volume

def perform_kmeans(volume: np.ndarray, centers: np.ndarray | list) -> tuple:
    """
    Perform k-means clustering on a non-zero volume and check lateralization of the resulting cluster centers.

    This function adapts the input volume by extracting the indices of its non-zero elements,
    performs k-means clustering with a fixed number of clusters and provided initial centroids,
    and then evaluates whether the calculated cluster centers satisfy a lateralization condition.

    Parameters:
        volume (np.ndarray): A multidimensional NumPy array representing the input volume data.
        centers (np.ndarray | list): Initial centroids for the k-means algorithm.

    Returns:
        tuple: A tuple containing
            - np.ndarray: The cluster centers computed by k-means.
            - np.ndarray: The labels assigned to each point in the volume by k-means.
            - bool: True if the cluster centers meet the lateralization condition, False otherwise.
    """
    # prepares the volume for kmeans
    kmeans = KMeans(n_clusters=2, init=centers, n_init=1, tol=1e-2)
    adapted_vol = np.argwhere(volume)

    # Performs kmeans clustering on the adapted volume
    kmeans.fit(adapted_vol)

    # Extract the data from the kmeans clustering
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Check lateralization condition
    is_correct_cluster_centers = check_lateralization_condition(cluster_centers)

    return cluster_centers, cluster_labels, is_correct_cluster_centers

def check_lateralization_condition(centers: np.ndarray) -> bool:
    """
    Check if the provided centers meet the lateralization condition.

    This function validates that:
        1. The left center does not cross to the right side compared to the right center along the z-axis.
        2. The x-axis variation between the centers is at least half the sum of the y and z variations.

    Parameters:
        centers (np.ndarray): A NumPy array containing exactly two center points. Each center point should be 
            an iterable with three numeric elements representing the x, y, and z coordinates respectively.
            The first element is interpreted as the left center and the second as the right center.

    Returns:
        bool: True if the centers satisfy the lateralization condition, otherwise False.
    """
    left_center, right_center = centers

    # It canot change sides
    if(left_center[2] > right_center[2]): return False

    # The x variation needs to be bigger than the y and z variations (combined)
    x_variation = abs(left_center[2] - right_center[2])
    y_variation = abs(left_center[1] - right_center[1])
    z_variation = abs(left_center[0] - right_center[0])
    if(x_variation > (y_variation + z_variation) * 0.5): return True

    return False

def generate_initial_centers(volume: np.ndarray, nr_centers: int = 20, range_val: int = 15) -> np.ndarray:
    """
    Generate initial centers for segmenting a 3D volume.
    This function computes the mean coordinates of all non-zero elements in the volume
    to determine a central point. It then creates two initial centers at the left and right
    extremes along the third dimension based on the computed mean. Additional centers are
    generated by adding random offsets to these initial points.

    Parameters:
        volume (numpy.ndarray): A 3D array representing the volume data.
        nr_centers (int, optional): The number of centers to generate (default is 20). Note that
            the algorithm starts with two centers and updates them iteratively with random offsets.

    Returns:
        numpy.ndarray: An array containing the generated center points. Each center is represented 
        by its 3D coordinates.
    """
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

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.1.3 Sub-subsection: Complex Separated Centroids - Destroying Bridge Approach
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def try_destroying_bridges(volume: np.ndarray, verbose: int) -> np.ndarray:
    """
    Attempts to destroy bridges in a volume using different methods to separate hemispheres.
    This function iteratively applies two different bridge-destruction techniques to the input
    volume: first using a Z-DIRECTED method and then a BALL method. For each method, it performs
    a loop with morphological opening until either the altered volume preserves less than 90% of
    the original similar volume or the hemispheres become separable. If the hemispheres are 
    successfully separated by either method, the function returns the resulting labeled array.
    If both techniques fail to achieve separation, the original volume is returned, potentially
    to be processed by a clustering algorithm later.

    Parameters:
        volume (np.ndarray): The 3D array representing the volume to process.
        verbose (int): An integer controlling the verbosity of the output during processing.
    Returns:
        np.ndarray: The labeled array with separated hemispheres if successful; otherwise, the
        original volume.
    """
    # Loop until either less than 90% of similar volume is kept or the hemispheres are separable (Z-DIRECTED VERSION)
    found_separation, labeled_array = loop_opening(volume, method='z_directed', verbose=verbose)
    if(found_separation): 
        if(verbose >= 8): print(f"                            🤓 Erosion with Z-DIRECTED method found separation! → Much Complex but Separable")
        return labeled_array

    # Loop until either less than 90% of similar volume is kept or the hemispheres are separable (BALL VERSION)
    found_separation, labeled_array = loop_opening(volume, method='ball', verbose=verbose)
    if(found_separation): 
        if(verbose >= 8): print(f"                            🤓 Erosion with BALL method found separation! → Much Complex but Separable")
        return labeled_array
        
    # If everything fails, return the volume as it is (need for clustering method)
    return volume

def loop_opening(volume, method: str, verbose: int) -> None:
    """
    Performs an iterative morphological opening on the input volume until a significant separation is detected, or until
    a maximum opening size is reached.
    The function applies morphological opening repeatedly on the given volume with an increasing kernel size (opening_size)
    until the computed similarity between the original volume and the eroded volume falls below a predefined threshold or
    until more than one relevant feature is detected.

    Parameters:
        volume: The input volume to be processed.
        method (str): The method or algorithm to be used in the morphological opening operation.
        verbose (int): Verbosity level, controlling the amount of runtime output (if applicable).
    Returns:
        tuple: A tuple containing
            - found_separation (bool): True if more than one relevant feature is detected indicating a separation, False otherwise.
            - labeled_array: The array output from the morphological opening function that labels identified features.
    """
    # Set threshold for similarity
    similarity_threshold = 90

    # Initialize variables
    eroded_volume, similarity_value = initiate_eroded_volume(volume)
    found_separation = False

    # Set the initial opening size based on the method
    if(method == 'z_directed'): opening_size = 2
    elif(method == 'ball'): opening_size = 1

    # Loop until either less than 90% of similar volume is kept or the hemispheres are separable
    while similarity_value >= similarity_threshold and opening_size <= 20 and not found_separation:
        eroded_volume, labeled_array, relevant_features = perform_morphological_opening(volume, opening_size, method)
        similarity_value = compute_volume_similarity(volume, eroded_volume)
        
        if(len(relevant_features) > 1): found_separation = True
        else: opening_size += 1
    
    return found_separation, labeled_array

def initiate_eroded_volume(volume: np.ndarray) -> tuple:
    """
    Initialize eroded volume from the provided volume.

    This function creates a copy of the input volume to represent an "eroded" version
    and assigns a constant similarity value of 100.

    Parameters:
        volume (np.ndarray): The input volume as a NumPy array.

    Returns:
        tuple: A tuple containing
            - np.ndarray: A copy of the input volume, representing the eroded volume.
            - int: The similarity value, which is set to 100.
    """
    eroded_volume = volume.copy()
    similarity_value = 100

    return eroded_volume, similarity_value

def extract_relevant_features_otsu(features: np.ndarray) -> np.ndarray:
    """
    Extract indices of features that exceed the Otsu threshold.

    This function computes the Otsu threshold for the given feature array and returns the indices
    of the elements that are greater than this threshold.

    Parameters:
        features (np.ndarray): A 1-dimensional array of feature values.

    Returns:
        np.ndarray: An array of indices corresponding to features that exceed the Otsu threshold.
    """
    thr = threshold_otsu(features)
    relevant_features = np.where(features > thr)[0]

    return relevant_features

def perform_morphological_opening(volume: np.ndarray, opening_size: int, method: str) -> np.ndarray:
    """
    Perform morphological opening and extract relevant features from a volumetric image.
    This function applies a morphological opening operation on a given volume using a
    specified structuring element defined by the 'method' parameter. After the opening,
    the function labels the connected components in the processed volume and extracts
    the features that pass Otsu's threshold.

    Parameters:
        volume (np.ndarray): Input 3D array representing the volume to be processed.
        opening_size (int): Size or radius parameter used to generate the structuring element.
        method (str): Technique to create the structuring element. Options are
            - 'ball': Uses a ball-shaped structuring element.
            - 'z_directed': Uses a structuring element with ones in the first dimension.
    Returns:
        tuple
            - np.ndarray: The volume after morphological opening.
            - np.ndarray: Array with labeled connected components.
            - np.ndarray: Array containing the relevant features extracted based on Otsu's threshold.
    """
    # Define the structuring element based on the method
    if(method == 'ball'): selem = ball(opening_size)
    elif(method == 'z_directed'): selem = np.ones((opening_size, 1, 1), dtype=bool)

    # Perform morphological opening
    eroded_volume = opening(volume, selem)

    if(np.count_nonzero(eroded_volume) == 0):
        return eroded_volume, eroded_volume, np.array([])

    # Perform labeling
    labeled_array, num_features = label(eroded_volume, structure=None)
    labeled_array, features = reorder_labels_array(labeled_array)
    
    # Extract only the relevant features based on Otsu's threshold
    relevant_features = extract_relevant_features_otsu(features)
    
    return eroded_volume, labeled_array, relevant_features

def reorder_labels_array(labeled_array: np.ndarray) -> tuple:
    """
    Reorder the labels in a labeled array based on their size.

    This function takes a numpy array with integer labels, where 0 represents the
    background, and reassigns label values so that the region with the largest size
    is given label 1, the second largest is given label 2, and so on.
    The original labels are mapped to new labels accordingly, and the background (0)
    is preserved.

    Parameters:
        labeled_array (np.ndarray): A numpy array of integer labels. The background should be denoted by 0.

    Returns:
        tuple: A tuple containing
            - sorted_labels (np.ndarray): The array with the updated labels after reordering.
            - sizes (np.ndarray): An array of counts corresponding to the sizes of the new labels,
              excluding the background.
    """
    # Extract sizes without background count
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0

    # Get the sorted positions
    sorted_old_labels = np.argsort(sizes)[::-1]

    # Remap the labels by size starting in voxel 1
    new_label_map = np.zeros_like(sizes, dtype=int)
    for new_lbl, old_lbl in enumerate(sorted_old_labels[:-1], start=1):
        new_label_map[old_lbl] = new_lbl

    # Assign to variables
    sorted_labels = new_label_map[labeled_array]
    sizes = np.bincount(sorted_labels.ravel())[1:]

    return sorted_labels, sizes

def compute_volume_similarity(original_volume: np.ndarray, comparing_volume: np.ndarray) -> float:
    """
    Compute the similarity between two binary volumes.

    This function compares two binary numpy arrays (volumes), where each volume is expected 
    to contain only 0s and 1s. It computes the similarity percentage by first ensuring 
    that both volumes are proper binary masks. The comparing volume is preprocessed by 
    inverting its background (0 values replaced with 100) so that background values differ 
    from foreground values (1). The similarity mask is generated by comparing the original 
    volume against this preprocessed comparing volume, and the percentage similarity is 
    calculated based on the number of matching foreground elements.

    Parameters:
        original_volume (np.ndarray): A binary numpy array representing the original volume.
        comparing_volume (np.ndarray): A binary numpy array representing the volume to compare.

    Returns:
        float: The similarity percentage between the original and comparing volumes, rounded 
        to two decimal places.

    Raises:
        AssertionError: If either input array does not consist solely of binary values (0s and 1s).
    """
    # Make sure both inputs only have 1s and 0s (assert)
    assert_binary_mask(original_volume, comparing_volume)

    # Prepares the comparing vector for subsequent comparison (makes background different)
    comparing_inverted = np.where(comparing_volume == 0, 100, comparing_volume)

    # Compute the similarity mask by comparing the original volume with the inverted comparing volume
    similarity_mask = (comparing_inverted == original_volume)
    similarity_mask = np.where(similarity_mask, 1, 0)

    # Compute similarity in percentage (e.g. 98.23%)
    volume_similarity = np.round(np.count_nonzero(similarity_mask)/np.count_nonzero(original_volume)*100,2)

    return volume_similarity

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.1.4 Sub-subsection: Hemisphere Handling
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def assign_side(hemispheres, centers):
    first_hemisphere, second_hemisphere = hemispheres
    first_center, second_center = centers

    if first_center[2] > second_center[2]:
        left_hemisphere = first_hemisphere
        right_hemisphere = second_hemisphere
        left_center = first_center
        right_center = second_center
    else:
        left_hemisphere = second_hemisphere
        right_hemisphere = first_hemisphere
        left_center = second_center
        right_center = first_center

    return (left_hemisphere, right_hemisphere), (left_center, right_center)

def rebuild_hemispheres(labeled_array: np.ndarray, verbose: int):
    first_hemisphere = np.where(labeled_array == 1, 1, 0)
    second_hemisphere = np.where(labeled_array == 2, 1, 0)

    first_center = np.mean(np.argwhere(first_hemisphere), axis=0)
    second_center = np.mean(np.argwhere(second_hemisphere), axis=0)

    # Assign the left most hemisphere as the left hemisphere (it is mirrored)
    (left_hemisphere, right_hemisphere), (left_center, right_center) = assign_side((first_hemisphere, second_hemisphere), (first_center, second_center))

    for i in range(3, np.max(labeled_array) + 1):
        if(verbose >= 10): print(f"                                    → Processing piece {i}...")
        piece = np.where(labeled_array == i, 1, 0)
        piece_center = np.mean(np.argwhere(piece), axis=0)

        if np.linalg.norm(piece_center - left_center) < np.linalg.norm(piece_center - right_center):
            # Update the left hemisphere center
            left_hemisphere += piece
            left_center = np.mean(np.argwhere(left_hemisphere), axis=0)
        else:
            # Update the right hemisphere
            right_hemisphere += piece
            right_center = np.mean(np.argwhere(right_hemisphere), axis=0)

    left_hemisphere = np.where(left_hemisphere > 0, 1, 0)
    right_hemisphere = np.where(right_hemisphere > 0, 1, 0)

    return left_hemisphere, right_hemisphere

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.1.5 Sub-subsection: Alerts and Warnings
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def assert_binary_mask(v1: np.ndarray, v2: np.ndarray) -> None:
    """
    Assert that both input numpy arrays are binary masks containing only 0s and 1s.

    Parameters:
        v1 (np.ndarray): First numpy array expected to contain only 0s and 1s.
        v2 (np.ndarray): Second numpy array expected to contain only 0s and 1s.

    Raises:
        ValueError: If either v1 or v2 contains elements other than 0 and 1.
    """
    if not (np.all(np.isin(v1, [0, 1])) and np.all(np.isin(v2, [0, 1]))):
        raise ValueError("Both original_volume and comparing_volume must contain only 0s and 1s.")



# ──────────────────────────────────────────────────────
# 4.2 Subsection: Each Segement Center Coordinates Extraction - Convertion
# ──────────────────────────────────────────────────────
def convert_to_ref(old_coords: np.ndarray, reference: np.ndarray, voxel_size: float, volumes_sizes: tuple, verbose: int) -> tuple:
    """
    Convert coordinates to the bregma-lambda plane and adjust for voxel size.

    Parameters:
        old_coords (numpy.ndarray): An Nx3 array of coordinates to be converted.
        reference (tuple): A tuple containing the bregma and lambda coordinates.
        voxel_size (float): The size of a voxel in micrometers.

    Returns:
        numpy.ndarray: An Nx3 array of converted coordinates in micrometers.

    Notes:
        - The function inverts the x-axis and y-axis for all coordinates.
        - It checks if any z-coordinate is not negative and prints a warning if any are positive.
        - The coordinates are converted to micrometers and rounded to three decimal places.
    """
    if(verbose >= 6): print("                    Converting the Coordinates to BL Space")

    # Bring to bregma-lambda plane
    new_coords = np.array(old_coords) - np.array(reference)[0]
    voxel_size = np.array(voxel_size)

    # Check if the convertion made sense
    alert_inconsistent_convertion(old_coords, new_coords, mode='voxel')

    if(verbose >= 7): 
        print(f"                        → Centroid Voxel Coordinates: {old_coords}")
        print(f"                        → Bregma-Lambda Coordinates: {reference}")
        print(f"                        → Centroid Voxel Coordinates in BL Space (No XY Invertion): {new_coords}")

    # Invert the x-axis and y-axis
    new_coords[:, 2] = new_coords[:, 2] * -1
    new_coords[:, 1] = new_coords[:, 1] * -1

    if(verbose >= 7): print(f"                        → Centroid Voxel Coordinates in BL Space (After XY Invertion): {new_coords}")

    # Check if any z is not negative
    alert_non_negative_z(old_coords, new_coords, reference)

    # Convert to um instead of voxels
    alert_not_isotropic_voxel(voxel_size)

    # Convert to um instead of voxels
    new_coords = new_coords * voxel_size
    new_coords = np.round(new_coords, 3)

    alert_inconsistent_convertion(old_coords, new_coords, mode='um')

    # Compute the volumes in um
    voxel_volume = np.prod(voxel_size)
    volumes_um = np.round(np.array(volumes_sizes) * voxel_volume, 5)

    if(verbose >= 7): print(f"                        → Centroid um Coordinates in BL Space: {new_coords}")
    return new_coords, volumes_um

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.2.1 Sub-subsection: Alert and Warnings
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def alert_inconsistent_convertion(old_coords: np.ndarray, new_coords: np.ndarray, mode: str):
    # Initiate the conditions
    have_old_equal_coord = (old_coords[0] == old_coords[1]).any()
    have_new_equal_coords = (new_coords[0] == new_coords[1]).any()

    # Warn if they are met
    if(have_old_equal_coord and not have_new_equal_coords):
        if(mode == 'voxel'): warnings.warn("WARNING: The conversion between the coordinates and the reference is not consistent (Voxel-Voxel Convertion).")
        elif(mode == 'um'): warnings.warn("WARNING: The conversion between the coordinates and the reference is not consistent (Voxel-um Convertion).")
        else: warnings.warn("WARNING: The conversion between the coordinates and the reference is not consistent")

def alert_non_negative_z(old_coords: np.ndarray, new_coords: np.ndarray, reference: np.ndarray):
    # Initiate the condition
    is_z_not_negative = np.any(new_coords[:, 0] > 0)

    # Warn if they are met
    if is_z_not_negative: 
        warnings.warn(f'WARNING: Some Z values are not negative')
        warnings.warn(f'Voxel Centroid Coords (Before Any Change) - {old_coords}')
        warnings.warn(f'Begma-lambda Coords - {reference}')
        warnings.warn(f'Converted Centroid Coords - {new_coords}')

def alert_not_isotropic_voxel(voxel_size: np.ndarray):
    # Initiate the condition
    is_isotropic = (voxel_size[0] == voxel_size[1]) and (voxel_size[0] == voxel_size[2])

    # Warn if they are met
    if(not is_isotropic): 
        warnings.warn("WARNING: Voxel size is not isotropic, this may cause issues in the conversion to um coordinates.")



# ──────────────────────────────────────────────────────
# 4.3 Subsection: Each Segement Center Coordinates Extraction - Statistics
# ──────────────────────────────────────────────────────
def extract_statistics(ref_centroids: np.array, verbose: int) -> tuple:
    """
    Computes statistical measures (mean, standard deviation, and standard error) 
    for a pair of reference centroids in micrometer (um) coordinates.

    Parameters:
        ref_centroids (np.array): A 2D numpy array containing two centroids 
                                  (left and right) as rows, where each centroid 
                                  is represented by a 3D coordinate (x, y, z).

    Returns:
        tuple: A tuple containing:
            - mean_um (np.array): The mean of the centroids in um coordinates.
            - std_um (np.array): The standard deviation of the centroids in um coordinates.
            - ste_um (np.array): The standard error of the centroids in um coordinates.
    """
    if(verbose >= 6): print("                    Extracting the Centroid Statistics")
    # Unpack the centroids
    left_ref_centroid, right_ref_centroid = ref_centroids

    # Prepares the um coordinates to have some statistical analysis
    if((right_ref_centroid == left_ref_centroid).all()):
        right_ref_centroid_mean = right_ref_centroid.copy()
    else:
        right_ref_centroid_mean = right_ref_centroid.copy()
        right_ref_centroid_mean[2] = right_ref_centroid_mean[2] * -1
    centroids_um = np.stack([left_ref_centroid, right_ref_centroid_mean], axis=0)

    # Compute the mean, standard error and standard deviation of um coordinates
    mean_um = np.mean(centroids_um, axis=0)
    std_um = np.std(centroids_um, axis=0, ddof=1)
    ste_um = std_um / np.sqrt(centroids_um.shape[0])

    return mean_um, std_um, ste_um

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 4.3.1 Sub-subsection: Format Data
# ›››››››››››››››››››››››››››››››››››››››››››››››››
def reorder_coords(coords: np.array, rounding: int):
    """
    Reorders the coordinates by swapping their positions and rounds them to the specified precision.

    Parameters:
        coords (np.array): A NumPy array containing the coordinates to be reordered.
        rounding (int): The number of decimal places to round the reordered coordinates.

    Returns:
        list: A list of reordered and rounded coordinates.
    """
    return str([float(x) for x in np.round(coords[[2, 1, 0]], rounding)])