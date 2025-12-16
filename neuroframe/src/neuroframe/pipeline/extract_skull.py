# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from scipy.ndimage import gaussian_filter

from ..mouse import Mouse
from ..logger import logger



# ================================================================
# 1. Section: Extract Skull
# ================================================================
def extract_skull(mouse: Mouse, method:str = 'cumsum') -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract a 2D skull projection map from 3D micro-CT data.

    This function applies a specified projection method to the micro-CT data
    of a mouse to generate a 2D representation of the skull.

    Parameters
    ----------
    mouse : Mouse
        An object representing the mouse, which must contain a `micro_ct`
        attribute holding the 3D scan data as a NumPy array.
    method : str, optional
        The projection method to use. Valid options are 'mean', 'view', and
        'cumsum'. If an invalid method is provided, it defaults to 'cumsum'.
        Default is 'cumsum'.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        - If `method` is 'view', returns a tuple containing the projection map
          and the depth map, both as NumPy arrays.
        - For all other methods, returns the projection map as a single
          NumPy array.

    Raises
    ------
    AttributeError
        If `mouse` does not have a `micro_ct.data` attribute.

    Side Effects
    ------------
    Logs a warning message if an invalid `method` is specified, indicating
    that the function will default to the 'cumsum' method.

    Notes
    -----
    The 'cumsum' method uses hardcoded parameters `[30, 20, 4000]`. The
    behavior of the underlying projection functions (`mean_projection`,
    `view_projection`, `cumsum_projection`) is not detailed here.

    Examples
    --------
    >>> import numpy as np
    >>> from unittest.mock import MagicMock
    >>> # Create a mock Mouse object with mock micro_ct data
    >>> mock_mouse = MagicMock()
    >>> mock_mouse.micro_ct.data = np.random.rand(100, 100, 100)

    >>> # Assume the existence of the projection helper functions
    >>> # For demonstration, we'll mock them to return predictable shapes
    >>> mean_projection = lambda x: np.mean(x, axis=0)
    >>> view_projection = lambda x: (np.max(x, axis=0), np.argmax(x, axis=0))
    >>> cumsum_projection = lambda x, params: np.sum(x, axis=0)

    >>> # Example with the default 'cumsum' method
    >>> projection_map = extract_skull(mock_mouse, method='cumsum')
    >>> print(projection_map.shape)
    (100, 100)

    >>> # Example with the 'view' method, returning two maps
    >>> projection_map, depth_map = extract_skull(mock_mouse, method='view')
    >>> print(projection_map.shape)
    (100, 100)
    >>> print(depth_map.shape)
    (100, 100)

    >>> # Example with an invalid method, which defaults to 'cumsum'
    >>> projection_map = extract_skull(mock_mouse, method='invalid_method')
    # A warning would be logged here.
    >>> print(projection_map.shape)
    (100, 100)
    """
    
    # Extracts the data from the mice object
    micro_ct = mouse.micro_ct.data
    depth_map = None

    # Applies the selected projection method
    if method == 'mean': projection_map = mean_projection(micro_ct)
    elif method == 'view': projection_map, depth_map = view_projection(micro_ct)
    elif method == 'cumsum': projection_map = cumsum_projection(micro_ct, [30, 20, 4000])
    else: 
        logger.warning("Invalid method. Choose 'mean' , 'view' or 'cumsum'. Will default to 'cumsum'.")
        projection_map = cumsum_projection(micro_ct, [30, 20, 4000])

    # Returns the projection map and depth map if method is 'view'
    if(method == 'view'): return projection_map, depth_map
    return projection_map


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Cumsum Projection Method
# ──────────────────────────────────────────────────────
def cumsum_projection(micro_ct: np.ndarray, parameters: tuple, lower_bound: None | int = None) -> np.ndarray:
    # Unpack the parameters
    margin, skull_threshold, depth_sigma = parameters

    z_len = micro_ct.shape[0]
    #lower_bound = 130 # Need to fix this
    if(lower_bound is None): lower_bound = int(z_len*0.65)
    selected_slices = micro_ct[lower_bound:,:,:]

    # Threshold the selected slices
    skull_mask = selected_slices > skull_threshold

    # Get the cumulative sum of the selected slices (each slice will have a value that defines the depth)
    cumulative_slices = np.cumsum(skull_mask, axis=0)

    # Get the background of these slices (everything that is not a mice)
    background = cumulative_slices < 1

    # Get the depth image and a gaussian smoothed version of it
    img_depth = np.sum(background, axis = 0).astype(float)
    gauss_depth = gaussian_filter(img_depth, depth_sigma).astype(int)

    # Extract surface of the skull by sampling along the z-axis using the gaussian smoothed depth image 
    # (we want the gaussian to vary smootly along the x and y axis -> adjust sigma for that)
    try:
        width, height = micro_ct.shape[1:3]
        pre_surface = np.fromfunction(lambda z,x,y: micro_ct[gauss_depth[x,y] + lower_bound + z - margin, x, y], (margin*2+1, width, height), dtype=int)
    except IndexError:
        logger.exception("The selected slices exceed the bounds of the micro-CT image. Trying with a different lower bound.")
        return cumsum_projection(micro_ct, parameters, lower_bound=(z_len*3)//5)

    # Compute the max projection
    skull_surface = np.max(pre_surface,axis=0).astype(float)

    skull_surface = np.where(skull_surface < 52, 0, skull_surface)

    return skull_surface
    

# ──────────────────────────────────────────────────────
# 1.2 Subsection: Mean Projection Method
# ──────────────────────────────────────────────────────
# BUG: For some reason it plots empty array, even though the array seems
# to be populated, since we don't use this method, for now is ok 
def mean_projection(micro_ct: np.ndarray) -> np.ndarray:
    z_len = micro_ct.shape[0]
    lower_bound = 3*int(z_len//4)
    average_skull = np.mean(micro_ct[lower_bound:,:,:], axis=0)

    # Logging information about the resulting skull
    if(np.sum(average_skull) < 1): logger.warning("Mean projection resulted in an empty skull. Check micro-CT data and lower bound.")
    logger.debug(f"Mean projection resulted in a skull with mean intensity of {np.mean(average_skull):.2f}.")
    logger.debug(f"Mean projection resulted in a skull with unique intensity of {np.unique(average_skull)}.")
    logger.debug(f"Mean projection resulted in a skull with shape {average_skull.shape}.")

    return average_skull


# ──────────────────────────────────────────────────────
# 1.3 Subsection: View Projection Method
# ──────────────────────────────────────────────────────
def view_projection(micro_ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Unpack the shape of the micro-CT image
    x_len = micro_ct.shape[2]
    y_len = micro_ct.shape[1]

    # Initialize threshold if not provided
    threshold = auto_thr_projection(micro_ct)

    # Initialize projection and depth maps
    projection_map = np.zeros((y_len, x_len))
    depth_map = np.zeros((y_len, x_len))

    # Vectorized approach to avoid nested for loops
    points_above_threshold = micro_ct > threshold
    last_intensity_indices = np.argmax(points_above_threshold[::-1, :, :], axis=0)
    last_intensity_indices = micro_ct.shape[0] - 1 - last_intensity_indices

    # Mask for pixels where no intensity is above the threshold
    no_intensity_mask = ~points_above_threshold.any(axis=0)

    # Get the intensity and depth values
    projection_map = micro_ct[last_intensity_indices, np.arange(y_len)[:, None], np.arange(x_len)]
    depth_map = last_intensity_indices

    # Apply the mask to set values to 0 where no intensity is above the threshold
    projection_map[no_intensity_mask] = 0
    depth_map[no_intensity_mask] = 0

    return projection_map, depth_map

# ››››››››››››››››››››››››››››››››››››››››››››››››
# 1.1.1 Sub-subsection: Auto Thresholding Function
# ››››››››››››››››››››››››››››››››››››››››››››››››
def auto_thr_projection(micro_ct: np.ndarray) -> float:
    # Gets the data from middle of the image
    middle_point = np.array(micro_ct.shape) // 2
    points = micro_ct[:, middle_point[1], middle_point[2]]
    gradient = np.gradient(points)

    # Extract the threshold from peak of the gradient
    depth = np.argmax(gradient)
    threshold = points[depth]

    logger.info(f"Auto-thresholding at: {threshold}")

    return threshold