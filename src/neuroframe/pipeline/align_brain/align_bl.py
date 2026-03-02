# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from ...mouse import Mouse
from ...utils import compute_separation, rotate_mice, transform_points, xy_fine_tune, logg_separation



# ================================================================
# 1. Section: Put the Mouse in the Bregma-Lambda Orientation
# ================================================================
def align_to_bl(mouse: Mouse, bregma_coords: np.array, lambda_coords: np.array, deviation: int = 5) -> tuple[np.array, np.array]:
    """Aligns the mouse brain data to the Bregma-Lambda axis.

    This function rotates the mouse's MRI and segmentation data so that the
    vector defined by the bregma and lambda landmarks aligns with the y-axis
    of the coordinate system. It also performs an optional fine-tuning step
    to adjust the alignment in the XY plane.

    Parameters
    ----------
    mouse : Mouse
        The mouse object containing the MRI and segmentation data to be aligned.
    bregma_coords : np.array
        The 3D coordinates of the bregma landmark.
    lambda_coords : np.array
        The 3D coordinates of the lambda landmark.
    deviation : int, optional
        The maximum deviation in pixels for the fine-tuning alignment in the
        XY plane. If set to 0, fine-tuning is skipped. Default is 5.

    Returns
    -------
    tuple[np.array, np.array]
        A tuple containing the updated integer coordinates of bregma and lambda
        after the alignment and transformation.

    Side Effects
    ------------
    - Modifies the `mouse` object in place by rotating its `mri.volume` and
      `segmentation.volume` attributes.
    - Logs the separation of brain hemispheres before and after alignment
      using the `logg_separation` function.

    Notes
    -----
    The function first performs a coarse alignment by rotating the entire
    volume. If `deviation` is greater than 0, a subsequent fine-tuning
    step (`bl_fine_tune`) is performed. The input `mouse` object is mutated
    during this process.

    Examples
    --------
    >>> import numpy as np
    >>> from unittest.mock import MagicMock
    >>> # Create a mock Mouse object for demonstration
    >>> mock_mouse = MagicMock()
    >>> mock_mouse.data_shape = (100, 100, 100)
    >>> mock_mouse.segmentation.volume = np.zeros((100, 100, 100))
    >>> bregma = np.array([50, 40, 50])
    >>> lambd = np.array([50, 60, 50])
    >>> # This is a conceptual example; real usage requires a valid Mouse object
    >>> # and dependencies. The function would be called like this:
    >>> # new_bregma, new_lambda = align_to_bl(mock_mouse, bregma, lambd, deviation=5)
    >>> # assert new_bregma is not None
    >>> # assert new_lambda is not None
    """

    # Center and rotate mice according to the Bregma and Lambda coordinates
    mri_shape = mouse.data_shape

    # Get the current separation (amount of brain in each side of the midline in %)
    previous_t = logg_separation(mouse.segmentation.volume, "start")

    # Rotate the mice according to the Bregma and Lambda coordinates
    bl_vector = np.array(lambda_coords) - np.array(bregma_coords)
    norm = np.linalg.norm(bl_vector)
    bl_vector = bl_vector / norm
    rotation_matrix, offset = rotate_mice(mouse, bl_vector, [0, 1, 0])
    bregma_coords = np.round(transform_points(bregma_coords, mri_shape, rotation_matrix, offset)).astype(int)
    lambda_coords = np.round(transform_points(lambda_coords, mri_shape, rotation_matrix, offset)).astype(int)

    # Compute the new separation
    previous_t = logg_separation(mouse.segmentation.volume, "after BL alignment", previous_t)

    # Fine tune the alignment in the XY plane
    if(deviation > 0): bl_fine_tune(mouse, bregma_coords, lambda_coords, deviation)

    return np.round(bregma_coords).astype(int), np.array(lambda_coords).astype(int)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Bregma-Lambda Fine Tuning
# ──────────────────────────────────────────────────────
def bl_fine_tune(mouse: Mouse, bregma_coords: np.array, lambda_coords: np.array, deviation: int) -> tuple[np.array, np.array]:
    # Extract needed data
    mri_shape = mouse.data_shape
    previous_t = compute_separation(mouse.segmentation.volume)

    # Fine tune the alignment in the XY plane
    align_matrix, align_offset = xy_fine_tune(mouse, bregma_coords, deviation)
    bregma_coords = np.round(transform_points(bregma_coords, mri_shape, align_matrix, align_offset)).astype(int)
    lambda_coords = np.round(transform_points(lambda_coords, mri_shape, align_matrix, align_offset)).astype(int)

    # Compute the new separation
    _ = logg_separation(mouse.segmentation.volume, "after BL fine-tuning", previous_t)

    return bregma_coords, lambda_coords
