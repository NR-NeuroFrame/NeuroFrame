# ================================================================
# 0. Section: Imports
# ================================================================
import cv2

import numpy as np
import SimpleITK as sitk

from ..utils import get_z_coord
from ..logger import logger
from ..registrator import Registrator, SUTURE_REGISTRATOR, convert_input, apply_shape
from ..mouse import Mouse


# ──────────────────────────────────────────────────────
# 0.1 Subsection: Universal Constants
# ──────────────────────────────────────────────────────
SUTURE_TEMPLATE = cv2.imread("src/neuroframe/templates/suture_template_t14.png", cv2.IMREAD_GRAYSCALE)
BREGMA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/bregma_template_t14.png", cv2.IMREAD_GRAYSCALE))
LAMBDA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/lambda_template_t14.png", cv2.IMREAD_GRAYSCALE))
REF_TEMPLATES = (BREGMA_TEMPLATE, LAMBDA_TEMPLATE)



# ================================================================
# 1. Section: Extract Bregma and Lambda Points
# ================================================================
def get_bregma_lambda(mouse: Mouse, skull_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the 3D coordinates of bregma and lambda on a mouse skull.
    
    This function identifies the bregma and lambda landmarks by applying a
    template matching approach. It first computes a deformation map from the
    provided skull surface. Then, it aligns reference templates for bregma and
    lambda to a specific slice of the mouse's segmentation data. Using these
    aligned templates and the deformation map, it finds the 2D (y, x) coordinates
    of the landmarks on the skull surface. Finally, it determines the corresponding
    z-coordinate from the micro-CT data to produce the full 3D coordinates.
    
    Parameters
    ----------
    mouse : Mouse
        A custom object containing mouse-specific data, including segmentation
        volumes (`mouse.segmentation.volume`) and micro-CT data
        (`mouse.micro_ct.data`).
    skull_surface : np.ndarray
        A NumPy array representing the skull surface, used to derive the
        deformation map and locate the landmarks.

    Returns
    -------
    tuple[tuple[int, int, int], tuple[int, int, int]]
        A tuple containing two elements:
        - The 3D coordinates of bregma as a tuple (z, y, x).
        - The 3D coordinates of lambda as a tuple (z, y, x).

    Side Effects
    ------------
    Logs the following information using the configured `logger`:
    - The calculated 3D coordinates for bregma and lambda.
    - The deviation of bregma and lambda from a reference, in millimeters.
    - The angle of deviation, in degrees.

    Notes
    -----
    - The function relies on a global constant `REF_TEMPLATES` which must be
      defined and contain the bregma and lambda templates.
    - It assumes the `mouse` object has `segmentation.volume` and
      `micro_ct.data` attributes.
    - A hardcoded slice index `100` from `mouse.segmentation.volume` is used
      as the reference slide for template alignment. This may not be
      appropriate for all datasets.
    - The returned coordinates are in the (z, y, x) order.
    - The accuracy of the results depends on the quality of the input data
      and the correctness of the helper functions (`extract_deformation_map`,
      `get_reference_point`, `get_z_coord`, `compute_deviation`).

    Examples
    --------
    >>> import numpy as np
    >>> from unittest.mock import MagicMock
    >>> # This is a conceptual example, as 'Mouse' and helper functions
    >>> # are specific to the NeuroFrame package.
    >>> # Mock the Mouse object and its attributes
    >>> mock_mouse = MagicMock()
    >>> mock_mouse.segmentation.volume = np.zeros((200, 512, 512))
    >>> mock_mouse.micro_ct.data = np.zeros((200, 512, 512))
    >>> # Mock a skull surface
    >>> mock_skull_surface = np.random.rand(512, 512)
    >>> # In a real scenario, you would call the function like this:
    >>> # bregma_coords, lambda_coords = get_bregma_lambda(mock_mouse, mock_skull_surface)
    >>> # The function would then log and return the coordinates, e.g.:
    >>> # bregma_coords = (110, 250, 260)
    >>> # lambda_coords = (105, 400, 255)
    """
    
    transform = extract_deformation_map(skull_surface)

    # Unpacks the templates
    bregma_template, lambda_template = REF_TEMPLATES
    reference_slide = convert_input(mouse.segmentation.volume[100,:,:])
    bregma_template = apply_shape(reference_slide, bregma_template)
    lambda_template = apply_shape(reference_slide, lambda_template)

    # Get the bregma and lambda coordinates (x, y)
    bregma_coords = np.round(get_reference_point(bregma_template, skull_surface, transform)).astype(int)
    lambda_coords = np.round(get_reference_point(lambda_template, skull_surface, transform)).astype(int)
    
    # Get the z coordinates
    bregma_z = get_z_coord(mouse.micro_ct.data, bregma_coords)
    lambda_z = get_z_coord(mouse.micro_ct.data, lambda_coords)

    # Get the coordinates (z, y, x)
    bregma_coords = (bregma_z, bregma_coords[0], bregma_coords[1])
    lambda_coords = (lambda_z, lambda_coords[0], lambda_coords[1])

    # Log the coordinates
    logger.info(f"Bregma Coordinates: {bregma_coords} (z, y, x)")
    logger.info(f"Lambda Coordinates: {lambda_coords} (z, y, x)")

    # Log the deviations and angle
    deviations, angle = compute_deviation(mouse, (bregma_coords, lambda_coords))
    logger.info(f"Deviation Bregma {deviations[0].round(1)} mm")
    logger.info(f"Deviation Lambda {deviations[1].round(1)} mm")
    logger.info(f"Angle: {angle.round(2)} degrees")

    return bregma_coords, lambda_coords


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Deformation Map Extraction
# ──────────────────────────────────────────────────────
def extract_deformation_map(skull_surface: np.ndarray, sutures_registration: Registrator = SUTURE_REGISTRATOR) -> sitk.Transform:    
    # Bspline registration to the suture template
    _, sutures_transform = sutures_registration.register(skull_surface, SUTURE_TEMPLATE)

    logger.detail(f"Obtained Transform Parameters: {sutures_transform.GetParameters()}")

    return sutures_transform


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Apply Deformation Map to Reference Templates
# ──────────────────────────────────────────────────────
def get_reference_point(reference_template: np.ndarray | sitk.Image, skull_surface: np.ndarray | sitk.Image, transform: sitk.Transform) -> np.ndarray:
    # Apply the same transformation to the reference template that was used to get the template
    resampler = Registrator(res_interpolator = 'nearest')
    deformed_template = resampler.resample(convert_input(skull_surface), convert_input(reference_template), transform)
    deformed_template = sitk.GetArrayFromImage(deformed_template)

    # Get the coordinates of the reference points
    points = np.argwhere(deformed_template > 0)
    reference_coords = np.mean(points, axis=0)
    
    return reference_coords


# ──────────────────────────────────────────────────────
# 1.3 Subsection: Get Deviations and Angle
# ──────────────────────────────────────────────────────
def compute_deviation(mouse: Mouse, coords: tuple) -> tuple[np.ndarray, float]:

    # Unpack coordinates
    bregma, lambda_ = coords
    midline = np.array(mouse.data_shape) // 2
    voxel_size = mouse.voxel_size

    # Get the x coordinates
    midline_x = midline[2]
    bregma_x = bregma[2]
    lambda_x = lambda_[2]

    # Compute deviations in mm
    deviation_bregma = (midline_x - bregma_x) * voxel_size[2]
    deviation_lambda = (midline_x - lambda_x) * voxel_size[2]

    # Get the vector for angle calculation
    vector = np.array(bregma) - np.array(lambda_)
    vector = np.array([vector[2], vector[1]])
    vector = vector / np.linalg.norm(vector)
    reference = np.array([0,1])

    # Calculate the angle between the vector and the reference
    angle = np.arccos(np.clip(np.dot(vector, reference), -1.0, 1.0))
    angle = 180 - np.degrees(angle)
    deviations = np.array([deviation_bregma, deviation_lambda])

    return deviations, angle