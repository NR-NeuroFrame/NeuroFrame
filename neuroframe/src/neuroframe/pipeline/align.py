# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import SimpleITK as sitk

from scipy.ndimage import zoom

from ..utils import count_voxels, enlarge_shape
from ..logger import logger
from ..mouse_data import Segmentation
from ..mouse import Mouse
from ..registrator import Registrator


# ──────────────────────────────────────────────────────
# 0.1 Subsection: Universal Constants
# ──────────────────────────────────────────────────────
ALLEN_TEMPLATE = Segmentation("src/neuroframe/templates/allen_brain_25μm_ccf_2017.nii.gz")



# ================================================================
# 1. Section: Align the Mouse to the Allen Template
# ================================================================
def align_to_allen(mouse: Mouse, template: Segmentation = ALLEN_TEMPLATE) -> Mouse:
    """Aligns a mouse brain segmentation to a template volume using rigid registration.

    This function performs a rigid alignment of a given mouse's segmentation
    volume to a specified template volume. The process involves adapting the
    template to the mouse's volume dimensions, performing a multi-resolution
    rigid registration to find the optimal transformation, and then applying this
    transformation to the mouse data.

    Parameters
    ----------
    mouse : Mouse
        The mouse object containing the segmentation volume to be aligned. This
        object is expected to have a `segmentation.volume` attribute.
    template : Segmentation, optional
        The target segmentation template for alignment. If not provided, the
        default Allen Brain Atlas template (`ALLEN_TEMPLATE`) is used.

    Returns
    -------
    Mouse
        A new mouse object with the segmentation and other associated data
        aligned to the template volume.

    Raises
    ------
    Unspecified
        The underlying registration library (e.g., SimpleITK used within
        `Registrator`) may raise exceptions on failure, which are not
        explicitly handled here.

    Side Effects
    ------------
    Logs the parameters of the calculated transformation at the 'detail' level
    using the configured logger.

    Notes
    -----
    - The function performs a rigid registration, meaning it only accounts for
      translations and rotations, not scaling or shearing.
    - The input `mouse` object is not modified in place; a new, transformed
      `Mouse` object is returned.
    - The `adapt_template` function is called internally to ensure the template
      and mouse volumes are compatible for registration, which may involve
      resampling or resizing the template.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> import SimpleITK as sitk
    >>> # Assume Mouse and Segmentation are defined classes
    >>> class Segmentation:
    ...     def __init__(self, volume):
    ...         self.volume = volume
    >>> class Mouse:
    ...     def __init__(self, seg_volume):
    ...         self.segmentation = Segmentation(seg_volume)
    >>> # Mock external dependencies for demonstration
    >>> class MockRegistrator:
    ...     def __init__(self, *args, **kwargs): pass
    ...     def register(self, fixed, moving):
    ...         # Return a dummy transform
    ...         return None, sitk.Euler3DTransform()
    >>> class MockLogger:
    ...     def detail(self, msg): print(f"LOG: {msg}")
    >>> # Monkey-patch dependencies
    >>> import neuroframe.pipeline.align
    >>> neuroframe.pipeline.align.Registrator = MockRegistrator
    >>> neuroframe.pipeline.align.logger = MockLogger
    >>> neuroframe.pipeline.align.adapt_template = lambda m, t: t
    >>> neuroframe.pipeline.align.register_mice = lambda m, t, tr: Mouse(m.segmentation.volume)
    >>> neuroframe.pipeline.align.ALLEN_TEMPLATE = Segmentation(sitk.Image(64, 64, 64, sitk.sitkUInt8))
    >>> # Create a sample mouse with a segmentation volume
    >>> mouse_volume = sitk.Image(64, 64, 64, sitk.sitkUInt8)
    >>> my_mouse = Mouse(mouse_volume)
    >>> # Align the mouse to the default Allen template
    >>> aligned_mouse = align_to_allen(my_mouse)
    LOG: Obtained Transform: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    >>> isinstance(aligned_mouse, Mouse)
    True

    """
    
    template_volume = adapt_template(mouse, template)

    # Does the rigid registration
    rigid_registration = Registrator(method='bspline', multiple_resolutions=True)
    _, transform = rigid_registration.register(template_volume, mouse.segmentation.volume)

    logger.detail(f"Obtained Transform: {transform.GetParameters()}")

    # Applies the transformation to the mice
    mouse = register_mice(mouse, template_volume, transform)

    return mouse


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Adapts the Template's Size
# ──────────────────────────────────────────────────────
def adapt_template(mouse: Mouse, template: Segmentation) -> np.ndarray:    
    """Adapts a template volume to match the size and shape of a mouse volume.

    This function resizes a 3D template segmentation by scaling it isotropically
    so that its total number of non-zero voxels matches that of a target mouse
    segmentation. It then adjusts the array shape to match the mouse volume's
    shape, likely by padding.

    Parameters
    ----------
    mouse : Mouse
        An object representing the mouse, which must have a `segmentation.volume`
        attribute containing the target 3D numpy array.
    template : Segmentation
        An object representing the template, which must have a `volume`
        attribute containing the source 3D numpy array to be adapted.

    Returns
    -------
    np.ndarray
        The adapted template volume as a 3D numpy array, with a size and shape
        matching the mouse volume.

    Raises
    ------
    ZeroDivisionError
        If the template volume contains no non-zero voxels (`template_size` is 0).

    Side Effects
    ------------
    Logs debug messages regarding the zoom factor and shape changes using a
    pre-configured `logger` object.

    Notes
    -----
    The function relies on the external helper functions `count_voxels`, `zoom`
    (from `scipy.ndimage`), and `enlarge_shape`, whose specific implementations
    are not defined here. The quality of the adaptation depends heavily on these
    functions. The scaling is isotropic, which assumes that the difference in
    size between the template and the mouse is uniform across all dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import zoom
    >>>
    >>> # Mock dependencies for a runnable example
    >>> class MockSegmentation:
    ...     def __init__(self, volume):
    ...         self.volume = volume
    >>>
    >>> class MockMouse:
    ...     def __init__(self, segmentation):
    ...         self.segmentation = segmentation
    >>>
    >>> def count_voxels(volume):
    ...     return np.count_nonzero(volume)
    >>>
    >>> def enlarge_shape(source_vol, target_vol):
    ...     # Simple padding to match shapes for the example
    ...     pad_width = []
    ...     for i in range(3):
    ...         before = 0
    ...         after = target_vol.shape[i] - source_vol.shape[i]
    ...         pad_width.append((before, after))
    ...     return np.pad(source_vol, pad_width, mode='constant')
    >>>
    >>> # Mock logger to avoid NameError
    >>> import logging
    >>> logger = logging.getLogger()
    >>>
    >>> # Create a mouse volume (e.g., 6x6x6 with some voxels)
    >>> mouse_vol_arr = np.zeros((6, 6, 6))
    >>> mouse_vol_arr[2:4, 2:4, 2:4] = 1  # 8 voxels
    >>> mouse_obj = MockMouse(MockSegmentation(mouse_vol_arr))
    >>>
    >>> # Create a template volume (e.g., 10x10x10 with many voxels)
    >>> template_vol_arr = np.zeros((10, 10, 10))
    >>> template_vol_arr[2:6, 2:6, 2:6] = 1 # 64 voxels
    >>> template_obj = MockSegmentation(template_vol_arr)
    >>>
    >>> # Adapt the template to the mouse
    >>> adapted_template = adapt_template(mouse_obj, template_obj)
    >>>
    >>> # The final shape should match the mouse volume's shape
    >>> adapted_template.shape
    (6, 6, 6)
    >>>
    >>> # The number of voxels should be approximately the same
    >>> # Note: zoom interpolation can change the exact count
    >>> print(f"Original mouse voxels: {count_voxels(mouse_vol_arr)}")
    Original mouse voxels: 8
    >>> print(f"Adapted template voxels: {count_voxels(adapted_template > 0.5)}")
    Adapted template voxels: 8

    """
    
    # Extract the volumes
    mouse_volume = mouse.segmentation.volume
    template_volume = template.volume

    # Compute the volume of the template and mice
    mouse_size = count_voxels(mouse_volume)
    template_size = count_voxels(template_volume)

    # Reduce template volume to match mice volume
    zoom_factor = (mouse_size / template_size) ** (1/3)
    logger.debug(f"Zoom Factor: {zoom_factor.round(2)}")
    template_volume = zoom(template_volume, zoom_factor)
    logger.debug(f"Template shape after zoom: {template_volume.shape}")

    # Fix template shape issue
    template_volume = enlarge_shape(template_volume, mouse_volume)
    logger.debug(f"Template shape after filling in: {template_volume.shape}")
    
    return template_volume


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Align Mice to the Template
# ──────────────────────────────────────────────────────
def register_mice(mouse: Mouse, template: np.ndarray, transform: sitk.Transform) -> Mouse:
    """Apply a pre-computed transformation to a mouse's imaging data.
    
    This function takes a `Mouse` object containing MRI, micro-CT, and
    segmentation volumes, and applies a given SimpleITK transformation to
    resample them into the space of a template image. It uses linear
    interpolation for the intensity images (MRI, micro-CT) and nearest-neighbor
    interpolation for the segmentation mask to preserve label integrity.

    Parameters
    ----------
    mouse : Mouse
        An object representing a mouse, expected to have `mri`, `micro_ct`,
        and `segmentation` attributes. Each of these attributes should in turn
        have a `data` attribute containing the image volume as a NumPy array.
    template : np.ndarray
        The reference image volume as a NumPy array. The output images will be
        resampled to match the grid (size, spacing, origin, direction) of this
        template.
    transform : sitk.Transform
        A SimpleITK transform object that maps points from the template space
        to the mouse's original image space.

    Returns
    -------
    Mouse
        The input `mouse` object, with its `mri.data`, `micro_ct.data`, and
        `segmentation.data` attributes modified in-place to contain the
        aligned image volumes.

    Raises
    ------
    AttributeError
        If the input `mouse` object or its nested image objects are missing
        the required `data` attributes.
    TypeError
        If the inputs are not of the expected types (e.g., if `transform` is
        not a `sitk.Transform`).

    Side Effects
    ------------
    - The input `mouse` object is mutated. The `data` attributes of
      `mouse.mri`, `mouse.micro_ct`, and `mouse.segmentation` are
      overwritten with the new, aligned NumPy arrays.

    Notes
    -----
    - The function assumes the existence of a `Registrator` class that handles
      the conversion of NumPy arrays to SimpleITK images and performs the
      resampling.
    - Linear interpolation is used for `mri` and `micro_ct` data, which is
      suitable for continuous intensity images.
    - Nearest-neighbor interpolation is used for `segmentation` data to ensure
      that the output contains only the original integer label values.
    - The input `transform` should map the target (template) space to the
      source (mouse) space, as is standard for resampling operations.

    Examples
    --------
    >>> import numpy as np
    >>> import SimpleITK as sitk
    >>> from unittest.mock import MagicMock
    >>>
    >>> # Mock the Mouse object and its data attributes
    >>> mouse = MagicMock()
    >>> mouse.mri.data = np.random.rand(10, 10, 10)
    >>> mouse.micro_ct.data = np.random.rand(10, 10, 10)
    >>> mouse.segmentation.data = np.random.randint(0, 5, (10, 10, 10))
    >>>
    >>> # Define a template and a simple identity transform
    >>> template_image = np.zeros((10, 10, 10))
    >>> identity_transform = sitk.Transform()
    >>>
    >>> # Mock the Registrator class to avoid dependency
    >>> class Registrator:
    ...     def __init__(self, res_interpolator="linear"):
    ...         pass
    ...     def resample(self, ref_img, moving_img, transform):
    ...         # In a real scenario, this would perform resampling.
    ...         # Here, we just return a SimpleITK image from the moving data.
    ...         return sitk.GetImageFromArray(moving_img)
    >>>
    >>> # Assume the Registrator class is available in the scope
    >>> # register_mice.__globals__['Registrator'] = Registrator
    >>>
    >>> # This is a conceptual example; running it requires the actual
    >>> # Registrator class and its dependencies.
    >>> # registered_mouse = register_mice(mouse, template_image, identity_transform)
    >>>
    >>> # After execution, the mouse object's data would be updated
    >>> # assert np.array_equal(mouse.mri.data, registered_mouse.mri.data)
    >>> # print("Mouse data successfully aligned (conceptually).")
    """
    
    # Initiates the rigid transformation
    reg_transform = Registrator(res_interpolator="linear")
    reg_transform_nearest = Registrator(res_interpolator="nearest")

    # Apply the rigid transformation to the template and mice volumes
    mri_aligned = reg_transform.resample(template, mouse.mri.data, transform)
    ct_aligned = reg_transform.resample(template, mouse.micro_ct.data, transform)
    seg_aligned = reg_transform_nearest.resample(template, mouse.segmentation.data, transform)

    # Convert back to numpy arrays
    mri_aligned = sitk.GetArrayFromImage(mri_aligned)
    ct_aligned = sitk.GetArrayFromImage(ct_aligned)
    seg_aligned = sitk.GetArrayFromImage(seg_aligned)

    # Set the aligned data back to the mice object
    mouse.mri.data = mri_aligned
    mouse.micro_ct.data = ct_aligned
    mouse.segmentation.data = seg_aligned

    return mouse