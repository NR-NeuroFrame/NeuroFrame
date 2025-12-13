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
    template_volume = adapt_template(mouse, template)

    # Does the rigid registration
    rigid_registration = Registrator(multiple_resolutions=True)
    _, transform = rigid_registration.register(template_volume, mouse.segmentation.volume)

    logger.detail(f"Obtained Transform: {transform.GetParameters()}")

    # Applies the transformation to the mice
    mouse = register_mice(mouse, template_volume, transform)

    return mouse


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Adapts the Template's Size
# ──────────────────────────────────────────────────────
def adapt_template(mouse: Mouse, template: Segmentation) -> np.ndarray:    
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
    logger.debug(f"Template is now adapted to the Mouse Volume!")
    return template_volume


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Align Mice to the Template
# ──────────────────────────────────────────────────────
def register_mice(mouse: Mouse, template: np.ndarray, transform: sitk.Transform) -> Mouse:
    
    # Initiates the rigid transformation
    rigid_transform = Registrator(res_interpolator="linear")
    rigid_transform_nearest = Registrator(res_interpolator="nearest")

    # Apply the rigid transformation to the template and mice volumes
    mri_aligned = rigid_transform.resample(template, mouse.mri.data, transform)
    ct_aligned = rigid_transform.resample(template, mouse.micro_ct.data, transform)
    seg_aligned = rigid_transform_nearest.resample(template, mouse.segmentation.data, transform)

    mri_aligned = sitk.GetArrayFromImage(mri_aligned)
    ct_aligned = sitk.GetArrayFromImage(ct_aligned)
    seg_aligned = sitk.GetArrayFromImage(seg_aligned)

    logger.detail(f"Aligned MRI shape: {mri_aligned.shape}")
    logger.detail(f"Aligned Micro-CT shape: {ct_aligned.shape}")
    logger.detail(f"Aligned Segmentation shape: {seg_aligned.shape}")
    logger.detail(f"Template shape: {template.shape}")

    if(mri_aligned == mouse.mri.data).all(): logger.warning("MRI data unchanged after registration.")
    if(ct_aligned == mouse.micro_ct.data).all(): logger.warning("Micro-CT data unchanged after registration.")
    if(seg_aligned == mouse.segmentation.data).all(): logger.warning("Segmentation data unchanged after registration.")

    # Set the aligned data back to the mice object
    mouse.mri.data = mri_aligned
    mouse.micro_ct.data = ct_aligned
    mouse.segmentation.data = seg_aligned

    if(mri_aligned == mouse.mri.data).all(): logger.debug("MRI data was sucessefuly assigned.")
    if(ct_aligned == mouse.micro_ct.data).all(): logger.debug("Micro-CT data was sucessefuly assigned.")
    if(seg_aligned == mouse.segmentation.data).all(): logger.debug("Segmentation data was sucessefuly assigned.")

    return mouse