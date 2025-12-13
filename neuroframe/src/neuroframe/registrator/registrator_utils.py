# ================================================================
# 0. Section: Imports
# ================================================================
import cv2

import SimpleITK as sitk
import numpy as np

from ..logger import logger
from .itk_utils import *



# ================================================================
# 1. Section: Utils
# ================================================================
def convert_input(input: sitk.Image | np.ndarray) -> sitk.Image:
    """
    Converts the input to a SimpleITK image if it is a numpy array.
    
    Parameters:
        input (sitk.Image | np.ndarray): The input image to be converted.
        
    Returns:
        sitk.Image: The converted SimpleITK image.
    """
    if isinstance(input, np.ndarray):
        return sitk.GetImageFromArray(input.astype(np.float32))
    return input
def apply_shape(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.Image:
    """
    Resizes the fixed_image to match the shape of moving_image if they differ.
    Works for sitk.Image inputs.
    """
    if moving_image.GetSize() != fixed_image.GetSize():
        logger.warning("The template and skull surface have different shapes. Resampling the fixed image to match the moving image.")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        moving_image = resampler.Execute(moving_image)
    return moving_image

# |----- View -----|
def view_registration(registration_method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    """
    Create a registration method with a viewer for visualizing the registration process.
    
    Returns:
        sitk.ImageRegistrationMethod: The configured registration method with a viewer.
    """
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(
        sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
    )
    registration_method.AddCommand(
        sitk.sitkIterationEvent, lambda: plot_values(registration_method)
    )

    print("     Registrator: Registration method with viewer created.")
    
    return registration_method