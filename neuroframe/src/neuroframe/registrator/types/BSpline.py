# ================================================================
# 0. Section: Imports
# ================================================================
import time

import SimpleITK as sitk
import numpy as np

from ..registrator_utils import convert_input, apply_shape, view_registration
from ...logger import logger
from ..itk_utils import *
from ..RegistratorSupport import RegistratorSupport



# ================================================================
# 1. Section: BSpline Class
# ================================================================
class BSpline(RegistratorSupport):
    def deform_transform(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray) -> tuple[np.ndarray, sitk.Transform]:
        # Properly convert the images to SimpleITK format
        fixed_image = convert_input(fixed_image)
        moving_image = convert_input(moving_image)

        # Check if the fixed and moving images have the same size, if not resamples the moving image
        if(self.check_shape): moving_image = apply_shape(fixed_image, moving_image)

        # Initialize the deformable registration
        registration_method = self.setup_deform(fixed_image, moving_image)
        
        # Execute registration
        start_time = time.time()
        transform = registration_method.Execute(fixed_image, moving_image)
        registration_time = time.time() - start_time
        logger.debug(f"BSpline registration executed in {registration_time}.")

        # Resample moving image
        resampled_image = self.resample(fixed_image, moving_image, transform)

        # Convert back to numpy
        deformed_np = sitk.GetArrayFromImage(resampled_image)

        # Log final metrics
        logger.info(f"Final metric value: {registration_method.GetMetricValue()}")
        logger.info(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")

        return deformed_np, transform
    

    # |----- Setup -----|
    def setup_deform(self, fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.ImageRegistrationMethod:
        # Initialize the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Set the metric, interpolator, and optimizer
        registration_method = self.define_loss(registration_method)
        registration_method = self.define_registration_interpolator(registration_method)
        registration_method = self.define_optimizer(registration_method)
        registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Connect all of the observers so that we can perform plotting during registration.
        if(self.view_update): registration_method = view_registration(registration_method)

        # Allow for multi-resolution registration
        registration_method = self.define_multiple_resolutions(registration_method)

        # Deformable transform
        if isinstance(self.grid_size, int):
            grid_size = [self.grid_size] * fixed_image.GetDimension()
        else:
            grid_size = self.grid_size
        
        if(self.isComposite):
            # Create BSpline transform
            bspline_transform = sitk.BSplineTransformInitializer(fixed_image, grid_size)
            
            # Combine affine and BSpline transforms using CompositeTransform
            composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
            for i in range(len(self.composite)):
                composite_transform.AddTransform(self.composite[i])
            composite_transform.AddTransform(bspline_transform)
        else:
            # Create BSpline transform
            composite_transform = sitk.BSplineTransformInitializer(fixed_image, grid_size)

        registration_method.SetInitialTransform(composite_transform, inPlace=False)

        return registration_method