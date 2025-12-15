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
# 1. Section: Rigid Class
# ================================================================
class Rigid(RegistratorSupport):
    def rigid_transform(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray) -> tuple[np.ndarray, sitk.Transform]:
        
        # Properly convert the images to SimpleITK format
        fixed_image = convert_input(fixed_image)
        moving_image = convert_input(moving_image)

        # Check if the fixed and moving images have the same size, if not resamples the moving image
        if(self.check_shape): moving_image = apply_shape(fixed_image, moving_image)

        # Initialize the rigid registration
        registration_method = self.setup_rigid(fixed_image, moving_image)
        logger.debug("Rigid registration setup complete.")

        # Execute registration
        start_time = time.time()
        transform = registration_method.Execute(fixed_image, moving_image)
        registration_time = time.time() - start_time
        logger.debug(f"Rigid registration executed in {registration_time}.")

        # Resample moving image
        resampled_image = self.resample(fixed_image, moving_image, transform)
        logger.debug("Resampling complete.")

        # Convert back to numpy
        registered_np = sitk.GetArrayFromImage(resampled_image)

        logger.info(f"Final metric value: {registration_method.GetMetricValue()}")
        logger.info(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")

        return registered_np, transform
    

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Setup Rigid Registration
    # ──────────────────────────────────────────────────────
    def setup_rigid(self, fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.ImageRegistrationMethod:
        
        # Initialize the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Set the metric, interpolator, and optimizer
        registration_method = self.define_loss(registration_method)
        registration_method = self.define_registration_interpolator(registration_method)
        registration_method = self.define_optimizer(registration_method)
        registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
        if(self.optimizer != 'LBFGS'): registration_method.SetOptimizerScalesFromPhysicalShift()

        # Connect all of the observers so that we can perform plotting during registration.
        if(self.view_update): registration_method = view_registration(registration_method)

        # Allow for multi-resolution registration
        registration_method = self.define_multiple_resolutions(registration_method)
        
        # Create the transform
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                            moving_image,
                                                            self.define_dimension_transform(),
                                                            self.define_center_type())
        
        # Set the initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        return registration_method