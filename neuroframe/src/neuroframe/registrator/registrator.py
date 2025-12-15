# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from SimpleITK import ImageRegistrationMethod

from ..logger import logger
from .registrator_utils import *
from .types import Rigid, Affine, BSpline



# ================================================================
# 1. Section: Registrator Class
# ================================================================
class Registrator(Rigid, Affine, BSpline):
    def __init__(self, 
                 method: str = 'rigid', 
                 loss: str = 'MI', 
                 optimizer: str = 'GD', 
                 dimension: int = 3, 
                 view_update: bool = False, 
                 check_shape: bool = False, 
                 verbose: int = 0, 
                 **kwargs):
        
        # Direct Assignments
        self.method = method
        self.loss = loss
        self.optimizer = optimizer
        self.dimension = dimension
        self.view_update = view_update
        self.check_shape = check_shape
        self.verbose = verbose

        # Kwargs Default Parameters
        self.numberOfIterations = kwargs['numberOfIterations'] if 'numberOfIterations' in kwargs else 100
        self.sampling_percentage = kwargs['sampling_percentage'] if 'sampling_percentage' in kwargs else 0.1
        self.reg_interpolator = kwargs['interpolator'] if 'interpolator' in kwargs else 'linear'
        self.res_interpolator = kwargs['res_interpolator'] if 'res_interpolator' in kwargs else 'nearest'
        self.rigid_type = kwargs['rigid_type'] if 'rigid_type' in kwargs else 'moments'
        self.multiple_resolutions = kwargs['multiple_resolutions'] if 'multiple_resolutions' in kwargs else False
        self.shrinkFactors = kwargs['shrinkFactors'] if 'shrinkFactors' in kwargs else [4, 2, 1]
        self.smoothingSigmas = kwargs['smoothingSigmas'] if 'smoothingSigmas' in kwargs else [2, 1, 0]

        # |----- Default Parameters for LBFGS-----|
        self.gradientConvergenceTolerance = kwargs['gradientConvergenceTolerance'] if 'gradientConvergenceTolerance' in kwargs else 1e-5
        self.maximumNumberOfCorrections = kwargs['maximumNumberOfCorrections'] if 'maximumNumberOfCorrections' in kwargs else 5

        # |----- Default Parameters for Gradient Descent -----|
        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1
        self.convergenceMinimumValue = kwargs['convergenceMinimumValue'] if 'convergenceMinimumValue' in kwargs else 1e-6
        self.convergenceWindowSize = kwargs['convergenceWindowSize'] if 'convergenceWindowSize' in kwargs else 10
        self.estimateLearningRate = kwargs['estimateLearningRate'] if 'estimateLearningRate' in kwargs else ImageRegistrationMethod.Once
        self.maximumStepSizeInPhysicalUnits = kwargs['maximumStepSizeInPhysicalUnits'] if 'maximumStepSizeInPhysicalUnits' in kwargs else 1.0
        
        # |----- Default Parameters for Exhaustive -----|
        self.numberOfSteps = kwargs['numberOfSteps'] if 'numberOfSteps' in kwargs else [0, 1, 1, 0 , 0, 0]
        self.stepLength = kwargs['stepLength'] if 'stepLength' in kwargs else 1.0

        ## |----- Default Parameters for Bspline -----|
        self.grid_size = kwargs['grid_size'] if 'grid_size' in kwargs else 2
        self.bin_size = kwargs['bin_size'] if 'bin_size' in kwargs else 50

        ## |----- Composite Transform -----|
        self.isComposite = kwargs['isComposite'] if 'isComposite' in kwargs else False
        self.composite = kwargs['composite'] if 'composite' in kwargs else []

    def register(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray, **kwargs) -> sitk.Image:

        self.composite = kwargs['composite'] if 'composite' in kwargs else self.composite

        if(self.verbose >= 1): print(f"NR_Registrator: Registrator initialized with method: {self.method}, loss: {self.loss}, optimizer: {self.optimizer}, dimension: {self.dimension}")
        if(self.method == 'rigid'): results = self.rigid_transform(fixed_image, moving_image)
        elif(self.method == 'bspline' or self.method == 'deform'): results = self.deform_transform(fixed_image, moving_image)
        elif(self.method == 'affine'): results = self.affine_transform(fixed_image, moving_image)
        else: logger.error(f"Method {self.method} not supported yet.")

        return results

def inspect_template(skull_surface, template, deformed):
    """
    Visualizes the transformations applied to a template in relation to the skull surface 
    and sutures. The function generates a grid of subplots to compare the original setup, 
    translated template, warped template, and their transformations.

    Parameters:
        mice_data (tuple): A tuple containing the skull surface and sutures as grayscale images.
        template (ndarray): The initial template image.
        deformed (tuple): A tuple containing the translated and warped templates.
        loss (tuple): A tuple containing the loss values for translation and warping.

    Returns:
        None: Displays the plots for visual inspection.
    """
    # Extract the data
    translated, warped = deformed

    # Extract Gradients
    skull_sutures = np.gradient(skull_surface)
    skull_sutures = np.sqrt(skull_sutures[0]**2 + skull_sutures[1]**2)
    template_sutures = np.gradient(template)
    template_sutures = np.sqrt(template_sutures[0]**2 + template_sutures[1]**2)

    # Remove background
    skull_surface = np.where(skull_surface > 0, skull_surface, np.nan)
    skull_sutures = np.where(skull_sutures > 0, skull_sutures, np.nan)
    template = np.where(template > 0, template, np.nan)
    template_sutures = np.where(template_sutures > 0, template_sutures, np.nan)
    translated = np.where(translated > 0, translated, np.nan)
    warped = np.where(warped > 0, warped, np.nan)

    # Plot to inspect template transformations
    plt.figure(figsize=(11, 11))
    rows, cols = 2, 5
    plt.subplot(rows, cols, 1)
    plt.title("Original Setup")
    plt.imshow(skull_surface, cmap='gray')
    plt.imshow(template, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 2)
    plt.title("After Rigid Transformation")
    plt.imshow(skull_surface, cmap='gray')
    plt.imshow(translated, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 3)
    plt.title("After Deformation")
    plt.imshow(skull_surface, cmap='gray')
    plt.imshow(warped, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 4)
    plt.title("Template Transformations (Rigid)")
    plt.imshow(template, cmap='gray', label='Template')
    plt.imshow(translated, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 5)
    plt.title("Template Transformations (Deformed)")
    plt.imshow(warped, cmap='summer')

    plt.subplot(rows, cols, 6)
    plt.title("Original Setup")
    plt.imshow(skull_sutures, cmap='gray')
    plt.imshow(template_sutures, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 7)
    plt.title("After Translation")
    plt.imshow(skull_sutures, cmap='gray')
    plt.imshow(translated, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 8)
    plt.title("After warping")
    plt.imshow(skull_sutures, cmap='gray')
    plt.imshow(warped, cmap='summer', alpha=0.5)

    plt.subplot(rows, cols, 9)
    plt.title("Template Transformations (Warped)")
    plt.imshow(template, cmap='gray')
    plt.imshow(warped, cmap='summer', alpha=0.5, label='Warped')

    plt.subplot(rows, cols, 10)
    plt.title("Skull Surface")
    plt.imshow(skull_surface, cmap='gray')

    plt.suptitle("Template Transformations", fontsize=16)

    plt.tight_layout()
    plt.show()
