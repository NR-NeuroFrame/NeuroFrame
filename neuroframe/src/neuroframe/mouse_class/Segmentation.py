# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from .MedicalImage import MedicalImage
from ..utils import normalize

from .dunders._Segmentation import Dunders
from .cached_properties._Segmentation import CachedProperties



# ================================================================
# 1. Section: MRI Class
# ================================================================
class Segmentation(Dunders, CachedProperties, MedicalImage):
    def __init__(self, path):
        super().__init__(path)