# ================================================================
# 0. Section: Imports
# ================================================================
from .MedicalImage import MedicalImage

from .dunders._Segmentation import Dunders
from .properties._Segmentation import Properties



# ================================================================
# 1. Section: Segmentation Class
# ================================================================
class Segmentation(Dunders, Properties, MedicalImage):
    def __init__(self, path):
        super().__init__(path)
