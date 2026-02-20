# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .MedicalImage import MedicalImage

from .dunders._SegmentationNEDT import Dunders
from .properties._SegmentationNEDT import Properties



# ================================================================
# 1. Section: Hemishpere Class
# ================================================================
class SegmentationNEDT(Dunders, Properties, MedicalImage):
    """Normalized"""
    def __init__(self, path):
        super().__init__(path)
