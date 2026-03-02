# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .MedicalImage import MedicalImage

from .dunders._SegmentationEDT import Dunders
from .properties._SegmentationEDT import Properties



# ================================================================
# 1. Section: Hemishpere Class
# ================================================================
class SegmentationEDT(Dunders, Properties, MedicalImage):
    """Not normalized"""
    def __init__(self, path):
        super().__init__(path)
