# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .MedicalImage import MedicalImage

from .dunders._Hemisphere import Dunders
from .properties._Hemisphere import Properties



# ================================================================
# 1. Section: Hemishpere Class
# ================================================================
class Hemisphere(Dunders, Properties, MedicalImage):
    """Let's make L1 and R2"""
    def __init__(self, path):
        super().__init__(path)
