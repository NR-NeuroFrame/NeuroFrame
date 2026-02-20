# ================================================================
# 0. Section: IMPORTS
# ================================================================
from .MedicalImage import MedicalImage

from .dunders._FieldBL import Dunders
from .properties._FieldBL import Properties



# ================================================================
# 1. Section: Hemishpere Class
# ================================================================
class FieldBL(Dunders, Properties, MedicalImage):
    """Stores a 3D Vector at each voxel"""
    def __init__(self, path):
        super().__init__(path)
