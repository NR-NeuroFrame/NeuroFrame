# ================================================================
# 0. Section: Imports
# ================================================================
from .MedicalImage import MedicalImage
from .dunders._MRI import Dunders
from .cached_properties._MRI import CachedProperties



# ================================================================
# 1. Section: MRI Class
# ================================================================
class MRI(Dunders, CachedProperties, MedicalImage):
    def __init__(self, path):
        super().__init__(path)