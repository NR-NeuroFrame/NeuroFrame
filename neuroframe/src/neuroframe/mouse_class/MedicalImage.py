# ================================================================
# 0. Section: Imports
# ================================================================
from .dunders._MedicalImage import Dunders
from .properties._MedicalImage import Properties
from .cached_properties._MedicalImage import CachedProperties



# ================================================================
# 1. Section: Image Classes
# ================================================================
class MedicalImage(Dunders, Properties, CachedProperties):
    def __init__(self, path: str):
        self.path = path