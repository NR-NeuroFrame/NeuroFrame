# ================================================================
# 0. Section: Imports
# ================================================================
from .MedicalImage import MedicalImage
from ..utils import normalize

from .cached_properties._MicroCT import CachedProperties
from .dunders._MicroCT import Dunders

# ================================================================
# 1. Section: MicroCT Class
# ================================================================
class MicroCT(Dunders, CachedProperties, MedicalImage):
    def __init__(self, path):
        super().__init__(path)