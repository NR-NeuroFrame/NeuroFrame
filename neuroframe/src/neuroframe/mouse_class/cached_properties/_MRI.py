# ================================================================
# 0. Section: Impports
# ================================================================
from functools import cached_property

from ...utils import normalize


class CachedProperties:
    @cached_property
    def data(self): return normalize(self.nib.get_fdata())