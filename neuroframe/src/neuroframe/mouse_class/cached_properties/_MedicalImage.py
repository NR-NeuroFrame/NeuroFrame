# ================================================================
# 0. Section: Impports
# ================================================================
import nibabel as nib

from functools import cached_property


class CachedProperties:
    # ================================================================
    # 1. Section: Properties
    # ================================================================
    @cached_property
    def nib(self): return nib.loadsave.load(self.path)
    
    @cached_property
    def data(self): return self.nib.get_fdata()