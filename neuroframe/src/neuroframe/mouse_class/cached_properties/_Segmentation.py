# ================================================================
# 0. Section: Impports
# ================================================================
import numpy as np

from functools import cached_property



class CachedProperties:
    @cached_property
    def volume(self): return np.where(self.data > 0, 1, 0).astype(np.uint8)

    @cached_property
    def labels(self):
        labels = np.unique(self.data)
        return labels[labels != 0]