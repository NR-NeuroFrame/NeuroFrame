# ================================================================
# 0. Section: Impports
# ================================================================
import numpy as np


class Properties:
    @property
    def volume(self):
        return np.where(self.data > 0, 1, 0).astype(np.uint8)

    @property
    def left(self):
        return np.where(self.data == 1, 1, 0).astype(np.uint8)

    @property
    def right(self):
        return np.where(self.data == 2, 1, 0).astype(np.uint8)
