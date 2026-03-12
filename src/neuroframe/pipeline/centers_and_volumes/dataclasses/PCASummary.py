# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PCASummary:
    id: int
    left_pca: np.ndarray
    right_pca: np.ndarray

    @classmethod
    def empty(cls, label: int) -> 'PCASummary':
        return cls(
            id=label,
            left_pca=np.zeros(3, dtype=float),
            right_pca=np.zeros(3, dtype=float),
        )
