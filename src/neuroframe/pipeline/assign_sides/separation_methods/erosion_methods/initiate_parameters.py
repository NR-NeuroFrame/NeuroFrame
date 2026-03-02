# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from skimage.morphology import ball



# ================================================================
# 1. Section: Initiate Salem
# ================================================================
def get_starting_opening_size(method: str) -> int:
    if(method == 'z_directed'): return 2
    elif(method == 'ball'): return 1

def get_selem_shape(method: str, opening_size: int) -> np.ndarray:
    if(method == 'ball'): return ball(opening_size)
    elif(method == 'z_directed'): return np.ones((opening_size, 1, 1), dtype=bool)
