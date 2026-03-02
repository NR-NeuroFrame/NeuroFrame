# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from ..mouse_data import FieldBL
from ..mouse import Mouse
from ..save import save_channel



# ================================================================
# 1. Section: Functions
# ================================================================
def generate_bl_space(mouse: Mouse, bregma: tuple | np.ndarray) -> np.ndarray:
    # 1. Generate a 3d array with a coordinate
    z, y, x = mouse.data_shape
    coords = np.stack(np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij"), axis=-1)
    voxel_size = np.asarray(mouse.voxel_size)

    # 2. Create this map into mm space
    coords_mm = coords * voxel_size
    bregma_mm = bregma * voxel_size

    # 3. Centers to bregma
    coords_bl = bregma_mm - coords_mm
    coords_bl[..., 0] *= -1 # mirrors the z

    # 4. Saves the BL Space
    bl_space_path = save_channel(mouse, coords_bl, "bl")
    mouse.add_path(bl_space_path, FieldBL)

    print(f" BL Channel saved at {bl_space_path}")
    return coords_bl


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def mirror_midline_x(coords: np.ndarray) -> np.ndarray:
    # 1. Extracts the data
    out = coords.copy()
    x = out[..., 2]

    # 2. Creates a mask for the right side and flips it
    mask = x < 0
    out[..., 2] = np.where(mask, -x, x)

    return out
