# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from ...logger import logger
from ...styling import alpha_red_cmap_256, alpha_blue_cmap_256



# ================================================================
# 1. Section: View Skull Projection
# ================================================================
def plot_skull(projection_map: np.ndarray, depth_map: None | np.ndarray = None, verbose: int = 1) -> None:
    projection_map = np.where(projection_map < 52, 0, projection_map)
    if depth_map is None:
        logger.info("Depth map not available for inspection.")
        _plot_projection_map(projection_map)
    else: _plot_projection_and_depth(projection_map, depth_map)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Projection Map vs Depth Map Plotting
# ──────────────────────────────────────────────────────
def _plot_projection_map(projection_map: np.ndarray) -> None:
    plt.figure(figsize=(3, 5))
    plt.imshow(projection_map, cmap=alpha_blue_cmap_256)
    plt.vlines(x=projection_map.shape[1]//2, ymin=0, ymax=projection_map.shape[0]-1, color='red', linestyle='--')
    plt.title('Skull Projection Map')
    plt.axis('off')
    plt.show()

def _plot_projection_and_depth(projection_map: np.ndarray, depth_map: np.ndarray) -> None:
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(projection_map, cmap=alpha_blue_cmap_256)
    plt.vlines(x=projection_map.shape[1]//2, ymin=0, ymax=projection_map.shape[0]-1, color='red', linestyle='--')
    plt.title('Skull Projection Map')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap=alpha_red_cmap_256)
    plt.vlines(x=projection_map.shape[1]//2, ymin=0, ymax=projection_map.shape[0]-1, color='red', linestyle='--')
    plt.title('Depth Map')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle('Skull Projection and Depth Map', fontsize=16)
    plt.show()
