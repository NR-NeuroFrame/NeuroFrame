# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from ...mouse import Mouse
from ...styling import alpha_red_cmap_256



# ================================================================
# 1. Section: Functions
# ================================================================
def debug_plot(
    mouse: Mouse,
    lateralized_volume: np.ndarray,
    original_volume: np.ndarray,
    separation_method: str,
    segment_id: int
) -> tuple:
    # 1. Get the number of voxels
    nr_voxels = np.sum(np.where(original_volume > 0))
    l_nr = np.sum(np.where(lateralized_volume == 1))
    r_nr = np.sum(np.where(lateralized_volume == 2))

    # 2. Initialize the Plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 3. Left based slice
    target_y = get_target_slice(np.where(lateralized_volume == 1, 1, 0))
    axs[0].imshow(mouse.mri.data[:, target_y, :])
    img = axs[0].imshow(lateralized_volume[:, target_y, :], cmap=alpha_red_cmap_256)
    fig.colorbar(img, ax=axs[0])
    axs[0].set_title(f"Left Side Focus {target_y}")

    # 4. Whole volume based slice
    target_y = get_target_slice(np.where(original_volume > 0, 1, 0))
    axs[1].imshow(mouse.mri.data[:, target_y, :])
    img = axs[1].imshow(lateralized_volume[:, target_y, :], cmap=alpha_red_cmap_256)
    fig.colorbar(img, ax=axs[1])
    axs[1].set_title(f"Main Side Focus {target_y}")

    # 5. Right based slice
    target_y = get_target_slice(np.where(lateralized_volume == 2, 1, 0))
    axs[2].imshow(mouse.mri.data[:, target_y, :])
    img = axs[2].imshow(lateralized_volume[:, target_y, :], cmap=alpha_red_cmap_256)
    fig.colorbar(img, ax=axs[2])
    axs[2].set_title(f"Right Side Focus {target_y}")

    # 6. Builds the title for more info
    title = (
        f"Separation Method: {separation_method}\n"
        f"ID: {segment_id} with {nr_voxels} voxels\n"
        f"Left: {l_nr} Right: {r_nr}\n"
        f"Ratio L/T {(l_nr/nr_voxels)*100:.2f}% and R/T {(r_nr/nr_voxels)*100:.2f}%"
    )

    fig.suptitle(title)
    plt.show()

    return fig, axs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_target_slice(volume: np.ndarray) -> int:
    """Returns the slice on the :2 axis where most segment is visible"""
    fg = volume != 0

    # 1. count foreground pixels per slice (sum over axes 0 and 1)
    per_slice = fg.sum(axis=(0, 2))  # shape (Z,)

    return int(per_slice.argmax())
