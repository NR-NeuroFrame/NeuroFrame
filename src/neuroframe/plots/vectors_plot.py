# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from ..pipeline.segment_pca import get_volume_pca



# ================================================================
# 1. Section: PCA Overlay Volume
# ================================================================
def plot_pca_orientations(volume: np.ndarray, **kwargs):
    # Compute PCA components for the volume
    volume_pca = get_volume_pca(volume)

    # Start the plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get the main slices and plot PCA on each
    main_slices = get_main_slices(volume)
    for idx, (orientation, slice) in enumerate(main_slices.items()):
        target_ax = axes[idx]
        _plot_slice_pca(target_ax, slice, volume_pca, orientation.lower(), **kwargs)

    # Add overall title and show plot
    fig.suptitle("PCA Components on Middle Slices")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _plot_slice_pca(axes: plt.Axes, slice: np.ndarray, volume_pca: np.ndarray, orientation: str, **kwargs):
    """Plot a single slice with PCA components overlaid as arrows."""
    scale = kwargs.get('scale', 20)
    colors = kwargs.get('colors', ['#B14E4E', '#4EB14E', '#4E4EB1'])

    axes.imshow(slice, cmap='gray')
    center_x, center_y = slice.shape[1] // 2, slice.shape[0] // 2
    x, y = _get_vector_indecs(orientation)

    for i in range(3):
        axes.arrow(center_x, center_y,
                    volume_pca[x, i] * scale,
                    volume_pca[y, i] * scale,
                    head_width=5, head_length=5, fc=colors[i], ec=colors[i])
    axes.set_title(f"{orientation.title()} View")
    axes.axis('off')

def _get_vector_indecs(orientation: str):
    """Get the indices of the PCA components to plot based on the slice orientation."""
    orientation = orientation.lower()
    if orientation == 'axial': return 2, 1
    elif orientation == 'coronal': return 2, 0
    elif orientation == 'sagittal': return 1, 0
    else: raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")

def get_main_slices(volume: np.ndarray) -> dict:
    mid_slices = {
        'axial': volume[volume.shape[0] // 2, :, :],
        'coronal': volume[:, volume.shape[1] // 2, :],
        'sagittal': volume[:, :, volume.shape[2] // 2]
    }
    return mid_slices
