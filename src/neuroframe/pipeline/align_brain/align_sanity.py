# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from ...styling import alpha_red_cmap_256, alpha_blue_cmap_256
from ...assertions import assert_same_shape



# ================================================================
# 1. Section: Template Analysis
# ================================================================
def plot_mouse_template_overlay(mouse_volume: np.ndarray, template_volume: np.ndarray) -> None:
    """Plot central slices of a mouse and template brain volume to inspect alignment.

    Generates a 3x3 grid of images using matplotlib to visually compare a
    registered mouse brain volume against a template volume. The grid shows
    central slices from three orthogonal planes (sagittal, coronal, and axial).
    For each plane, it displays the template slice, the mouse brain slice, and a
    transparent overlay of both.

    Parameters
    ----------
    mouse_volume : np.ndarray
        A 3D NumPy array representing the mouse brain volume to be compared.
        It is assumed to be registered to the template's space.
    template_volume : np.ndarray
        A 3D NumPy array representing the reference template brain volume.

    Side Effects
    ------------
    Displays a matplotlib figure window with the 3x3 plot grid.

    Notes
    -----
    - This function assumes that `template_volume` and `mouse_volume` have the
      same shape and are spatially aligned for the overlay to be meaningful.
    - The central slice indices are calculated based on the shape of the
      `template_volume`.
    - The function relies on two externally defined matplotlib colormaps:
      `alpha_red_cmap_256` and `alpha_blue_cmap_256`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>>
    >>> # Create dummy 3D brain volumes
    >>> shape = (100, 100, 100)
    >>> template = np.zeros(shape)
    >>> mouse = np.zeros(shape)
    >>>
    >>> # Add some features to the volumes
    >>> template[40:60, 40:60, 40:60] = 1
    >>> mouse[45:65, 45:65, 45:65] = 1
    >>>
    >>> # Generate and display the alignment plot
    >>> # In a real application, plt.show() would be called by the function.
    >>> # For non-interactive testing, we can check if the figure is created.
    >>> plot_mouse_template_overlay(template, mouse)
    >>> plt.close() # Close the plot to prevent it from blocking execution
    """

    # Warns the debugger in case of shape mismatch
    assert_same_shape(mouse_volume, template_volume)

    # Unpacks the data for the loop, more consise this way
    cx, cy, cz = (s // 2 for s in template_volume.shape)
    slices = [(cx, slice(None), slice(None))] * 3 + [(slice(None), cy, slice(None))] * 3 + [(slice(None), slice(None), cz)] * 3
    titles = ["Template Brain", "Mouse Brain", "Brain Overlay"] + ['','',''] * 2
    data = [template_volume, mouse_volume, [template_volume, mouse_volume]] * 3
    cmaps = [alpha_red_cmap_256, alpha_blue_cmap_256, [alpha_red_cmap_256, alpha_blue_cmap_256]] * 3

    # Loop over each subplot and fills it accordingly
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for idx, (ax, title, img, slc, cmap) in enumerate(zip(axes.flat, titles, data, slices, cmaps)):
        if isinstance(img, list):
            for im, cm in zip(img, cmap): ax.imshow(im[slc], cmap=cm, alpha=0.5)
        else: ax.imshow(img[slc], cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"Mouse-Template Aligement Inspection", fontsize=16)
    plt.tight_layout()
    plt.show()
