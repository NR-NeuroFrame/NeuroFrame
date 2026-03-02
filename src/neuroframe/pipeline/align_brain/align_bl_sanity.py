# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass as ndimage_center_of_mass

from ...mouse import Mouse
from ...utils import get_z_coord



# ================================================================
# 1. Section: Plot Alignment Sanity Check
# ================================================================
def plot_alignment(mouse: Mouse) -> None:
    """Plot alignment sanity check for MRI, micro-CT, and segmentation data.

    This function generates a three-panel plot to visually inspect the
    alignment of different imaging modalities for a given mouse. It displays
    mean projections of MRI and micro-CT data, and an overlay of the
    segmentation on the micro-CT data. Key anatomical landmarks, such as the
    center of mass and estimated left/right boundaries, are marked.

    Parameters
    ----------
    mouse : Mouse
        A `Mouse` object containing the imaging data. It is expected to have
        `mri.data`, `micro_ct.data`, `segmentation.volume`, and `data_shape`
        attributes.

    Returns
    -------
    None

    Side Effects
    ------------
    - Displays a matplotlib figure with three subplots.

    Notes
    -----
    - The function calculates projections by taking the mean of the top 25% of
      slices along the z-axis.
    - The center of mass is computed from the `mouse.segmentation.volume`.
    - The plot is displayed in a blocking manner via `plt.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from unittest.mock import Mock
    >>> # Create a mock Mouse object for demonstration
    >>> mock_mouse = Mock()
    >>> mock_mouse.data_shape = (100, 128, 128)
    >>> mock_mouse.mri.data = np.random.rand(100, 128, 128)
    >>> mock_mouse.micro_ct.data = np.random.rand(100, 128, 128)
    >>> mock_mouse.segmentation.volume = np.zeros((100, 128, 128))
    >>> # Create a simple structure in the segmentation to get a center of mass
    >>> mock_mouse.segmentation.volume[75:85, 50:70, 50:70] = 1
    >>> # The following line would generate and display the plot
    >>> # plot_alignment(mock_mouse)

    """

    # Get center of mass
    center_of_mass = np.round(ndimage_center_of_mass(mouse.segmentation.volume)).astype(int)

    # Get the left and right points of micro-CT and segmentation
    left_point, right_point = _get_left_right_points(mouse.micro_ct.data, center_of_mass)
    left_point_seg, right_point_seg = _get_left_right_points(mouse.segmentation.volume, center_of_mass)

    # Extract the projections to plot
    mri_proj, micro_ct_proj, segment_proj = _get_projections(mouse)

    # Start packing data for ploting
    data = [mri_proj, micro_ct_proj, (segment_proj, micro_ct_proj)]
    titles = ["MRI Data (Mean Slice)", "Micro-CT Data (Mean Slice)", "Overlay of MRI and Micro-CT (Mean Slice)"]
    v_lines = [mouse.data_shape[2]//2, mouse.data_shape[1]-1]
    points = [(None, None), (left_point, right_point), (left_point_seg, right_point_seg)]

     # Loop over each subplot and fills it accordingly
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, title, img, point in zip(axes.flat, titles, data, points):
        if(not isinstance(img, tuple)): ax.imshow(img, cmap='gray')
        else:
            for im in img: ax.imshow(im, alpha=0.5, cmap='gray')

        ax.vlines(v_lines[0], 0, v_lines[1], color='r', linestyle='--', linewidth=1, label='Midline')
        ax.scatter(center_of_mass[2], center_of_mass[1], c="blue", marker='x', s=15, label=f"COM (z={center_of_mass[0]})")

        if(point[0] is not None and point[1] is not None):
            ax.scatter(point[0][2], point[0][1], c="green", marker='x', s=15, label=f"Left Point (z={point[0][0]})")
            ax.scatter(point[1][2], point[1][1], c="orange", marker='x', s=15, label=f"Right Point (z={point[1][0]})")

        ax.legend()
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"BL Alignment Inpsection", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Data processing for Plotting
# ──────────────────────────────────────────────────────
def _get_projections(mouse: Mouse) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine slice to plot
    z_len = mouse.data_shape[0]
    z_low = (z_len*3)//4

    # Get the data to plot
    mri_proj = mouse.mri.data[z_low:, :, :].mean(axis=0)
    micro_ct_proj = mouse.micro_ct.data[z_low:, :, :].mean(axis=0)
    segment_proj = mouse.segmentation.volume[z_low:, :, :].mean(axis=0)

    return mri_proj, micro_ct_proj, segment_proj

def _get_left_right_points(volume: np.ndarray, center_of_mass: np.array, offset: int = 40) -> tuple[np.array, np.array]:
    left_point = np.array([0, center_of_mass[1], center_of_mass[2]-offset]).astype(int)
    left_point[0] = get_z_coord(volume, left_point[1:])
    right_point = np.array([0, center_of_mass[1], center_of_mass[2]+offset]).astype(int)
    right_point[0] = get_z_coord(volume, right_point[1:])
    return left_point, right_point
