# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt


from ....mouse import Mouse
from ....registrator import Registrator
from ..pca import get_volume_pca_components
from ..dataclasses import Center, PCASummary
from .inner_center import compute_safe_inner
from ....styling import alpha_red_cmap_256, alpha_blue_cmap_256
from ..pca import get_segment_pca

SEGMENT_REGISTRATOR = Registrator(
    method="rigid",
    multiple_resolutions=True
)



# ================================================================
# 1. Section: Functions
# ================================================================
def get_shape_centers(
    seg_lab: int,
    seg_left: np.ndarray,
    seg_right: np.ndarray,
    template_mouse: Mouse
) -> tuple:
    # 0. If segment not present in WT just skip this (make all 0,0,0 ?)
    if seg_lab not in template_mouse.segmentation.labels:
        return (
            Center.empty(seg_lab),
            PCASummary.empty(seg_lab)
        )

    # 1. Get the WT Laterals
    wt_seg = np.where(template_mouse.segmentation.data == seg_lab, template_mouse.hemisphere.data, 0)
    wt_left = np.where(wt_seg == 1, 1, 0)
    wt_right = np.where(wt_seg == 2, 1, 0)

    # 2. Do the rigid registration to the mci shape
    wt_trs_left, left_transform = SEGMENT_REGISTRATOR.register(
        seg_left, wt_left
    )
    wt_trs_right, right_transform = SEGMENT_REGISTRATOR.register(
        seg_right, wt_right
    )

    # 3. Get the WT NEDT
    wt_left_nedt = np.where(wt_seg == 1, template_mouse.segmentation_nedt.data, 0)
    wt_right_nedt = np.where(wt_seg == 2, template_mouse.segmentation_nedt.data, 0)

    # 4. Apply it to the nedt
    wt_left_trs_nedt = SEGMENT_REGISTRATOR.apply_transform(wt_left_nedt, left_transform)
    wt_right_trs_nedt = SEGMENT_REGISTRATOR.apply_transform(wt_right_nedt, right_transform)

    # 5. Get the centers
    left_center = compute_safe_inner(wt_left_trs_nedt)
    right_center = compute_safe_inner(wt_right_trs_nedt)
    seg_center = Center(
        id=seg_lab,
        left_center=left_center,
        right_center=right_center
    )

    # 6. Get the PCA of the transformed
    pca_data = get_segment_pca(seg_lab, wt_trs_left, wt_trs_right)
    print(pca_data.left_pca)
    print(pca_data.left_pca.shape)

    return seg_center, pca_data

def plot_compare_seg(volume, wt_volume):
    main_slices = get_main_slices(volume)
    main_slices_wt = get_main_slices(wt_volume)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (orientation, slice) in enumerate(main_slices.items()):
        target_ax = axes[idx]
        target_ax.imshow(slice, cmap=alpha_red_cmap_256)
        target_ax.imshow(main_slices_wt[orientation], cmap=alpha_blue_cmap_256)

    # Add overall title and show plot
    fig.suptitle("Comparing WT-MCI Segment shape")
    plt.tight_layout()
    plt.show()


def plot_pca_orientations(volume: np.ndarray, **kwargs):
    # Compute PCA components for the volume
    volume_pca = get_volume_pca_components(volume)

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
