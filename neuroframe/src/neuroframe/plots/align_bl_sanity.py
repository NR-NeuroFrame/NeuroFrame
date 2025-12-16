# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass as ndimage_center_of_mass

from ..mouse import Mouse
from ..utils import get_z_coord


# ================================================================
# 1. Section: Plot Alignment Sanity Check
# ================================================================
def plot_alignment(mouse: Mouse):
    # Extract the data
    mri = mouse.mri.data
    micro_ct = mouse.micro_ct.data
    segment_volume = mouse.segmentation.volume
    z_len = segment_volume.shape[0]
    z_low = (z_len*3)//4

    # Get center of mass
    center_of_mass = np.round(ndimage_center_of_mass(segment_volume)).astype(int)

    # Get the left and right points of micro-CT
    left_point = np.array([0, center_of_mass[1], center_of_mass[2]-40]).astype(int)
    left_point[0] = get_z_coord(mouse.micro_ct.data, left_point[1:])
    right_point = np.array([0, center_of_mass[1], center_of_mass[2]+40]).astype(int)
    right_point[0] = get_z_coord(mouse.micro_ct.data, right_point[1:])
    
    # Get the left and right points of segmentation
    left_point_seg = np.array([0, center_of_mass[1], center_of_mass[2]-40]).astype(int)
    left_point_seg[0] = get_z_coord(mouse.segmentation.volume, left_point_seg[1:])
    right_point_seg = np.array([0, center_of_mass[1], center_of_mass[2]+40]).astype(int)
    right_point_seg[0] = get_z_coord(mouse.segmentation.volume, right_point_seg[1:])


    # Plotting the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # MRI Data subplot
    axes[0].imshow(mri[z_low:, :, :].mean(axis=0), cmap='gray')
    axes[0].vlines(mri.shape[2]//2, 0, mri.shape[1]-1, color='r', linestyle='--', linewidth=1, label='Midline')
    axes[0].scatter(center_of_mass[2], center_of_mass[1], c="blue", marker='x', s=15, label=f"COM (z={center_of_mass[0]})")
    axes[0].legend()
    axes[0].set_title("MRI Data (Mean Slice)")
    
    # Micro-CT Data subplot
    axes[1].imshow(micro_ct[z_low:,:,:].mean(axis=0), cmap='gray')
    axes[1].vlines(mri.shape[2]//2, 0, mri.shape[1]-1, color='r', linestyle='--', linewidth=1, label='Midline')
    axes[1].scatter(center_of_mass[2], center_of_mass[1], c="blue", marker='x', s=15, label=f"COM (z={center_of_mass[0]})")
    axes[1].scatter(left_point[2], left_point[1], c="green", marker='x', s=15, label=f"Left Point (z={left_point[0]})")
    axes[1].scatter(right_point[2], right_point[1], c="orange", marker='x', s=15, label=f"Right Point (z={right_point[0]})")
    axes[1].legend()
    axes[1].set_title("Micro-CT Data (Mean Slice)")

    axes[2].imshow(segment_volume[z_low:, :, :].mean(axis=0), cmap='gray', alpha=0.5)
    axes[2].imshow(micro_ct[z_low:, :, :].mean(axis=0), cmap='gray', alpha=0.5)
    axes[2].vlines(mri.shape[2]//2, 0, mri.shape[1]-1, color='r', linestyle='--', linewidth=1, label='Midline')
    axes[2].scatter(center_of_mass[2], center_of_mass[1], c="blue", marker='x', s=15, label=f"COM (z={center_of_mass[0]})")
    axes[2].scatter(left_point_seg[2], left_point_seg[1], c="green", marker='x', s=15, label=f"Left Point (z={left_point_seg[0]})")
    axes[2].scatter(right_point_seg[2], right_point_seg[1], c="orange", marker='x', s=15, label=f"Right Point (z={right_point_seg[0]})")
    axes[2].legend()
    axes[2].set_title("Overlay of MRI and Micro-CT (Mean Slice)")
    
    plt.tight_layout()
    plt.show()

