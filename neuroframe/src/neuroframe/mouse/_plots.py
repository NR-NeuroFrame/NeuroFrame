# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt


class Plots:
    # ================================================================
    # 1. Section: Plots
    # ================================================================
    def plot_multimodal_midplanes(self, slice_offset: int = 0) -> None:
        # Get the data from the images
        micro_ct = self.micro_ct.data
        mri = self.mri.data
        segmentation = self.segmentation.volume

        # Get the middle point of the images
        middle_point = np.array(self.data_shape) // 2

        # Plotting the images
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        titles = [
            "MicroCT - Axial", "MicroCT - Coronal", "MicroCT - Sagittal",
            "MRI - Axial", "MRI - Coronal", "MRI - Sagittal",
            "Segmentations - Axial", "Segmentations - Coronal", "Segmentations - Sagittal"
        ]
        data = [micro_ct, micro_ct, micro_ct, mri, mri, mri, segmentation, segmentation, segmentation]

        slices = [
            (middle_point[0]+slice_offset, slice(None), slice(None)),
            (slice(None), middle_point[1]+slice_offset, slice(None)),
            (slice(None), slice(None), middle_point[2]+slice_offset)
        ] * 3

        for ax, title, img, slc in zip(axes.flat, titles, data, slices):
            ax.imshow(img[slc], cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.suptitle(f"All Multimodal Midplane Slices of Mice {self.id}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_segmentations_overlay(self, slice_offset: int = 20) -> None:
        # Extract Data
        micro_ct = self.micro_ct.data
        volume = self.segmentation.volume
        mri = self.mri.data

        z_pos = self.data_shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=(8, 8))
        data = [micro_ct] * 3 + [mri] * 3
        titles = ["MicroCT and Segmentations"] * 3 + ["MRI and Segmentations"] * 3
        titles[0], titles[2], titles[3], titles[5] = "", "", "", ""
        slices = [z_pos - slice_offset, z_pos, z_pos + slice_offset] * 2

        for ax, title, img, slc in zip(axes.flat, titles, data, slices):
            ax.imshow(img[slc, :, :], cmap="gray")
            ax.imshow(volume[slc, :, :], cmap="jet", alpha=0.5)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Segmentation Overlay of Mice {self.id}", fontsize=16)
        plt.tight_layout()
        plt.show()