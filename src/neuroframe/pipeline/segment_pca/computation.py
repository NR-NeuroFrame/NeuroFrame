# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .PCASummary import PCASummary


# ================================================================
# 1. Section: Get a Segment PCA
# ================================================================
def get_segment_pca(seg_lab, seg_left, seg_right) -> PCASummary:
    left_pca = get_volume_pca(seg_left)
    right_pca = get_volume_pca(seg_right)

    return PCASummary(
        id=seg_lab,
        left_pca=left_pca,
        right_pca=right_pca,
    )



# ================================================================
# 2. Section: Get a volumes PCA
# ================================================================
def get_volume_pca(volume: np.ndarray) -> np.ndarray:
    mri_mask = np.where(volume > 0, 1, 0)

    coords = np.column_stack(np.where(mri_mask > 0))
    coords_mean = np.mean(coords, axis=0)
    centered_coords = coords - coords_mean

    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]

    return principal_components
