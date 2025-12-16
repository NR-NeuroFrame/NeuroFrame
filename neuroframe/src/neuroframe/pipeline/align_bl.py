# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from ..logger import logger
from ..mouse import Mouse
from ..utils import get_z_coord, compute_separation, rotate_mice, transform_points, fit_plane



# ================================================================
# 1. Section: Put the Mouse in the Bregma-Lambda Orientation
# ================================================================
def align_to_bl(mouse: Mouse, bregma_coords: np.array, lambda_coords: np.array, deviation: int = 5):
    # Center and rotate mice according to the Bregma and Lambda coordinates
    mri_shape = mouse.data_shape

    t = compute_separation(mouse.segmentation.volume)
    logger.info(f"Difference before any transformation: {t}%")

    w = compute_separation(mouse.segmentation.volume)
    logger.info(f"Improvement after centering: {t-w}% ({w}%)")

    # Rotate the mice according to the Bregma and Lambda coordinates
    bl_vector = np.array(lambda_coords) - np.array(bregma_coords)
    norm = np.linalg.norm(bl_vector)
    bl_vector = bl_vector / norm
    rotation_matrix, offset = rotate_mice(mouse, bl_vector, [0, 1, 0])
    bregma_coords = np.round(transform_points(bregma_coords, mri_shape, rotation_matrix, offset)).astype(int)
    lambda_coords = np.round(transform_points(lambda_coords, mri_shape, rotation_matrix, offset)).astype(int)

    t = compute_separation(mouse.segmentation.volume)
    logger.info(f"Improvement after rotation: {w-t}% ({t}%)")

    # Fine tune the alignment in the XY plane
    if(deviation > 0):
        align_matrix, align_offset = xy_fine_tune(mouse, bregma_coords, deviation)
        bregma_coords = np.round(transform_points(bregma_coords, mri_shape, align_matrix, align_offset)).astype(int)
        lambda_coords = np.round(transform_points(lambda_coords, mri_shape, align_matrix, align_offset)).astype(int)

    w = compute_separation(mouse.segmentation.volume)
    logger.info(f"Improvement after XY fine tuning: {t-w}% ({w}%)")
    logger.info(f"Final difference: {w}%")

    return np.round(bregma_coords).astype(int), np.array(lambda_coords).astype(int)


def get_helper_points(mouse: Mouse, ref_coords: np.array, deviation: int):
    # Get a point to the left (in the x-axis)
    left_point = np.array([0, ref_coords[1], ref_coords[2]-deviation]).astype(int)
    left_point[0] = get_z_coord(mouse.micro_ct.data, left_point[1:])

    # Get a point to the right (in the x-axis)
    right_point = np.array([0, ref_coords[1], ref_coords[2]+deviation]).astype(int)
    right_point[0] = get_z_coord(mouse.micro_ct.data, right_point[1:])

    return left_point, right_point

def xy_fine_tune(mouse: Mouse, bregma_coords: np.array, deviation: int):
    # Get auxiliar points from Bregma
    left_bregma, right_bregma = get_helper_points(mouse, bregma_coords, deviation)
    second_bregma = np.array([0, bregma_coords[1] - 2, bregma_coords[2]]).astype(int)
    sleft_bregma, sr_bregma = get_helper_points(mouse, second_bregma, deviation)

    # Fit a plane to the points
    points = np.stack((left_bregma, right_bregma, sleft_bregma, sr_bregma), axis=0)
    normal, D, centroid = fit_plane(points)
    normal[1] = 0 # we only need to rotate around the x-axis
    logger.info(f"\nPlane normal: {normal}")
    logger.debug(f"Points: {points}")

    # Get the upper normal
    if normal[0] < 0: normal = -normal

    # Get the rotation matrix
    align_matrix, offset = rotate_mice(mouse, normal, [1, 0, 0])

    mri_shape = mouse.data_shape

    for point in points:
        point = np.round(transform_points(point, mri_shape, align_matrix, offset)).astype(int)
        logger.debug(f"Points after rotation: {point}")

    return align_matrix, offset