# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

from ..logger import logger
from ..mouse import Mouse
from ..assertions import assert_points_transformed_properly
from .image_utils import get_z_coord



# ================================================================
# 1. Section: Rotation Related Functions
# ================================================================
def rotate_mice(mouse: Mouse, vector: np.ndarray, ref_vector: np.ndarray, offset=-1) -> tuple:
    # Extract the data
    mri = mouse.mri.data
    micro_ct = mouse.micro_ct.data
    segmentation = mouse.segmentation.data

    # Compute the quaternion that rotates vector to ref_vector and builds the rotation matrix
    quaternion = quaternion_from_vectors(vector, ref_vector)
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    if(offset == -1):
        center = np.array(mri.shape) / 2
        offset = center - rotation_matrix.T @ center
    else: offset = np.array([0, 0, 0])

    # Rotate the volume using the rotation matrix
    rotated_mri = affine_transform(mri, rotation_matrix.T, offset=offset, order=1)
    rotated_micro_ct = affine_transform(micro_ct, rotation_matrix.T, offset=offset, order=1)
    rotated_segmentation = affine_transform(segmentation, rotation_matrix.T, offset=offset, order=0)

    # Apply the translation to the volume
    mouse.mri.data = rotated_mri
    mouse.micro_ct.data = rotated_micro_ct
    mouse.segmentation.data = rotated_segmentation

    return rotation_matrix, offset

def quaternion_from_vectors(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Normalize the vectors
    v = v / np.linalg.norm(v)
    t = t / np.linalg.norm(t)
    
    dot = np.dot(v, t)
    
    # Handle the case when vectors are opposite
    if np.isclose(dot, -1.0):
        # Find an arbitrary perpendicular vector
        arbitrary = np.array([1, 0, 0])
        if np.linalg.norm(np.cross(v, arbitrary)) < 1e-6:
            arbitrary = np.array([0, 0, 1])
        axis = np.cross(v, arbitrary)
        axis = axis / np.linalg.norm(axis)
        # Quaternion representing 180 degree rotation about the chosen axis
        q = np.concatenate((axis * np.sin(np.pi / 2), [np.cos(np.pi / 2)]))
        return q
    
    # Calculate quaternion components
    s = np.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s
    cross = np.cross(v, t)
    q = np.array([cross[0] * invs, cross[1] * invs, cross[2] * invs, s * 0.5])
    
    # Normalize the quaternion for safety
    q /= np.linalg.norm(q)
    return q

def transform_points(point: np.ndarray, shape: np.ndarray, rotation_matrix: np.ndarray, offset: None | int = None):
    # Create empty volume
    temp_vol = np.zeros(shape)

    # Set a single 1 at the point location
    temp_vol[tuple(np.round(point).astype(int))] = 1

    # Apply the affine transform (same as done to MRI)
    transformed_vol = affine_transform(
        temp_vol,
        rotation_matrix.T,
        offset=offset,
        order=0
    )

    # Find the new location of the 1
    transformed_coords = np.argwhere(transformed_vol > 0.1)

    # Validate the transformation
    assert_points_transformed_properly(transformed_coords)

    return transformed_coords[0]  # Should only be one point




# ================================================================
# 2. Section: Plane Fitting Functions
# ================================================================
def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    # Makes sure the input is type np.int32
    points = points.astype(np.int32)

    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Center the points by subtracting the centroid
    centered_points = points - centroid
    
    # Perform SVD on the centered points
    U, S, Vt = np.linalg.svd(centered_points)
    
    # The normal of the plane is the last singular vector (smallest singular value)
    normal = Vt[-1]
    
    # Compute D using the plane equation: n . (x - centroid) = 0 => n . x + D = 0,
    # thus D = -n . centroid
    D = -np.dot(normal, centroid)
    
    return normal, D, centroid

def get_helper_points(mouse: Mouse, ref_coords: np.array, deviation: int) -> tuple[np.array, np.array]:
    # Get a point to the left (in the x-axis)
    left_point = np.array([0, ref_coords[1], ref_coords[2]-deviation]).astype(int)
    left_point[0] = get_z_coord(mouse.micro_ct.data, left_point[1:])

    # Get a point to the right (in the x-axis)
    right_point = np.array([0, ref_coords[1], ref_coords[2]+deviation]).astype(int)
    right_point[0] = get_z_coord(mouse.micro_ct.data, right_point[1:])

    return left_point, right_point

def xy_fine_tune(mouse: Mouse, bregma_coords: np.array, deviation: int) -> tuple[np.array, np.array]:
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



# ================================================================
# 3. Section: Center Calculation Functions
# ================================================================
def compute_inner_center(binary_mask: np.ndarray, get_map: bool = False) -> np.ndarray:
    """Compute the inner center of a binary mask using the Euclidean Distance Transform (EDT).

    Parameters:
        binary_mask (numpy.ndarray): A 3D binary mask where the object of interest is represented by non-zero values.

    Returns:
        numpy.ndarray: A 1D array containing the 3D coordinates of the inner center of the binary mask.
    """
    
    # Compute the Euclidean Distance Transform (EDT)
    distances = distance_transform_edt(binary_mask) 
    
    # Find the 3d coordinate of the maximum distance
    max_index = np.argmax(distances)
    center = np.unravel_index(max_index, binary_mask.shape)

    if get_map: return np.array(center), distances
    return np.array(center)