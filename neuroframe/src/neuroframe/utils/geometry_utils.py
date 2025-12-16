# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

from ..mouse import Mouse
from ..assertions import assert_points_transformed_properly



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