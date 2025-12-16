# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np


# ================================================================
# 1. Section: Operations
# ================================================================
def normalize(volume: np.ndarray) -> np.ndarray:
	"""
	Normalize a numeric volume to the 0–255 range and return as int16.

	This function performs a linear min–max scaling of the input NumPy array so
	that the minimum value becomes 0 and the maximum value becomes 255. The
	input array's shape is preserved and the result is returned with dtype
	numpy.int16.

	Parameters
	----------
	volume : np.ndarray
			Input array of numeric type (e.g., image or volumetric data). The array
			may be of any shape; normalization is applied elementwise using the
			global minimum and maximum of the array.

	Returns
	-------
	np.ndarray
			Normalized array with the same shape as `volume` and dtype `numpy.int16`.
			Values are mapped approximately to the integer range [0, 255].

	Raises
	------
	ValueError
			If `volume` is constant (max == min), the computation would perform a
			division by zero which produces invalid values that cannot be converted
			to integers. Callers should check for or handle constant-valued inputs
			before calling this function.

	Notes
	-----
	- Scaling formula: ((volume - min(volume)) / (max(volume) - min(volume))) * 255.
	- No explicit clipping beyond the linear mapping is applied.
	- Conversion to `int16` truncates fractional parts; rounding is not applied.
	- For constant inputs, consider returning a constant output (e.g., all zeros)
		or handling that case separately to avoid errors.

	Examples
	--------
	>>> import numpy as np
	>>> arr = np.array([0.0, 0.5, 1.0])
	>>> normalize(arr)
	array([  0, 127, 255], dtype=int16)
	"""
	# Obtain the minimum and maximum values of the data
	min_val = np.min(volume)
	max_val = np.max(volume)

	# Normalize the data to the range [0, 1]
	data_normalized = ((volume - min_val) / (max_val - min_val)) * 255

	return data_normalized.astype(np.int16)

def get_z_coord(volume: np.ndarray, coords: tuple, window_size: int = 10) -> int:
	y, x = coords.astype(int)

	# Get the z coordinate
	ct_z = volume[:, y-window_size:y+window_size, x-window_size:x+window_size]
	ct_z_mean = np.mean(ct_z, axis=(1, 2))  # Mean intensity of each slice

	# Get the first slice above threshold
	gradient = np.gradient(ct_z_mean)
	curvature = np.gradient(gradient)

	gradient[:100] = 0
	curvature[:100] = 0

	z = np.argmin(gradient)

	return np.round(z).astype(int)

def compute_separation(volume: np.ndarray) -> float:
    # Split the volume into left and right hemispheres
    left_hemisphere, right_hemisphere = separate_volume(volume)

    left_per = np.sum(left_hemisphere > 0) / np.sum(volume > 0)
    right_per = np.sum(right_hemisphere > 0) / np.sum(volume > 0)
    
    #print(f"Difference: {abs(left_per - right_per):.2%}")
    return round(abs(left_per - right_per) * 100, 2)

def separate_volume(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_hemisphere = volume.copy()
    right_hemisphere = volume.copy()

    # Set the left hemisphere to zero where the right hemisphere is non-zero
    left_hemisphere[:, :, :volume.shape[2] // 2] = 0
    right_hemisphere[:, :, volume.shape[2] // 2:] = 0

    hemispheres = (left_hemisphere, right_hemisphere)
    return hemispheres