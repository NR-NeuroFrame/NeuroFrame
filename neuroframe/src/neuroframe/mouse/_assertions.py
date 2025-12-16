# ================================================================
# 0. Section: Imports
# ================================================================
import os
from ..logger import logger


# ================================================================
# 1. Section: Property Assertions
# ================================================================
def assert_folder_consitency(paths_dict: dict[str, str]) -> None:
    # Get's all the folder's
    micro_ct_folder = paths_dict["ct_path"].rsplit('/', 1)[0]
    mri_folder = paths_dict["mri_path"].rsplit('/', 1)[0]
    segmentations_folder = paths_dict["segmentations_path"].rsplit('/', 1)[0]

    # Check if they all match
    if not (micro_ct_folder == mri_folder == segmentations_folder): logger.warning("The folder paths of the mouse data do not match, defaulted to MRI folder path.")

def assert_shape_consitency(shapes: list[tuple[int, int, int]]) -> None:
    # Check if all shapes match
    if not all(shape == shapes[0] for shape in shapes): logger.warning("The shapes of the mouse data do not match, defaulted to MRI shape.")

def assert_voxel_size_consitency(voxel_sizes: list[tuple[float, float, float]]) -> None:
    # Check if all voxel sizes match
    if not all(voxel_size == voxel_sizes[0] for voxel_size in voxel_sizes): 
        logger.warning("The voxel sizes of the mouse data do not match, defaulted to MRI voxel size.")
        logger.debug(f"Voxel sizes found: {voxel_sizes}")

def assert_id_folder_consitency(folder_path: str, mouse_id: str) -> None:
    # Check if the folder path contains the mouse ID
    condition = mouse_id.lower() in folder_path.lower()
    if not condition: logger.warning(f"The folder path '{folder_path}' does not contain the mice ID '{mouse_id}'.")



# ================================================================
# 2. Section: Class Constructors Assertions
# ================================================================
def assert_required_files(folder: str):
    required_files = ['_mri', '_uCT', '_seg']
    folder_files = os.listdir(folder)

    for req_file in required_files:
        if not any(req_file in file for file in folder_files):
            logger.critical(f"Required file ending with '{req_file}' not found in folder '{folder}'.")
            raise FileNotFoundError(f"Required file ending with '{req_file}' not found in folder '{folder}'.")
        
def assert_no_extra_files(folder: str):
    required_files = ['_mri', '_uCT', '_seg']
    folder_files = os.listdir(folder)

    for file in folder_files:
        if sum(file.endswith(req_file) for req_file in required_files) > 1:
            logger.critical(f"File '{file}' in folder '{folder}' matches multiple required file endings.")
            raise FileNotFoundError(f"Extra file '{file}' found in folder '{folder}' which is not required.")