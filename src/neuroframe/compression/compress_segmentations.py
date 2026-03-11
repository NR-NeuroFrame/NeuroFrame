# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import nibabel as nib

from pathlib import Path
from ..mouse import Mouse



# ================================================================
# 1. Section: Functions
# ================================================================
def compress_seg(mouse: Mouse, dtype=np.uint32) -> str:
    # 1. Extract the data
    seg_nib = mouse.segmentation.nib
    seg_array = mouse.segmentation.data.astype(dtype)

    # 2. Build the new nib with dtype
    new_img = nib.Nifti1Image(seg_array, affine=seg_nib.affine, header=seg_nib.header.copy())
    new_img.set_data_dtype(dtype)
    new_img.update_header()

    # 3. Save it
    file_name = Path(mouse.paths["segmentations_path"]).stem
    file_name = f"{file_name}_{np.dtype(dtype).name}.nii.gz"
    output_path = Path(mouse.folder) / file_name
    nib.save(new_img, str(output_path))

    return output_path
