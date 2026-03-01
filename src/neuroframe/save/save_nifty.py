# ================================================================
# 0. Section: IMPORTS
# ================================================================
import nibabel as nib
import numpy as np

from pathlib import Path

from ..mouse import Mouse



# ================================================================
# 1. Section: Functions
# ================================================================
def save_channel(
    mouse: Mouse,
    channel: np.ndarray,
    channel_type: str,
    reference_nib: nib.Nifty1Image | None = None
) -> Path:

    # 1. Handles the channel type variables
    if(channel_type.lower() == "hemisphere"):
        file_name = f"{mouse.id.lower()}_sides.nii.gz"
        data = channel.astype(np.uint8, copy=False)
    elif(channel_type.lower() == "edt"):
        file_name = f"{mouse.id.lower()}_edt.nii.gz"
        data = channel.astype(np.float32, copy=False)
    elif(channel_type.lower() == "nedt"):
        file_name = f"{mouse.id.lower()}_nedt.nii.gz"
        data = channel.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown channel_type: {channel_type!r}")

    # 2. Makes sure the reference is usable (asserting shape)
    if(reference_nib is None): reference_nib = mouse.mri.nib
    assert_shape(reference_nib, data)

    # 3. Sets up the metadata
    affine = reference_nib.affine
    header = reference_nib.header.copy()
    header.set_data_dtype(data.dtype)

    # 4. Saves the files
    output_folder = Path(mouse.folder)
    output_path = output_folder / file_name
    out_img = nib.Nifti1Image(data, affine, header=header)
    nib.save(out_img, str(output_path))

    return output_path


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Assertions
# ──────────────────────────────────────────────────────
def assert_shape(ref_nib: nib.Nifty1Image, data: np.ndarray) -> None:
    ref_shape = ref_nib.shape[:3]
    if data.shape != ref_shape:
        raise ValueError(f"Shape mismatch: channel {data.shape} vs reference {ref_shape}")
