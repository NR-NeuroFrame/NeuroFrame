# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path
from neuroframe import Mouse
from neuroframe.templates import ALLEN_TEMPLATE
from neuroframe.save import (
    save_channel,
    save_bl_coords,
    load_bl_coords
)
from neuroframe.mouse_data import (
    Hemisphere,
    SegmentationEDT,
    SegmentationNEDT,
    FieldBL,
    MRI,
    MicroCT,
    Segmentation
)
from neuroframe.pipeline import (
    adapt_template,
    align_to_allen,
    align_to_bl,
    extract_skull,
    get_bregma_lambda,
    layer_colapsing,
    preprocess_reference_df,
    separate_segments,
    edt_segments,
    generate_bl_space,
    get_segments_data,
    get_segments_pca
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE_FOLDER: Path = Path("../data/P874")
MOUSE_ID: str = MOUSE_FOLDER.stem
WT_MOUSE_FOLDER: Path = Path("../data/W001")
WT_MOUSE_ID: Path = WT_MOUSE_FOLDER.stem

SEGMENT_INFO_PATH: Path = Path("data/annotations_info.csv")

TYPE_OF_COORDS: str = "auto"
TYPE_OF_CENTER: str = "wt_shape"



# ================================================================
# 1. Section: FUNCTIONS
# ================================================================
def run_preprocessing_step(mouse: Mouse) -> tuple:
    adapt_template(mouse, ALLEN_TEMPLATE)
    align_to_allen(mouse)
    skull = extract_skull(mouse)
    bregma, lambda_ = get_bregma_lambda(mouse, skull)
    new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)
    _ = layer_colapsing(mouse, segmentation_info)

    save_channel(mouse, mouse.mri.data, "mri")
    save_channel(mouse, mouse.micro_ct.data, "ct")
    save_channel(mouse, mouse.segmentation.data, "seg")
    save_bl_coords(mouse, (new_bregma, new_lambda), ("bregma", "lambda"))

    return new_bregma, new_lambda

def has_pattern_file(folder_path: Path, pattern: str = "*_sides.nii.gz*") -> bool:
    it = folder_path.rglob(pattern)
    return any(p.is_file() for p in it)

def get_pattern_file(mouse_folder: Path, pattern: str = "*_sides.nii.gz*") -> Path:
    # 1. Get the matches
    it = mouse_folder.rglob(pattern)
    matches = sorted([p for p in it if p.is_file()])

    if len(matches) > 1:
        print(
            f"Found multiple '{pattern}' files in {mouse_folder}. "
            f"Using the first: {matches[0].name}. All: {[m.name for m in matches]}"
        )

    return matches[0]



# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Import the data
    mouse = Mouse.from_folder(MOUSE_ID, MOUSE_FOLDER)
    segmentation_info = pd.read_csv(SEGMENT_INFO_PATH)

    if(TYPE_OF_CENTER == "wt_shape"):
        wt_mouse = Mouse.from_folder(WT_MOUSE_ID, WT_MOUSE_FOLDER)
        mri_path = get_pattern_file(WT_MOUSE_FOLDER, "*_proc_mri.nii.gz*")
        ct_path = get_pattern_file(WT_MOUSE_FOLDER, "*_proc_ct.nii.gz*")
        seg_path = get_pattern_file(WT_MOUSE_FOLDER, "*_proc_seg.nii.gz*")
        hemisphere_path = get_pattern_file(WT_MOUSE_FOLDER, "*_sides.nii.gz*")
        edt_path = get_pattern_file(WT_MOUSE_FOLDER, "*_edt.nii.gz*")
        nedt_path = get_pattern_file(WT_MOUSE_FOLDER, "*_nedt.nii.gz*")
        bl_space_path = get_pattern_file(WT_MOUSE_FOLDER, "*_bl_space.nii.gz*")

        wt_mouse.mri = MRI(str(mri_path))
        wt_mouse.micro_ct = MicroCT(str(ct_path))
        wt_mouse.segmentation = Segmentation(str(seg_path))
        wt_mouse.add_path(hemisphere_path, Hemisphere)
        wt_mouse.add_path(edt_path, SegmentationEDT)
        wt_mouse.add_path(nedt_path, SegmentationNEDT)
        wt_mouse.add_path(bl_space_path, FieldBL)

    # 2. Apply pipeline
    if not has_pattern_file(MOUSE_FOLDER, "*_proc_mri.nii.gz*"):
        print("Applying Brain Alignment to BL space")
        bregma, lambda_ = run_preprocessing_step(mouse)
    else:
        mri_path = get_pattern_file(MOUSE_FOLDER, "*_proc_mri.nii.gz*")
        ct_path = get_pattern_file(MOUSE_FOLDER, "*_proc_ct.nii.gz*")
        seg_path = get_pattern_file(MOUSE_FOLDER, "*_proc_seg.nii.gz*")
        mouse.mri = MRI(str(mri_path))
        mouse.micro_ct = MicroCT(str(ct_path))
        mouse.segmentation = Segmentation(str(seg_path))
        bregma, lambda_ = load_bl_coords(mouse)
    segmentation_info = preprocess_reference_df(mouse, segmentation_info)

    # 3. Get the left-right separations channel
    if not has_pattern_file(MOUSE_FOLDER, "*_sides.nii.gz*"): lateralization = separate_segments(mouse)
    else:
        hemisphere_path = get_pattern_file(MOUSE_FOLDER, "*_sides.nii.gz*")
        mouse.add_path(hemisphere_path, Hemisphere)

    # 4. Get the edt and nedt spaces channels
    if not has_pattern_file(MOUSE_FOLDER, "*edt.nii.gz*"): edt_space, nedt_space = edt_segments(mouse)
    else:
        edt_path = get_pattern_file(MOUSE_FOLDER, "*_edt.nii.gz*")
        nedt_path = get_pattern_file(MOUSE_FOLDER, "*_nedt.nii.gz*")
        mouse.add_path(edt_path, SegmentationEDT)
        mouse.add_path(nedt_path, SegmentationNEDT)

    # 5. Get the BL space channel
    if not has_pattern_file(MOUSE_FOLDER, "*_bl_space.nii.gz*"): bl_space = generate_bl_space(mouse, bregma)
    else:
        bl_space_path = get_pattern_file(MOUSE_FOLDER, "*_bl_space.nii.gz*")
        mouse.add_path(bl_space_path, FieldBL)

    # 6. Calculates the center
    centers_df = get_segments_data(mouse, segmentation_info, TYPE_OF_CENTER, wt_mouse)
    centers_df = get_segments_data(mouse, segmentation_info, "inner")

    # 7. Build the PCA excel for the segments volume
    pca_df = get_segments_pca(mouse, segmentation_info)
