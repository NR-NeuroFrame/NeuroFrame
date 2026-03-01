# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path
from neuroframe import Mouse
from neuroframe.templates import ALLEN_TEMPLATE
from neuroframe.mouse_data import Hemisphere
from neuroframe.pipeline import (
    adapt_template,
    align_to_allen,
    align_to_bl,
    extract_skull,
    get_bregma_lambda,
    layer_colapsing,
    preprocess_reference_df,
    separate_segments,
    edt_segments
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE_ID: str = "P324"
#MOUSE_FODLER: Path = Path("tests/integration/fixtures/test_experiment/test_mouse_p874")
MOUSE_FODLER: Path = Path("../data/P324")
SEGMENT_INFO_PATH: Path = Path("data/annotations_info.csv")
TYPE_OF_COORDS: str = "auto"
TYPE_OF_CENTER: str = "inner"



# ================================================================
# 1. Section: FUNCTIONS
# ================================================================
def has_sides_file(folder_path: Path, recursive: bool = False) -> bool:
    pattern = "*_sides*"
    it = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
    return any(p.is_file() for p in it)

def get_sides_file(mouse_folder: Path, recursive: bool = False) -> Path:
    pattern = "*_sides*"
    it = mouse_folder.rglob(pattern) if recursive else mouse_folder.glob(pattern)

    matches = sorted([p for p in it if p.is_file()])

    if len(matches) > 1:
        print(
            f"Found multiple '*_sides*' files in {mouse_folder}. "
            f"Using the first: {matches[0].name}. All: {[m.name for m in matches]}"
        )

    return matches[0]



# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Import the data
    mouse = Mouse.from_folder(MOUSE_ID, MOUSE_FODLER)
    segmentation_info = pd.read_csv(SEGMENT_INFO_PATH)

    # 2. Apply pipeline
    adapt_template(mouse, ALLEN_TEMPLATE)
    align_to_allen(mouse)
    skull = extract_skull(mouse)
    bregma, lambda_ = get_bregma_lambda(mouse, skull)
    new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)
    _ = layer_colapsing(mouse, segmentation_info)
    segmentation_info = preprocess_reference_df(mouse, segmentation_info)

    # 3. Save preprocessed mouse (proc_mri, proc_ct, proc_seg)
    # TODO

    # 4. Get the left-right separations
    if not has_sides_file(MOUSE_FODLER): lateralization = separate_segments(mouse)
    else:
        hemisphere_path = get_sides_file(MOUSE_FODLER)
        mouse.add_path(hemisphere_path, Hemisphere)

    # 5. Get the edt and nedt spaces
    edt_data, nedt_data = edt_segments(mouse)

    # dataframe_coords = nf.stereotaxic_coordinates(mice_p324, reference_df, (bregma_coords, lambda_coords), is_parallelized=True, verbose=2, mode='full_inner')
