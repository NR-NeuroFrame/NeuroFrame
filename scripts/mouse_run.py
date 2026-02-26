# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path
from neuroframe import Mouse
from neuroframe.templates import ALLEN_TEMPLATE
from neuroframe.pipeline import (
    adapt_template,
    align_to_allen,
    align_to_bl,
    extract_skull,
    get_bregma_lambda,
    layer_colapsing,
    preprocess_reference_df,
    separate_segments
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE_ID: str = "P874"
#MOUSE_FODLER: Path = Path("tests/integration/fixtures/test_experiment/test_mouse_p874")
MOUSE_FODLER: Path = Path("../data/P324")
SEGMENT_INFO_PATH: Path = Path("data/annotations_info.csv")
TYPE_OF_COORDS: str = "auto"
TYPE_OF_CENTER: str = "inner"



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

    # 3. Get the left-right separations
    lateralization = separate_segments(mouse)


    # dataframe_coords = nf.stereotaxic_coordinates(mice_p324, reference_df, (bregma_coords, lambda_coords), is_parallelized=True, verbose=2, mode='full_inner')
