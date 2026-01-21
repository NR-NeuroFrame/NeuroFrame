import pandas as pd

from neuroframe import (
    ALLEN_TEMPLATE,
    Mouse,
    adapt_template,
    align_to_allen,
    align_to_bl,
    extract_skull,
    get_bregma_lambda,
    layer_colapsing,
    preprocess_reference_df,
    stereotaxic_coordinates
)
from neuroframe.utils import get_folders



def main():
    mouse_ids = get_folders("data")
    segment_df = pd.read_csv("data/annotations_info.csv")

    for mouse_id in mouse_ids:
        print(f"Starting mouse {mouse_id}")
        mouse_run(mouse_id, "data", segment_df)


def mouse_run(mouse_id: str, group_folder_path: str, segment_df: pd.DataFrame) -> None:
    # Initialize the Brains
    folder_path = f"{group_folder_path}/{mouse_id}"
    mouse = Mouse.from_folder(mouse_id, folder_path)
    adapt_template(mouse, ALLEN_TEMPLATE)

    # Perform the NeuroFrame steps
    align_to_allen(mouse)
    skull = extract_skull(mouse)
    bregma, lambda_ = get_bregma_lambda(mouse, skull)
    new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)
    layer_colapsing(mouse, segment_df)
    segment_df = preprocess_reference_df(mouse, segment_df)
    stereotaxic_coordinates(
        mouse = mouse,
        reference_df = segment_df,
        ref_coords=(new_bregma, new_lambda),
        group_folder=group_folder_path,
        file_name="stereotaxic_coordinates_MEAN",
        mode="full_mean"
    )


if __name__ == "__main__":
    main()
