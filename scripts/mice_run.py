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
    preprocess_reference_df
)


def main():
    print("Is working")
    mouse = Mouse.from_folder("P874", "tests/integration/fixtures/test_experiment/test_mouse_p874")
    segment_dataframe = pd.read_csv("tests/integration/fixtures/test_segmentation_info.csv")

    # Updates the segmentation data of the mice
    layer_colapsing(mouse, segment_dataframe)

    segment_dataframe = preprocess_reference_df(mouse, segment_dataframe)

    # mouse_run("P874", "tests/integration/fixtures/test_experiment/test_mouse_p874")


def mouse_run(mouse_id: str, folder_path: str) -> None:
    # Initialize the Brains
    mouse = Mouse.from_folder(mouse_id, folder_path)
    adapt_template(mouse, ALLEN_TEMPLATE)

    # Perform the NeuroFrame steps
    align_to_allen(mouse)
    skull = extract_skull(mouse)
    bregma, lambda_ = get_bregma_lambda(mouse, skull)
    new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)


if __name__ == "__main__":
    main()
