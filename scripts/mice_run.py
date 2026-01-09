from neuroframe import *

def main():
    mouse_run('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')


def mouse_run(mouse_id: str, folder_path: str) -> None:
    # Initialize the Brains
    mouse = Mouse.from_folder(mouse_id, folder_path)
    template_vol = adapt_template(mouse, ALLEN_TEMPLATE)

    # Perform the NeuroFrame steps
    align_to_allen(mouse)
    skull = extract_skull(mouse)
    bregma, lambda_ = get_bregma_lambda(mouse, skull)
    new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)

if __name__ == "__main__":
    main()