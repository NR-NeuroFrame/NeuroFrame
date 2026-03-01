# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from neuroframe import Mouse



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE_ID: str = "P324"
MOUSE_FODLER: Path = Path("tests/integration/fixtures/test_experiment/test_mouse_p874")



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    mouse = Mouse.from_folder(MOUSE_ID, MOUSE_FODLER)

    mouse.plot_segmentations_overlay(slice_offset=70)
    mouse.plot_multimodal_midplanes()
