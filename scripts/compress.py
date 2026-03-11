# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path

from neuroframe import Mouse
from neuroframe.compression import compress_seg


# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE_FODLER: Path = Path("../data/W001")
MOUSE_ID: str = MOUSE_FODLER.stem


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    mouse = Mouse.from_folder(MOUSE_ID, MOUSE_FODLER)

    path = compress_seg(mouse)
    print(f"Saved at {path}")
