# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pandas as pd

from pathlib import Path

from ..mouse import Mouse



# ================================================================
# 1. Section: Functions
# ================================================================
def save_bl_coords(mouse: Mouse, coords: tuple | np.ndarray, coords_label: tuple | np.ndarray) -> str:
    coords_df = pd.DataFrame(columns=["id", "z", "y", "x"])

    # 1. Generates a DF for BL
    for idx, point in enumerate(coords):
        coords_df.loc[len(coords_df)] = [
            f"{mouse.id}-{coords_label[idx].title()}",
            point[0],
            point[1],
            point[2],
        ]

    # 2. Saves the files
    file_name = f"{mouse.id.lower()}_bl_coords.csv"
    output_folder = Path(mouse.folder)
    output_path = output_folder / file_name
    coords_df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path

def load_bl_coords(mouse: Mouse) -> tuple[np.ndarray, np.ndarray]:
    file_name = f"{mouse.id.lower()}_bl_coords.csv"
    output_folder = Path(mouse.folder)
    file_path = output_folder / file_name

    coords_df = pd.read_csv(file_path)

    # normalize id strings for robust matching
    ids = coords_df["id"].astype(str).str.strip().str.lower()

    def _get_point(name: str) -> np.ndarray:
        # match rows where id ends with "-<name>" (e.g., "-bregma", "-lambda")
        mask = ids.str.endswith(f"-{name.lower()}")
        if not mask.any():
            # fallback: maybe id is exactly "bregma" / "lambda"
            mask = ids.eq(name.lower())
        if not mask.any():
            raise ValueError(f"Could not find '{name}' in {file_path}. Available ids: {coords_df['id'].tolist()}")

        row = coords_df.loc[mask].iloc[0]
        return np.asarray([row["z"], row["y"], row["x"]], dtype=np.float32)

    bregma = _get_point("bregma")
    lambda_ = _get_point("lambda")

    return (bregma, lambda_)
