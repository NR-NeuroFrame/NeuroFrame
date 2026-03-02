# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path

from ..mouse import Mouse


# ================================================================
# 1. Section: Functions
# ================================================================
def save_summary(
    mouse: Mouse,
    summary_df: pd.DataFrame,
    channel_type: str
) -> Path:

    # 1. Handles the channel type variables
    if(channel_type.lower() == "hemisphere"): file_name = f"{mouse.id.lower()}_sides_summary.csv"
    else: raise ValueError(f"Unknown channel_type: {channel_type!r}")

    # 2. Saves the files
    output_folder = Path(mouse.folder)
    output_path = output_folder / file_name

    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path
