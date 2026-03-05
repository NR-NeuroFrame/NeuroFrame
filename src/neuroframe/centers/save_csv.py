# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path

from .DataDF import DataDF
from ..mouse import Mouse



# ================================================================
# 1. Section: Functions
# ================================================================
def save_mouse_results(mouse: Mouse, data_dfs: DataDF, mode: str) -> Path:
    # 1. Get the path
    output_folder = Path(mouse.folder)
    output_path = output_folder / f"{mouse.id.lower()}_{mode}_results.xlsx"

    # 2. Save the excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data_dfs.left_df.to_excel(writer, sheet_name="left", index=False)
        data_dfs.right_df.to_excel(writer, sheet_name="right", index=False)
        data_dfs.average_df.to_excel(writer, sheet_name="average", index=False)

    return output_path
