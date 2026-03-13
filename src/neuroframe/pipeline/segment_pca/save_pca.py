# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from pathlib import Path

from ...mouse import Mouse
from .pca_df import PCADF



# ================================================================
# 1. Section: Functions
# ================================================================
def save_mouse_pca(mouse: Mouse, pca_dfs: PCADF, tag: str | None = None) -> Path:
    # 0. handle tag
    tag = tag_setter(tag)

    # 1. Get the path
    output_folder = Path(mouse.folder)
    output_path = output_folder / f"{mouse.id.lower()}_pca_results{tag}.xlsx"

    # 2. Save the excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pca_dfs.left_df.to_excel(writer, sheet_name="left", index=False)
        pca_dfs.right_df.to_excel(writer, sheet_name="right", index=False)

    return output_path


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def tag_setter(tag: str | None) -> str:
    if tag is None:
        return ""
    else:
        return f"_{tag}"
