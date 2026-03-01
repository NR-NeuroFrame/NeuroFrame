# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd
import numpy as np




# ================================================================
# 1. Section: Functions
# ================================================================
def build_lateralization_summary(lat_array: np.ndarray):
    # 1. Builds the empty df
    summary = pd.DataFrame(columns=["id", "method", "left_ratio", "right_ratio"])

    # 2. Fills the dataset
    for lateralization in lat_array:
        summary.loc[len(summary)] = [
            lateralization.id,
            lateralization.separation_method,
            lateralization.left_ratio,
            lateralization. right_ratio
        ]

    return summary


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
