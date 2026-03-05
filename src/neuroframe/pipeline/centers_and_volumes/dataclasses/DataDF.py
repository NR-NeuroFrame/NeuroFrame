# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pandas as pd

from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DataDF:
    left_df: pd.DataFrame
    right_df: pd.DataFrame
    average_df: pd.DataFrame
