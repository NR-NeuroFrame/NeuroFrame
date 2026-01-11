# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd
import numpy as np

from ..logger import logger



# ================================================================
# 1. Section: Asserts for layer Colapse
# ================================================================
def assert_all_from_same_parent(data: pd.DataFrame, layer_indexs: list) -> None:
    parents_different = len(set(data['parent_id'].iloc[layer_indexs])) > 1
    if(parents_different): logger.warning("Not all layers have the same parent_id, this may cause issues in the colapsing process.")



# ================================================================
# 2. Section: Asserts for Preprocessing Reference DF
# ================================================================
def assert_no_missing_layers(labels: np.ndarray, reference_df: pd.DataFrame) -> None:
    missing_labels = set(labels) - set(reference_df['id'])
    if len(missing_labels) > 0: logger.warning(f"Missing labels in reference DataFrame: {missing_labels}")
