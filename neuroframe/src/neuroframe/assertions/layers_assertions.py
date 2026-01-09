# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd

from ..logger import logger




# ================================================================
# 0. Section: Asserts for layer Colapse
# ================================================================
def assert_all_from_same_parent(data: pd.DataFrame, layer_indexs: list) -> None:
    parents_different = len(set(data['parent_id'].iloc[layer_indexs])) > 1
    if(parents_different): logger.warning("Not all layers have the same parent_id, this may cause issues in the colapsing process.")