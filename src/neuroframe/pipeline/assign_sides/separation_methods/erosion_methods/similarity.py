# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_volume_similarity(original_volume: np.ndarray, comparing_volume: np.ndarray) -> float:
    # 1. Prepares the comparing vector for subsequent comparison (makes background different)
    comparing_inverted = np.where(comparing_volume == 0, 100, comparing_volume)

    # 2. Compute the similarity mask by comparing the original volume with the inverted comparing volume
    similarity_mask = (comparing_inverted == original_volume)
    similarity_mask = np.where(similarity_mask, 1, 0)

    # 3. Compute similarity in percentage (e.g. 98.23%)
    volume_similarity = np.round(np.count_nonzero(similarity_mask)/np.count_nonzero(original_volume)*100,2)

    return volume_similarity
