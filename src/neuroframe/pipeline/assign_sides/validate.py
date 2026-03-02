# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ...logger import logger


# ================================================================
# 1. Section: Functions
# ================================================================
def validate_lateralization(lateralized_volume):
    if(len(np.unique(lateralized_volume)) > 3):
        logger.critical("The lateralization had a problem and we have overlaps!\n",
            f"Unique values (should only be 0,1 and 2): {np.unique(lateralized_volume)}"
        )
        raise ValueError(
            "The lateralization had a problem and we have overlaps!\n",
            f"Unique values (should only be 0,1 and 2): {np.unique(lateralized_volume)}"
        )


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
