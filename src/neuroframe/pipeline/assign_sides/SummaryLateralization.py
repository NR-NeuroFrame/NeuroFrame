# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SummaryLateralization:
    id: int
    separation_method: str
    left_ratio: float
    right_ratio: float
