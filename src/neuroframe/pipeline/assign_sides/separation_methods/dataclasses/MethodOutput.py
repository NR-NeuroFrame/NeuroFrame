# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .LateralizedSegment import LateralizedSegment
from .grouping import ClusterData



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MethodOutput:
    lateralized_segment: LateralizedSegment
    condition: bool
    cluster_data: ClusterData | None = None
