# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from ..logger import logger



# ================================================================
# 1. Section: Shapes
# ================================================================
def assert_points_transformed_properly(points: np.ndarray) -> None:
    if len(points) == 0:
        logger.error("Point got lost in transformation — check bounds or threshold.")
        raise ValueError("Point got lost in transformation — check bounds or threshold.")
    elif len(points) > 1:
        logger.error(f"Multiple points found in transformation — check the transformation matrix. {len(points)} points found.")
        raise ValueError(f"Multiple points found in transformation — check the transformation matrix. {len(points)} points found.")