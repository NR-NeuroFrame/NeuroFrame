# ================================================================
# 0. Section: Imports
# ================================================================
from .registrator import Registrator



# ================================================================
# 1. Section: Suture Registrator
# ================================================================
SUTURE_REGISTRATOR = Registrator(
    method="deform",
    loss="MI",
    optimizer="GD",
    dimension=2,
    numberOfIterations=1000,
    check_shape=True,
    grid_size=10,
    convergenceMinimumValue=1e-25,
    convergenceWindowSize=100,
    learningRate=1e-20,
    sampling_percentage=1,
    bin_size=100
)
