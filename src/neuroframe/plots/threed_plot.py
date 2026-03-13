# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figures import Figure



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_3d_volumes(
    volume_1: np.ndarray,
    volume_2: np.ndarray,
    step: int = 4,
    max_points: int = 8000
) -> tuple[Figure, Axes]:

    # 1. Prepare the data
    c1 = np.argwhere(volume_1[::step, ::step, ::step] > 0)
    c2 = np.argwhere(volume_2[::step, ::step, ::step] > 0)

    # 2. Downsample if volumes are too big
    if len(c1) > max_points:
        c1 = c1[np.random.choice(len(c1), max_points, replace=False)]
    if len(c2) > max_points:
        c2 = c2[np.random.choice(len(c2), max_points, replace=False)]

    # 3. Do the plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], s=1, alpha=0.5, label='volume 1')
    ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], s=1, alpha=0.5, label='volume 2')
    ax.legend()

    return fig, ax
