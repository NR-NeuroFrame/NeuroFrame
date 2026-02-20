# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from ...styling import NR_BLUE, NR_RED



# ================================================================
# 1. Section: Check Bregma-Lambda Positions
# ================================================================
def plot_bl(skull_surface: np.array, bregma_coords: np.array, lambda_coords: np.array) -> None:
    """Plot a 2D view of the skull surface with Bregma and Lambda landmarks.

    This function is intended for sanity-checking transformations by visualizing a
    2D slice of the skull surface and overlaying the projected coordinates of
    Bregma and Lambda.

    Parameters
    ----------
    skull_surface : np.array
        A 2D NumPy array representing the grayscale image of the skull surface.
    bregma_coords : np.array
        A 1D array of shape (3,) representing the coordinates of Bregma. The
        assumed order is (z, y, x).
    lambda_coords : np.array
        A 1D array of shape (3,) representing the coordinates of Lambda. The
        assumed order is (z, y, x).

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates and displays a matplotlib plot.

    Notes
    -----
    - The function assumes that the input coordinates for Bregma and Lambda are
      in the order (z, y, x). The plot uses the y-coordinate for the vertical
      axis and the x-coordinate for the horizontal axis of the image.
    - The z-coordinate from `bregma_coords` and `lambda_coords` is used only for
      the label text in the plot legend.
    - The line `plt.tight_layout` in the original implementation is a property
      access and has no effect. It should be `plt.tight_layout()` to adjust
      plot parameters for a tight layout.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Create a dummy skull surface image
    >>> skull_img = np.zeros((512, 512))
    >>> # Define dummy coordinates for Bregma and Lambda
    >>> bregma = np.array([10, 200, 256])
    >>> lambd = np.array([12, 400, 256])
    >>> # The plot will be displayed in an interactive window
    >>> # plot_bl(skull_img, bregma, lambd)

    """

    plt.figure(figsize=(8,8))
    plt.imshow(skull_surface, cmap="gray")
    plt.scatter(bregma_coords[2], bregma_coords[1], c=NR_RED, marker='x', s=15, label=f"Bregma (z={bregma_coords[0]})")
    plt.scatter(lambda_coords[2], lambda_coords[1], c=NR_BLUE, marker='x', s=15, label=f"Lambd (z={lambda_coords[0]})")
    plt.title("Transformed Bregma Template")
    plt.legend()
    plt.axis("off")

    plt.tight_layout
    plt.show()
