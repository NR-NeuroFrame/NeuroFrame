# ================================================================
# 0. Section: Imports
# ================================================================
import pandas as pd

from ..mouse import Mouse
from ..logger import logger
from ..assertions import assert_no_missing_layers


# ================================================================
# 1. Section: Preprocessing for Reference DataFrame
# ================================================================
def preprocess_reference_df(mouse: Mouse, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a reference DataFrame based on mouse segmentation data.

        This function filters the `reference_df` to include only the entries
        corresponding to labels found in the mouse's segmentation. It also
        removes RGB color columns and logs the result.

        Parameters
        ----------
        mouse : Mouse
            A `Mouse` object containing segmentation data, specifically `mouse.segmentation.labels`.
        reference_df : pd.DataFrame
            The reference DataFrame to be processed. It must contain an 'id' column.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame, filtered and cleaned.

        Raises
        ------
        AssertionError
            If any labels from `mouse.segmentation.labels` are not found in the
            'id' column of `reference_df`, as checked by `assert_no_missing_layers`.

        Side Effects
        ------------
        Logs the content of the processed DataFrame and the count of remaining
        entries using `logger.debug`.

        Notes
        -----
        The function relies on `assert_no_missing_layers` to validate the
        completeness of the reference data against segmentation labels. It also
        calls `remove_rbg_columns` to clean the DataFrame.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> import pandas as pd
        >>> mouse = Mock()
        >>> mouse.segmentation.labels = [1, 3]
        >>> reference_df = pd.DataFrame({
        ...     'id': [1, 2, 3],
        ...     'name': ['A', 'B', 'C'],
        ...     'red': [255, 0, 0],
        ...     'green': [0, 255, 0],
        ...     'blue': [0, 0, 255]
        ... })
        >>> preprocess_reference_df(mouse, reference_df)
        id name
        0   1    A
        2   3    C"""

    # Extract the data from the mouse
    labels = mouse.segmentation.labels

    # Remove every entry from the reference DataFrame that does not correspond to any segmentation label
    reference_df = reference_df[reference_df['id'].isin(labels)]

    # Print any entry that was present in the labels but not in the reference DataFrame
    assert_no_missing_layers(labels, reference_df)

    # Clean the columns
    reference_df = remove_rbg_columns(reference_df)

    logger.debug(f"The Dataframe:\n{reference_df}")
    logger.debug(f"Preprocessed reference DataFrame: {len(reference_df)} entries remaining.\n")
    return reference_df

def remove_rbg_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the data from columns called reed, blue and green
    if 'red' in df.columns: df = df.drop(columns=['red'])
    if 'blue' in df.columns: df = df.drop(columns=['blue'])
    if 'green' in df.columns: df = df.drop(columns=['green'])

    return df
