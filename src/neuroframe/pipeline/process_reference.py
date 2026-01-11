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
