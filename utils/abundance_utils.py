import numpy as np
import pandas as pd


def calculate_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative abundances (%) for a given abundance column.

    Args:
        df: pandas DataFrame with species abundances
            each contains the absolute abundances

    Returns:
        df: new dataframe containing relative abundances
    """
    row_sum = df.sum(axis = 1)
    df_relative_frequency = df.div(row_sum, axis = 0)

    return df_relative_frequency

def normalise_abundance(abundances: list) -> np.ndarray:
    """
    Normalize abundances to sum to 1.

    Args:
        abundances: list or array of abundances

    Returns:
        normalized_abundances: np.ndarray of normalized abundances
    """
    abundances = np.array(abundances)
    normalized_abundances = abundances / abundances.sum()
    return normalized_abundances