import json
import pandas as pd

from sklearn.preprocessing import StandardScaler
"""
Additional Functions that is used for generation etc. 
"""

def parse_geo_string(geo_str):
    '''
    Select longitude and latitute values from a string.
    '''
    # Fix doubled quotes
    fixed = geo_str.replace('""', '"').strip('"')
    
    # Convert to dict
    d = json.loads(fixed)

    # Extract lon / lat
    lon, lat = d["coordinates"]
    return lon, lat


def truncate_after_n_underscores(s: str, n: int = 4) -> str:
    """
    Truncate a string after the fourth underscore.
    Using this to standardize plot IDs (PIDs).

    Args:
        s (str): Input string.
        N (int): number of _ before truncation. Defaults to 4 
    Returns:
        str: String containing only the first four underscore-separated segments.
    """
    parts = s.split("_")
    return "_".join(parts[:n])

def standardize_trait_matrix(trait_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the trait matrix to have mean 0 and variance 1 for each trait.

    Args:
        trait_matrix (pd.DataFrame): DataFrame with species as rows and traits as columns.

    Returns:
        pd.DataFrame: Standardized trait matrix.
    """

    scaler = StandardScaler()
    standardized_traits = pd.DataFrame(
        scaler.fit_transform(trait_matrix),
        index=trait_matrix.index,
        columns=trait_matrix.columns
    )
    
    return standardized_traits