import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler

# --- Abundance calculation functions ---


def calculate_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative abundances (%) for a given abundance column.

    Args:
        df: pandas DataFrame with species abundances
            each contains the absolute abundances

    Returns:
        df: new dataframe containing relative abundances
    """
    row_sum = df.sum(axis=1)
    df_relative_frequency = df.div(row_sum, axis=0)

    return df_relative_frequency


def normalize_abundance(abundances: list) -> np.ndarray:
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


# --- Trait standardization ---


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
        columns=trait_matrix.columns,
    )

    return standardized_traits


# --- Distance calculation functions ---


def euclidean_distance(
    traits: pd.DataFrame, metric: str = "euclidean", standardize: bool = True
) -> pd.DataFrame:
    """Compute the pairwise distance matrix for a given trait matrix.

    Args:
        traits (pd.DataFrame): DataFrame with species as rows and traits as columns
        metric (str, default="euclidean"): distance metric to use
        standardize (bool, default=True): whether to standardize the traits before computing distances

    Returns:
        pd.DataFrame: pairwise (Species x Species) distance matrix
    """

    if standardize:
        traits = standardize_trait_matrix(traits)

    return pd.DataFrame(
        squareform(pdist(traits, metric=metric)),
        index=traits.index,
        columns=traits.index,
    )
