import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform


# --- Abundance calculation functions ---


def compute_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative abundances (%) for a given abundance column.

    Args:
        df (pd.DataFrame): pandas DataFrame with species abundances
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


def standardize_trait_matrix(
    trait_matrix: pd.DataFrame, method: str = None
) -> pd.DataFrame:
    """
    Wrapper function to standardize the trait matrix using the specified method.

    Args:
        trait_matrix (pd.DataFrame): DataFrame with species as rows and traits as columns.
        method (str, default=None): Method for standardization.
            - "z_score": standardize traits to mean=0 and var=1.
            - "min_max": standardize traits to range [0, 1].
            - None: do not standardize traits.

    Returns:
        pd.DataFrame: Standardized trait matrix.
    """
    if method is None:
        return trait_matrix.copy()

    if method == "z_score":
        return (trait_matrix - trait_matrix.mean()) / trait_matrix.std(ddof=1)
    elif method == "min_max":
        return (trait_matrix - trait_matrix.min()) / (
            trait_matrix.max() - trait_matrix.min()
        )
    else:
        raise ValueError("Invalid method. Choose 'z_score', 'min_max' or None.")


# --- Distance calculation functions ---


def compute_distance_matrix(
    traits: pd.DataFrame, metric: str, standardize_method: str = None
) -> pd.DataFrame:
    """Compute the pairwise distance matrix for a given trait matrix.

    Args:
        traits (pd.DataFrame): DataFrame with species as rows and traits as columns

        metric (str): distance metric to use
            - scipy.spatial.distance.pdist metrics: "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalsneath", "sqeuclidean", "yule"

        standardize_method (str, default=None): Method for standardizing traits before computing distances

    Returns:
        pd.DataFrame: pairwise (Species x Species) distance matrix
    """

    traits = standardize_trait_matrix(traits, method=standardize_method)

    SCIPY_METRICS = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]

    if metric in SCIPY_METRICS:
        distance_matrix = squareform(pdist(traits.values, metric=metric))
    else:
        raise ValueError("Invalid distance metric. Choose from SCIPY_METRICS.")

    return pd.DataFrame(distance_matrix, index=traits.index, columns=traits.index)
