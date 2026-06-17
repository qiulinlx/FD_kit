import pandas as pd

from scipy.spatial.distance import pdist, squareform

from utils.utils import standardize_trait_matrix

def euclidean_distance(traits: pd.DataFrame, metric: str = "euclidean", standardize: bool = True) -> pd.DataFrame:
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
        columns=traits.index
        )