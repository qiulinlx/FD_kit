import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial import Delaunay

from networkx import from_numpy_array, minimum_spanning_tree

from .preprocessing import calculate_relative_abundance
from .preprocessing import standardize_trait_matrix
from .preprocessing import euclidean_distance


"""
TODO:
Add argument:
Euclidean vs Gower

Functional volume intersections (FRic_intersect),
"""


def functional_richness(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame,
    relative_abundance: bool = False,
    standardize_traits_method: str = None,
    local_standardization: bool = False,
) -> pd.DataFrame:
    """
    Compute Functional Richness (FRic) as the volume of the convex hull
    in standardized trait space.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the traits DataFrame "traits.index"
            - Values: Abundance of each species in the corresponding plot

        traits (pd.DataFrame): functional trait matrix of shape (S, T) where S = species, T = traits
            Structure:
            - Row Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"
            - Columns: Trait names
            - Values: Trait values for each species (must be continuous numeric values)

        relative_abundance (bool, default=False): if sp_loc already contains relative abundances, set to True.
            If False, relative abundances will be calculated from absolute abundances.

        standardize_traits_method (str, default=None): Method for standardising traits.
            - "z_score": standardize traits to mean=0 and var=1 before computing FRic.
            - "min_max": standardize traits to range [0, 1] before computing FRic.
            - None: do not standardize traits.

        local_standardization (bool, default=False): If True, standardize traits per plot instead of globally.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Functional_Richness": Functional Richness (FRic) value for each plot

    Notes:
        Trait standardization is done globally across all species, not per plot.

        FRic is undefined for plots with fewer species than traits (S <= T) or fewer than 3 species (S < 3).

        For a single trait (1D), FRic is the range of trait values (max - min) for species present in the plot.
    """
    # Pre-checks
    if "Species" in traits.columns:
        traits = traits.set_index("Species")

    # Calculate relative abundances if not specified
    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    # Standardize traits globaly
    if not local_standardization:
        traits = standardize_trait_matrix(traits, method=standardize_traits_method)

    pIDs = []
    FRic_list = []

    # Core loop over each plot
    for pID in sp_loc.index:
        # Get species present in the plot (have non-zero abundance)
        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        # Subset the species in the traits DataFrame to only those present in the plot
        valid_species = traits.index.intersection(present_species)

        # Subset the traits DataFrame to only include the valid species
        traits_sub = traits.loc[valid_species].copy()

        n_species, n_traits = traits_sub.shape

        # Edge Cases:
        # Number of species must exceed number of traits for convex hull to be defined
        # In the case of 2D traits, at least 3 species are needed to form a convex hull
        if n_species <= n_traits or n_species < 3:
            FRic_list.append(np.nan)
            pIDs.append(pID)
            continue

        if local_standardization:
            traits_sub = standardize_trait_matrix(
                traits_sub, method=standardize_traits_method
            )

        if n_traits == 1:
            # For a single trait, FRic is the range of trait values (max - min)
            FRic = traits_sub.values.max() - traits_sub.values.min()
        else:
            # For multi-dimensional traits, compute the convex hull and its volume
            hull = ConvexHull(traits_sub.values)
            FRic = hull.volume

        pIDs.append(pID)
        FRic_list.append(FRic)

    return pd.DataFrame({"PID": pIDs, "Functional_Richness": FRic_list})


def functional_evenness(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame = None,  # use when local_standardization is True
    distance_matrix: pd.DataFrame = None,  # use when local_standardization is False
    relative_abundance: bool = False,
    abundance_weighted: bool = True,
    local_standardization: bool = False,
    standardize_traits_method: str = None,
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute Functional Evenness (FEve) using the Minimum Spanning Tree (MST) approach (Villéger et al., 2008)

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the distance_matrix DataFrame "distance_matrix.index"
            - Values: Abundance of each species in the corresponding plot

        traits (pd.DataFrame, default=None): functional trait matrix of shape (S, T) where S = species, T = traits
            Structure:
            - Row Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"
            - Columns: Trait names
            - Values: Trait values for each species (must be continuous numeric values)

        distance_matrix (pd.DataFrame, default=None): pre-computed distance matrix
            Structure:
            - Species x Species distance matrix (square form)
            - Row and Column Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"

        relative_abundance (bool, default=False): if sp_loc already contains relative abundances, set to True.
            If False, relative abundances will be calculated from absolute abundances.

        abundance_weighted (bool, default=True): whether to weight edges by abundances.
            True -  each edge in the MST will be weighted by the relativeabundances of the species it connects.
            False - all edges in the MST will be weighted equally as 1/S.

        local_standardization (bool, default=False): If True, standardize traits per plot instead of globally.

        standardize_traits_method (str, default=None): Method for standardizing traits.
            - "z_score": standardize traits to mean=0 and var=1 before computing FEve.
            - "min_max": standardize traits to range [0, 1] before computing FEve.
            - None: do not standardize traits.

        distance_metric (str, default="euclidean"): distance metric to use for computing the distance matrix.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Functional_Evenness": Functional Evenness (FEve) value for each plot

    Notes:
        FEve is undefined for plots with fewer than 3 species.
    """

    # Pre-checks
    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    if not local_standardization:
        if distance_matrix is None:
            if traits is None:
                raise ValueError("Either distance_matrix or traits must be provided.")
            distance_matrix = euclidean_distance(
                traits,
                metric=distance_metric,
                standardize_method=standardize_traits_method,
            )
    else:
        if traits is None:
            raise ValueError(
                "traits must be provided when local_standardization is True."
            )

    pIDs = []
    FEve_list = []

    # Core loop over each plot
    for pID in sp_loc.index:
        # Get species present in the plot (have non-zero abundance)
        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        # Subset the species in the distance matrix to only those present in the plot
        valid_species = distance_matrix.index.intersection(present_species)

        if local_standardization:
            # Subset traits for present species
            traits_sub = traits.loc[valid_species].copy()

            # Compute distance matrix for the standardized traits
            valid_dist_matrix = euclidean_distance(
                traits_sub,
                metric=distance_metric,
                standardize_method=standardize_traits_method,
            )
        else:
            valid_dist_matrix = distance_matrix.loc[valid_species, valid_species].values

        S = len(valid_species)

        # Edge Cases:
        # FEve is undefined for <3 species
        if S < 3:
            FEve_list.append(np.nan)
            pIDs.append(pID)
            continue

        # Minimum Spanning Tree using NetworkX
        G = from_numpy_array(valid_dist_matrix)
        mst = minimum_spanning_tree(G)

        if abundance_weighted:
            weights = site_row[valid_species].values
        else:
            weights = np.full(S, 1 / S)

        # Weighted branch lengths
        EW_list = []

        # Calculate Weighted Evenness (EW) for each branch in the MST
        for i, j, data in mst.edges(data=True):
            EW = data["weight"] / (weights[i] + weights[j])
            EW_list.append(EW)

        # Calculate Partial Weighted Evenness (PEW)
        EW_list = np.array(EW_list)
        PEW = EW_list / EW_list.sum()

        # Calculate Functional Evenness (FEve)
        numerator = np.sum(np.minimum(PEW, 1 / (S - 1))) - 1 / (S - 1)
        denominator = 1 - 1 / (S - 1)
        FEve = numerator / denominator

        FEve_list.append(FEve)
        pIDs.append(pID)

    return pd.DataFrame({"PID": pIDs, "Functional_Evenness": FEve_list})


def functional_divergence(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame,
    relative_abundance: bool = False,
    standardize_traits_method: str = None,
    local_standardization: bool = False,
) -> pd.DataFrame:
    """
    Compute Functional Divergence (FDiv).

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the traits DataFrame "traits.index"
            - Values: Abundance of each species in the corresponding plot

        traits (pd.DataFrame): functional trait matrix of shape (S, T) where S = species, T = traits
            Structure:
            - Row Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"
            - Columns: Trait names
            - Values: Trait values for each species (must be continuous numeric values)

        relative_abundance (bool, default=False): if sp_loc already contains relative abundances, set to True.
            If False, relative abundances will be calculated from absolute abundances.

        standardize_traits_method (str, default=None): Method for standardizing traits.
            - "z_score": standardize traits to mean=0 and var=1 before computing FDiv.
            - "min_max": standardize traits to range [0, 1] before computing FDiv.
            - None: do not standardize traits.

        local_standardization (bool, default=False): If True, standardize traits per plot instead of globally.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Functional_Divergence": Functional Divergence (FDiv) value for each plot

    Notes:
        - FDiv is undefined for plots with fewer than 3 species.
        - Distance to centroid is calculated using Euclidean distance.
    """

    # Pre-checks
    if "Species" in traits.columns:
        traits = traits.set_index("Species")

    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    if not local_standardization:
        traits = standardize_trait_matrix(traits, method=standardize_traits_method)

    pIDs = []
    FDiv_values = []

    # Core loop over each plot
    for pID in sp_loc.index:
        # Get species present in the plot (have non-zero abundance)
        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        # Subset the species in the traits DataFrame to only those present in the plot
        valid_species = traits.index.intersection(present_species)

        S = len(valid_species)

        # Edge Cases:
        # FDiv is undefined for <3 species
        if S < 3:
            FDiv_values.append(np.nan)
            pIDs.append(pID)
            continue

        # Subset traits and abundances for present species
        traits_sub = traits.loc[valid_species].values
        abundances = site_row[valid_species].values

        if local_standardization:
            traits_sub = standardize_trait_matrix(
                traits_sub, method=standardize_traits_method
            )

        # If the number of species is less than the number of traits, FDiv is undefined
        if S < traits_sub.shape[1]:
            FDiv_values.append(np.nan)
            pIDs.append(pID)
            continue

        # Compute centroid based on vertices of the convex hull (Villéger et al., 2008)
        hull = ConvexHull(traits_sub)
        hull_vertices = traits_sub[hull.vertices]
        centroid = np.mean(hull_vertices, axis=0)

        # Compute distances from centroid (Euclidean distance)
        distances = np.linalg.norm(traits_sub - centroid, axis=1)
        # Mean distance to centroid
        dG = np.mean(distances)

        # Weight the distances by abundances
        delta_d = np.sum(abundances * (distances - dG))
        abs_delta_d = np.sum(abundances * np.abs(distances - dG))

        # Compute Functional Divergence (FDiv)
        FDiv = (delta_d + dG) / (abs_delta_d + dG)

        FDiv_values.append(FDiv)
        pIDs.append(pID)

    return pd.DataFrame({"PID": pIDs, "Functional_Divergence": FDiv_values})


def functional_dispersion(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame,
    weighted: bool = True,
    relative_abundance: bool = False,
    standardize_traits_method: str = None,
    local_standardization: bool = False,
) -> pd.DataFrame:
    """
    Compute Functional Dispersion (FDis) for a community.

    FDis measures the spread of species in trait space.
    Can be computed as abundance-weighted or unweighted.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the traits DataFrame "traits.index"
            - Values: Abundance of each species in the corresponding plot

        traits (pd.DataFrame): functional trait matrix of shape (S, T) where S = species, T = traits
            Structure:
            - Row Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"
            - Columns: Trait names
            - Values: Trait values for each species (must be continuous numeric values)

        weighted (bool, default=False): If True, compute abundance-weighted FDis. If False, compute unweighted FDis.

        relative_abundance (bool, default=False): if sp_loc already contains relative abundances, set to True.
            If False, relative abundances will be calculated from absolute abundances.

        standardize_traits_method (str, default=None): Method for standardizing traits.
            - "z_score": standardize traits to mean=0 and var=1 before computing FDis.
            - "min_max": standardize traits to range [0, 1] before computing FDis.
            - None: do not standardize traits.

        local_standardization (bool, default=False): If True, standardize traits per plot instead of globally.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Functional_Dispersion": Functional Dispersion (FDis) value for each plot
    """
    # Pre-checks
    if "Species" in traits.columns:
        traits = traits.set_index("Species")

    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    if not local_standardization:
        traits = standardize_trait_matrix(traits, method=standardize_traits_method)

    pIDs = []
    FDis_values = []

    # Core loop over each plot
    for pID in sp_loc.index:
        # Get species present in the plot (have non-zero abundance)
        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        # Subset the species in the traits DataFrame to only those present in the plot
        valid_species = traits.index.intersection(present_species)

        S = len(valid_species)

        if S < 2:
            FDis_values.append(np.nan)
            pIDs.append(pID)
            continue

        # Subset traits for present species
        traits_sub = traits.loc[valid_species].values

        if local_standardization:
            traits_sub = standardize_trait_matrix(
                traits_sub, method=standardize_traits_method
            )

        if weighted:
            # Use abundance-weighted centroid
            abundances = site_row[valid_species].values

            centroid = np.sum(traits_sub * abundances[:, None], axis=0)
        else:
            # Use unweighted centroid
            centroid = np.mean(traits_sub, axis=0)

        # Distances from centroid
        distances = np.linalg.norm(traits_sub - centroid, axis=1)

        # Compute FDis
        if weighted:
            FDis = np.sum(distances * abundances) / np.sum(abundances)
        else:
            FDis = np.mean(distances)

        FDis_values.append(FDis)
        pIDs.append(pID)

    return pd.DataFrame({"PID": pIDs, "Functional_Dispersion": FDis_values})


def raos_Q(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame = None,  # use when local_standardization is True
    distance_matrix: pd.DataFrame = None,  # use when local_standardization is False
    relative_abundance: bool = False,
    local_standardization: bool = False,
    standardize_traits_method: str = None,
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute Rao's Quadratic Entropy (RaoQ) from a trait distance matrix.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the distance_matrix DataFrame "distance_matrix.index"
            - Values: Abundance of each species in the corresponding plot

        traits (pd.DataFrame, default=None): functional trait matrix of shape (S, T) where S = species, T = traits
            Structure:
            - Row Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"
            - Columns: Trait names
            - Values: Trait values for each species (must be continuous numeric values)

        distance_matrix (pd.DataFrame): pre-computed distance matrix
            Structure:
            - Species x Species distance matrix (square form)
            - Row and Column Index: Species names matching the strings in the sp_loc DataFrame "sp_loc.columns"

        relative_abundance (bool, default=False): if sp_loc already contains relative abundances, set to True.
            If False, relative abundances will be calculated from absolute abundances.

        local_standardization (bool, default=False): If True, standardize traits per plot instead of globally.

        standardize_traits_method (str, default=None): Method for standardizing traits.
            - "z_score": standardize traits to mean=0 and var=1 before computing RaoQ.
            - "min_max": standardize traits to range [0, 1] before computing RaoQ.
            - None: do not standardize traits.

        distance_metric (str, default="euclidean"): distance metric to use for computing the distance matrix.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Raos_Q": Rao's Quadratic Entropy (RaoQ) value for each plot
    """

    # Pre-checks
    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    if not local_standardization:
        if distance_matrix is None:
            if traits is None:
                raise ValueError("Either distance_matrix or traits must be provided.")
            distance_matrix = euclidean_distance(
                traits,
                metric=distance_metric,
                standardize_method=standardize_traits_method,
            )
    else:
        if traits is None:
            raise ValueError(
                "traits must be provided when local_standardization is True."
            )

    pIDs = []
    RaosQ_values = []

    # Core loop over each plot
    for pID in sp_loc.index:
        # Get species present in the plot (have non-zero abundance)
        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        # Subset the species in the distance_matrix DataFrame to only those present in the plot
        valid_species = distance_matrix.index.intersection(present_species)

        if local_standardization:
            # Subset traits for present species
            traits_sub = traits.loc[valid_species].copy()

            # Compute distance matrix for the standardized traits
            valid_dist_matrix = euclidean_distance(
                traits_sub,
                metric=distance_metric,
                standardize_method=standardize_traits_method,
            )
        else:
            valid_dist_matrix = distance_matrix.loc[valid_species, valid_species].values

        S = len(valid_species)

        # Edge Cases:
        # Rao's Q is undefined for <2 species
        if S < 2:
            RaosQ_values.append(np.nan)
            pIDs.append(pID)
            continue

        # Subset the relative abundances for the valid species
        rel_abundances = site_row[valid_species].values

        # Compute Rao's Quadratic Entropy
        # Formula = rel_abundances^T * (dist_matrix^2) * rel_abundances
        RaoQ = (
            np.sum((valid_dist_matrix**2) * np.outer(rel_abundances, rel_abundances))
            / 2
        )

        RaosQ_values.append(RaoQ)
        pIDs.append(pID)

    return pd.DataFrame({"PID": pIDs, "Raos_Q": RaosQ_values})


def MPD(sp_loc: pd.DataFrame, traits: np.ndarray) -> pd.DataFrame:
    pID = []
    mpd_list = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):
        S = len(species)

        if S < 3:
            # Metric undefined for <2 species
            mpd_list.append(np.nan)
            pID.append(pid)

            continue

        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=["Species"], inplace=True)

        # pairwise distances
        distances = pdist(traits_sub.values, metric="euclidean")

        # MPD
        mpd = distances.mean()
        mpd_list.append(mpd)
        pID.append(pid)

    return pd.DataFrame({"PID": pID, "Mean Pairwise D": mpd_list})


def frich_intersect(hull1, hull2, n_samples: int = 100000) -> float:
    """
    GPT GENERATED REQUIRES MORE VERIFICATION
    Compute approximate Functional Volume Intersection (FRic_intersect)
    between two communities using convex hulls.

    Args:
        hull1, hull2: scipy.spatial.ConvexHull objects for each community
        n_samples: number of Monte Carlo samples for approximation

    Returns:
        FRic_intersect: float between 0 and 1
    """
    # Combine points to get bounding box
    all_points = np.vstack([hull1.points, hull2.points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    # Generate random points in bounding box
    samples = np.random.uniform(mins, maxs, size=(n_samples, hull1.points.shape[1]))

    # Helper: check if points are inside a convex hull
    def in_hull(points, hull):
        delaunay = Delaunay(hull.points[hull.vertices])
        return delaunay.find_simplex(points) >= 0

    inside1 = in_hull(samples, hull1)
    inside2 = in_hull(samples, hull2)

    # Intersection fraction
    intersection_fraction = np.sum(inside1 & inside2) / n_samples
    union_fraction = np.sum(inside1 | inside2) / n_samples

    FRic_intersect = intersection_fraction / union_fraction if union_fraction > 0 else 0
    return FRic_intersect
