import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler

import networkx as nx
from .abundance_utils import calculate_relative_abundance
from .utils import standardize_trait_matrix


from scipy.spatial import Delaunay

"""
TODO: 
Add argument:
Euclidean vs Gower
Abundance Weighting for Centroids and more


Functional richness (FRic),
Functional volume intersections (FRic_intersect),
Functional divergence (FDiv),
Functional evenness (FEve),
Functional dispersion (FDis)

Rao's Quadratic Entropy (RaoQ)

"""


def functional_richness(
    sp_loc: pd.DataFrame,
    traits: pd.DataFrame,
    relative_abundance: bool = False,
    standardize_traits=True,
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

        standardize_traits (bool, default=True): if True, traits will be standardized to mean=0 and std=1 before computing FRic.

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Functional_Richness": Functional Richness (FRic) value for each plot

    Notes:
    """
    # Pre-checks
    # Calculate relative abundances if not specified
    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    if "Species" in traits.columns:
        traits = traits.set_index("Species")

    # Standardize traits globaly
    if standardize_traits:
        traits = standardize_trait_matrix(traits)

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

        # Compute the convex hull and its volume
        hull = ConvexHull(traits_sub)
        FRic = hull.volume

        pIDs.append(pID)
        FRic_list.append(FRic)

    return pd.DataFrame({"PID": pIDs, "Functional_Richness": FRic_list})


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


def functional_evenness(
    sp_loc: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    relative_abundance: bool = False,
    abundance_weighted: bool = True,
) -> pd.DataFrame:
    """
    Compute Functional Evenness (FEve) using the MST approach (Villéger et al., 2008)
    for presence/absence data.

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        distance_matrix: pre-computed distance matrix
        relative_abundance: whether to use relative abundances
        abundance_weighted: whether to weight edges by abundances

    Returns:
        FEve_df: DataFrame with PID and Functional Evenness
    """
    if not relative_abundance:
        sp_loc = calculate_relative_abundance(sp_loc)

    pIDs = []
    FEve_list = []

    # Get species present in each PID
    for pID in sp_loc.index:

        site_row = sp_loc.loc[pID]
        present_species = site_row[site_row > 0].index

        valid_species = distance_matrix.index.intersection(present_species)
        S = len(valid_species)
        if S < 2:
            # FEve undefined for < 2 species
            FEve_list.append(np.nan)
            pIDs.append(pID)

            continue

        # Subset traits for present species
        valid_dist_matrix = distance_matrix.loc[valid_species, valid_species].values

        # Minimum Spanning Tree
        G = nx.from_numpy_array(valid_dist_matrix)
        mst = nx.minimum_spanning_tree(G)

        if abundance_weighted:
            weights = site_row[valid_species].values
        else:
            weights = np.full(S, 1 / S)

        # Weighted branch lengths
        EW_list = []

        for i, j, data in mst.edges(data=True):
            EW = data["weight"] / (weights[i] + weights[j])
            EW_list.append(EW)

        EW_list = np.array(EW_list)
        PEW = EW_list / EW_list.sum()

        numerator = np.sum(np.minimum(PEW, 1 / (S - 1))) - 1 / (S - 1)
        denominator = 1 - 1 / (S - 1)

        FEve = numerator / denominator

        FEve_list.append(FEve)
        pIDs.append(pID)

    FEve_df = pd.DataFrame({"PID": pIDs, "Functional_Evenness": FEve_list})
    return FEve_df


def functional_divergence(sp_loc: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Functional Divergence (FDiv).

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        trait_array: np.ndarray of shape (S, T)

    Returns:
        FDiv (float)
    """

    pID = []
    FDivergence = []

    # Get species present in each PID
    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):

        S = len(species)

        if S < 3:
            # FEve undefined for <2 species
            FDivergence.append(np.nan)
            pID.append(pid)

            continue

        ab = sp_loc[sp_loc.index == pid]
        # Relative abundancesp
        ab = ab[[c for c in species if c in ab.columns]]

        ab = ab.loc[
            :, ab.columns.isin(traits["Species"])
        ]  # Check that species are in both Dfs

        ab = ab.div(ab.sum(axis=1), axis=0)
        ab = np.array(ab)[0]

        # Subset traits for present species
        trait_array = traits[traits["Species"].isin(species)].copy()
        trait_array.drop(columns=["Species"], inplace=True)

        # Compute community centroid
        centroid = np.array(np.mean(trait_array, axis=0))
        trait_array = np.array(trait_array)
        distances = np.linalg.norm(trait_array - centroid, axis=1)
        dG = np.mean(distances)

        # Distances to centroid
        delta_d = np.sum(ab * (distances - dG))

        abs_delta_d = np.sum(ab * np.abs(distances - dG))

        FDiv = (delta_d + dG) / (abs_delta_d + dG)
        FDivergence.append(FDiv)
        pID.append(pid)

    FDiv_df = pd.DataFrame({"PID": pID, "Functional_Divergences": FDivergence})

    return FDiv_df


def functional_dispersion(
    sp_loc: pd.DataFrame, traits: np.ndarray, weighted: bool = False
) -> pd.DataFrame:
    """
    Compute Functional Dispersion (FDis) for a community.

    FDis measures the spread of species in trait space.
    Can be computed as abundance-weighted or unweighted.

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits (np.ndarray): Trait matrix of shape (S, T), where S = species, T = traits.
        weighted (bool): If True, compute abundance-weighted FDis. If False, compute unweighted FDis.

    Returns:
        float: Functional Dispersion (FDis)
    """

    pID = []
    FDispersion = []

    # Get species present in each PID
    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):

        S = len(species)

        if S < 3:
            # FEve undefined for <2 species
            FDispersion.append(np.nan)
            pID.append(pid)

            continue

        # Subset traits for present species
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=["Species"], inplace=True)

        if weighted:

            ab = sp_loc[sp_loc.index == pid]
            ab = ab[[c for c in species if c in ab.columns]]
            ab = ab.div(ab.sum(axis=1), axis=0)
            ab = np.array(ab)[0]

            centroid = np.sum(
                traits_sub * ab[:, None], axis=0
            )  # Abundance-weighted centroid

        else:
            centroid = np.mean(traits_sub, axis=0)  # Unweighted centroid

        # Distances from centroid
        distances = np.linalg.norm(traits_sub - centroid, axis=1)

        # Compute FDis
        FDis = np.sum(distances * ab) if weighted else np.mean(distances)
        FDispersion.append(FDis)
        pID.append(pid)

    FDis_df = pd.DataFrame({"PID": pID, "Functional_Dispersion": FDispersion})

    return FDis_df


def raos_Q(sp_loc: pd.DataFrame, traits: np.ndarray) -> pd.DataFrame:
    """
    Compute Rao's Quadratic Entropy (RaoQ) from a trait distance matrix.

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        trait_array: np.ndarray of shape (S, T)

    Returns:
        RaoQ (float)
    """

    pID = []
    RaosQ = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):

        S = len(species)

        if S < 3:
            # FEve undefined for <2 species
            RaosQ.append(np.nan)
            pID.append(pid)

            continue

        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=["Species"], inplace=True)

        dist_matrix = squareform(pdist(traits_sub, metric="euclidean"))

        ab = sp_loc[sp_loc.index == pid]
        ab = ab[[c for c in species if c in ab.columns]]

        ab = ab.loc[
            :, ab.columns.isin(traits["Species"])
        ]  # Check that species are in both Dfs

        ab = ab.div(ab.sum(axis=1), axis=0)
        ab = np.array(ab)[0]

        # Compute abundance weight matrix
        weight_matrix = np.outer(ab, ab)

        # Optional: set diagonal to zero
        # np.fill_diagonal(weight_matrix, 0)

        # Rao's Q = sum(p_i * p_j * d_ij)
        RaoQ = np.sum(weight_matrix * dist_matrix)

        RaosQ.append(RaoQ)
        pID.append(pid)

    RQ_df = pd.DataFrame({"PID": pID, "Raos_Q": RaosQ})

    return RQ_df


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
