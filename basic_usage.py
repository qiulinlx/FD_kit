import pandas as pd

from utils import calculate_relative_abundance
from utils import euclidean_distance
from utils import (
    functional_richness,
    functional_evenness,
    functional_divergence,
    raos_Q,
    functional_dispersion,
)

traits = pd.DataFrame(
    [[1, 2], [2, 3], [3, 1], [4, 2]],
    columns=["Trait_1", "Trait_2"],
    index=["Sp_0", "Sp_1", "Sp_2", "Sp_3"],
)

abundances = pd.DataFrame(
    [[5, 3, 2, 1], [1, 2, 0, 2]],
    columns=["Sp_0", "Sp_1", "Sp_2", "Sp_3"],
    index=["Plot_A", "Plot_B"],
)

relative_abundances = calculate_relative_abundance(abundances)

FRic = functional_richness(
    abundances, traits, relative_abundance=False, standardize_traits_method="z_score"
)

print("Functional Richness:\n", FRic)

distance_matrix_euclidean = euclidean_distance(
    traits, metric="euclidean", standardize_method="z_score"
)

FEve = functional_evenness(
    abundances, distance_matrix_euclidean, relative_abundance=False
)
print("Functional Evenness:\n", FEve)

FDiv = functional_divergence(abundances, traits)
print("Functional Divergence:\n", FDiv)

FDis = functional_dispersion(
    abundances, traits, relative_abundance=False, standardize_traits_method="z_score"
)
print("Functional Dispersion:\n", FDis)


raos_Q_df = raos_Q(abundances, distance_matrix_euclidean, relative_abundance=False)
print("Rao's Quadratic Entropy:\n", raos_Q_df)


tree_loc = pd.read_csv("./data/example/trees/tree_location.csv", index_col=0)
tree_traits = pd.read_csv("./data/example/trees/tree_traits.csv", index_col=0)

FRic_tree = functional_richness(
    tree_loc, tree_traits, relative_abundance=False, standardize_traits_method="z_score"
)

distance_matrix_euclidean_tree = euclidean_distance(tree_traits)

FEve_tree = functional_evenness(
    tree_loc,
    distance_matrix_euclidean_tree,
    relative_abundance=False,
    abundance_weighted=True,
)

FDiv_tree = functional_divergence(
    tree_loc, tree_traits, relative_abundance=False, standardize_traits_method="z_score"
)
