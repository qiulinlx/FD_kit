import pandas as pd

from utils import functional_richness, functional_evenness, functional_divergence
from utils import calculate_relative_abundance, euclidean_distance

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

FRic = functional_richness(abundances, traits, relative_abundance=False)

print("Functional Richness:\n", FRic)

distance_matrix_euclidean = euclidean_distance(
    traits, metric="euclidean", standardize=True
)

FEve = functional_evenness(
    abundances, distance_matrix_euclidean, relative_abundance=False
)
print("Functional Evenness:\n", FEve)

FDiv = functional_divergence(abundances, traits)

print("Functional Divergence:\n", FDiv)


bird_loc = pd.read_csv("./data/example/bird/bird_location.csv")
bird_traits = pd.read_csv("./data/example/bird/bird_traits.csv")

bird_loc = bird_loc.set_index("PID")
bird_traits = bird_traits.set_index("Species")

FRic = functional_richness(bird_loc, bird_traits, relative_abundance=False)
print(FRic)

distance_matrix_euclidean = euclidean_distance(
    bird_traits, metric="euclidean", standardize=True
)
FEve = functional_evenness(
    bird_loc, distance_matrix_euclidean, relative_abundance=False
)

print(FEve)

FDiv = functional_divergence(bird_loc, bird_traits)

print(FDiv)
