import numpy as np
import pandas as pd
from utils.fdiv import functional_richness, functional_evenness
from utils.abundance_utils import calculate_relative_abundance
from utils.utils import standardize_trait_matrix
from utils.distance_utils import euclidean_distance

traits = pd.DataFrame(
    {
        "Species": ["Sp_0", "Sp_1", "Sp_2", "Sp_3"],
        "Trait_1": [1, 2, 3, 4],
        "Trait_2": [2, 3, 1, 2],
    }
).set_index("Species")

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
print("Euclidean Distance Matrix:\n", distance_matrix_euclidean)

FEve = functional_evenness(
    abundances, distance_matrix_euclidean, relative_abundance=False
)
print("Functional Evenness:\n", FEve)


bird_loc = pd.read_csv("./data/example/bird/bird_location.csv")
bird_traits = pd.read_csv("./data/example/bird/bird_traits.csv")

bird_loc = bird_loc.set_index("PID")
bird_traits = bird_traits.set_index("Species")

FRic = functional_richness(bird_loc, bird_traits, relative_abundance = False)
print(FRic)

distance_matrix_euclidean = euclidean_distance(bird_traits, metric="euclidean", standardize=True)   
FEve = functional_evenness(bird_loc, distance_matrix_euclidean, relative_abundance = False)

print(FEve)