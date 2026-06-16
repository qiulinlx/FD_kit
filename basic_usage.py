import numpy as np
import pandas as pd
from utils.fdiv import Functional_Richness, Functional_Evenness

traits = pd.DataFrame({
    "Species": ["Sp_0", "Sp_1", "Sp_2"],
    "Trait_1": [1, 2, 3],
    "Trait_2": [2, 3, 5]
})

abundances = pd.DataFrame(
    [[0.5, 0.3, 0.2]], 
    columns=["Sp_0", "Sp_1", "Sp_2"], 
    index =["Plot_A"]
)

print(traits)
print(abundances)

FEve = Functional_Evenness(abundances, traits, Relative_abundance=True)
FRic = Functional_Richness(abundances, traits)



print("Functional Evenness:", FEve)
print("Functional Richness:", FRic)


