import pandas as pd
import numpy as np

from .preprocessing import calculate_relative_abundance


def species_richness(sp_loc: pd.DataFrame) -> pd.DataFrame:
    pIDs = sp_loc.index.copy()
    SRic = np.sum(sp_loc > 0, axis=1)

    return pd.DataFrame({"PID": pIDs, "Species Richness": SRic})


def shannon_diversity(sp_loc: pd.DataFrame):

    pIDs = sp_loc.index.copy()

    relative_abundance = calculate_relative_abundance(sp_loc).values

    valid_abundances = relative_abundance > 0

    log_p = np.zeros_like(relative_abundance)
    log_p[valid_abundances] = np.log(relative_abundance[valid_abundances])

    shannon_values = -np.sum(relative_abundance * log_p, axis=1)

    return pd.DataFrame({"PID": pIDs, "Shannon Diversity": shannon_values})


def simpsons_index(sp_loc):
    """
    Docstring for simpsons_index

    Measure of Dominance /
    """
    pIDs = sp_loc.index.copy()

    total = np.sum(sp_loc * (sp_loc - 1), axis=1)

    N = np.sum(sp_loc, axis=1)
    N_sum = N * (N - 1)

    simpsons_values = 1 - (total / N_sum)

    return pd.DataFrame({"PID": pIDs, "Simpson's Index": simpsons_values})


def shannon_equitability(sp_loc: pd.DataFrame):
    """
    Also known as Pielou's Evenness. Measure of Evennes
    Calculate the Shannon Equivalence (also known as the Effective Number of Species)

    """
    relative_abundance = calculate_relative_abundance(sp_loc).values
    pIDs = sp_loc.index.copy()

    S = np.sum(sp_loc > 0, axis=1)

    valid_abundances = relative_abundance > 0

    log_p = np.zeros_like(relative_abundance)
    log_p[valid_abundances] = np.log(relative_abundance[valid_abundances])

    shannon_values = -np.sum(relative_abundance * log_p, axis=1)

    shannon_equitability_values = np.zeros_like(shannon_values)

    valid_richness = S > 0
    shannon_equitability_values[valid_richness] = shannon_values[
        valid_richness
    ] / np.log(S[valid_richness])

    return pd.DataFrame(
        {"PID": pIDs, "Shannon Equitabiltiy Index": shannon_equitability_values}
    )
