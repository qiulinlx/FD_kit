import pandas as pd
import numpy as np

from .preprocessing import calculate_relative_abundance


def species_richness(sp_loc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the species richness for each plot in the sp_loc DataFrame.
    Defined as the number of distinct species present in each plot.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names matching the strings in the distance_matrix DataFrame "distance_matrix.index"
            - Values: Abundance of each species in the corresponding plot

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Species Richness": Species richness value for each plot
    """

    pIDs = sp_loc.index.copy()
    # For each plot count the number of species with abundance greater than 0
    SRic = np.sum(sp_loc > 0, axis=1).values

    return pd.DataFrame({"PID": pIDs, "Species Richness": SRic})


def shannon_diversity(sp_loc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Shannon Diversity Index for each plot in the sp_loc DataFrame.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names
            - Values: Abundance of each species in the corresponding plot

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Shannon Diversity": Shannon Diversity value for each plot
    """
    pIDs = sp_loc.index.copy()

    # Calculate relative abundances for each species in each plot
    # The Shannon index requires relative abundances, not absolute abundances
    relative_abundance = calculate_relative_abundance(sp_loc).values

    # Boolean mask to identify valid relative abundances (avoid log(0))
    valid_abundances = relative_abundance > 0

    # Compute the logarithm of relative abundances, setting log(0) to 0 for invalid entries
    log_p = np.zeros_like(relative_abundance)
    log_p[valid_abundances] = np.log(relative_abundance[valid_abundances])

    # Formula: Shannon Diversity = -sum(p_i * log(p_i)) for all species i in the plot
    shannon_values = -np.sum(relative_abundance * log_p, axis=1)

    return pd.DataFrame({"PID": pIDs, "Shannon Diversity": shannon_values})


def simpsons_index(sp_loc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Simpson's Index for each plot in the sp_loc DataFrame.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names
            - Values: Absolute abundance of each species in the corresponding plot

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Simpson's Index": Simpson's Index value for each plot
    """
    pIDs = sp_loc.index.copy()

    # Calculate relative abundances for each species in each plot
    relative_abundance = calculate_relative_abundance(sp_loc).to_numpy()

    #
    simpsons_values = 1 - np.sum(relative_abundance**2, axis=1)

    return pd.DataFrame({"PID": pIDs, "Simpson's Index": simpsons_values}).reset_index(
        drop=True
    )


def shannon_equitability(sp_loc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Shannon Equivalence (also known as the Effective Number of Species).
    Also known as Pielou's Evenness. Measure of Evenness.

    Args:
        sp_loc (pd.DataFrame): Pivot table of Plot IDs and Species
            Structure:
            - Row Index: Plot Identifier (Plot Index)
            - Columns: Species names
            - Values: Absolute abundance of each species in the corresponding plot

    Returns:
        pd.DataFrame: Dataframe containing two columns:
            - "PID": Plot Identifier
            - "Shannon Equitability Index": Shannon Equitability value for each plot
    """
    # Calculate relative abundances for each species in each plot
    # The Shannon Equitability index requires relative abundances, not absolute abundances
    relative_abundance = calculate_relative_abundance(sp_loc).to_numpy()

    pIDs = sp_loc.index.copy()

    # Boolean mask to identify valid relative abundances (avoid log(0))
    valid_abundances = relative_abundance > 0

    # Compute the logarithm of relative abundances, setting log(0) to 0 for invalid entries
    log_p = np.zeros_like(relative_abundance)
    log_p[valid_abundances] = np.log(relative_abundance[valid_abundances])

    # Formula: Shannon Diversity = -sum(p_i * log(p_i)) for all species i in the plot
    shannon_values = -np.sum(relative_abundance * log_p, axis=1)

    # Count the number of species with abundance greater than 0 for each plot
    S = np.sum(sp_loc > 0, axis=1)

    # Boolean mask to identify plots with more than one species (S > 1)
    valid_richness = S > 1

    # Calculate Shannon Equitability Index (Pielou's Evenness)
    # Formula: Shannon Equitability = Shannon Diversity / log(S) for plots with S > 1
    # For plots with S <= 1, the Shannon Equitability is undefined (set to 0)
    shannon_equitability_values = np.zeros_like(shannon_values)
    shannon_equitability_values[valid_richness] = shannon_values[
        valid_richness
    ] / np.log(S[valid_richness])

    return pd.DataFrame(
        {"PID": pIDs, "Shannon Equitability Index": shannon_equitability_values}
    )
