import numpy as np
import pandas as pd

from typing import Tuple

def linear_function(
    mu_0: float, 
    sigma_0: float, 
    alpha: float, 
    global_tas: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the estimated μ(tas) and σ(tas) using a linear model.

    Parameters:
    -----------
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alpha : float
        The linear coefficient for temperature changes.
    global_tas : pd.Series
        Time series of global temperature anomalies.

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        A tuple containing the estimated μ(tas) and σ(tas).
    """
    # Compute estimated μ(tas) and σ(tas)
    mu_tas_est = mu_0 + alpha * global_tas
    sigma_tas_est = sigma_0 * 1

    return mu_tas_est, sigma_tas_est

###############################################################################

def exponential_function(
    mu_0: float, 
    sigma_0: float, 
    alpha: float, 
    global_tas: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the estimated μ(tas) and σ(tas) using an exponential model.

    Parameters:
    -----------
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alpha : float
        The exponential coefficient for temperature changes.
    global_tas : pd.Series
        Time series of global temperature anomalies.

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        A tuple containing the estimated μ(tas) and σ(tas) based on an 
        exponential relationship.
    """
    # Compute estimated μ(tas) and σ(tas)
    exp_term = np.exp(alpha * global_tas / mu_0)
    mu_tas_est = mu_0 * exp_term
    sigma_tas_est = sigma_0 * exp_term

    return mu_tas_est, sigma_tas_est

###############################################################################

def choose_strategy(
    strategy: str,    
    mu_0: float, 
    sigma_0: float, 
    alpha: float, 
    global_tas: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Select and apply a model (linear or exponential) for computing μ(tas) and σ(tas).

    Parameters:
    -----------
    strategy : str
        The strategy to use ('linear' or 'exponential').
    mu_0 : float
        The baseline mean (μ_0) at the initial temperature.
    sigma_0 : float
        The baseline standard deviation (σ_0) at the initial temperature.
    alpha : float
        The coefficient for temperature changes.
    global_tas : pd.Series
        Time series of global temperature anomalies.

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        A tuple containing the estimated μ(tas) and σ(tas) based on the selected 
        strategy.
    """
    # Compute estimated μ(tas) and σ(tas)
    if strategy == 'linear':
        mu_tas_est, sigma_tas_est = linear_function(
            mu_0, sigma_0, alpha, global_tas
        )
    else:
        mu_tas_est, sigma_tas_est = exponential_function(
            mu_0, sigma_0, alpha, global_tas
        )

    return mu_tas_est, sigma_tas_est

###############################################################################
