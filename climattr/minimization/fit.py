import warnings

import numpy as np
import pandas as pd

from datetime import datetime
from scipy import stats
from typing import List, Union, Dict

from climattr.minimization import likelihood
from climattr.minimization.utils import (
    exponential_function,
    linear_function
)


def fit_data(
    x: pd.Series, 
    global_tas: pd.Series, 
    fit_function_name: str, 
    strategy: str,
    verbose: bool = True) -> np.ndarray:
    """
    Fits a statistical model to the provided data using a specified distribution 
    and parameterization strategy.

    This function fits a statistical model to the observed data `x` by modeling 
    the relationship between the data and global temperature anomalies `global_tas`. 
    It supports fitting using different statistical distributions and 
    parameterization strategies to account for potential changes in the 
    distribution parameters with respect to the global temperature anomalies.

    Parameters
    ----------
    x : pd.Series
        The observed data to be fitted.
    global_tas : pd.Series
        Time series of global temperature anomalies corresponding to the data `x`.
    fit_function_name : str
        The name of the distribution to fit. Supported options are:
        - `'genextreme'`: Generalized Extreme Value distribution.
        - `'norm'`: Normal (Gaussian) distribution.
        - `'gamma'`: Gamma distribution.
    strategy : str
        The parameterization strategy to use for modeling how the distribution 
        parameters change with `global_tas`. Supported options are:
        - `'linear'`: Parameters change linearly with `global_tas`.
        - `'exponential'`: Parameters change exponentially with `global_tas`.
    verbose : bool, optional
        If `True`, prints a summary of the fit results. Default is `True`.

    Returns
    -------
    np.ndarray
        An array containing the fitted model parameters. The specific parameters 
        returned depend on the chosen distribution and strategy.
    """
    fit_functions = {
        'genextreme': likelihood.GEVModel,
        'norm': likelihood.NormModel,
        'gamma': likelihood.GammaModel,
        'genpareto': likelihood.GPDModel
    }

    model = fit_functions[fit_function_name](x, global_tas, strategy=strategy)

    # Fit the model
    if verbose:
        result = model.fit(disp=True)
        print(result.summary())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False)

    return result.params

###############################################################################

def extrapolate_data(
    global_tas: pd.Series, 
    params: List[float], 
    date: Union[str, datetime], 
    fit_function_name: str, 
    strategy: str, 
    size: int = 500) -> np.ndarray:
    """
    Extrapolates data for a future time point based on the fitted model parameters 
    and global temperature anomaly.

    Parameters:
    -----------
    global_tas : pd.Series
        Time series of global temperature anomalies.
    params : List[float]
        The fitted parameters [mu_0, sigma_0, alpha, xi].
    date : Union[str, datetime]
        The date for which the extrapolation is to be made.
    fit_function_name : str
        The name of the distribution to use ('genextreme', 'norm', 'gamma').
    strategy : str
        The strategy to use ('linear' or 'exponential') for computing the estimated 
        parameters.
    size : int, optional
        The number of extrapolated data points to generate (default is 500).

    Returns:
    --------
    np.ndarray
        A NumPy array of extrapolated data points generated from the specified 
        distribution.
    """
    if strategy == 'linear':
        mu, sigma = linear_function(
            params[0], params[1], params[3], global_tas.loc[date, 'tas']
        )
    else:
        mu, sigma = exponential_function(
            params[0], params[1], params[3], global_tas.loc[date, 'tas']
        )
    
    function_params = {
        'norm': [mu, sigma],
        'genextreme': [params[2], mu, sigma],
        'gamma': [params[2], mu, sigma],
        'genpareto': [params[2], mu, sigma]
    }

    extrapolated_data = getattr(
        stats, fit_function_name
    ).rvs(*function_params[fit_function_name], size=size)   

    return function_params[fit_function_name], extrapolated_data

###############################################################################

def aic(log_likelihood: float, k: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC).

    Parameters
    ----------
    log_likelihood : float
        The maximized log-likelihood of the fitted model.
    k : int
        The number of estimated parameters in the model.

    Returns
    -------
    float
        The AIC value. Lower values indicate a better trade-off
        between goodness-of-fit and model complexity.
    """
    return 2 * k - 2 * log_likelihood

###############################################################################

def bic(log_likelihood: float, k: int, n: int) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    log_likelihood : float
        The maximized log-likelihood of the fitted model.
    k : int
        The number of estimated parameters in the model.
    n : int
        The number of observations in the dataset.

    Returns
    -------
    float
        The BIC value. Lower values indicate a better trade-off
        between goodness-of-fit and model complexity, with a stronger
        penalty for complexity than AIC.
    """
    return k * np.log(n) - 2 * log_likelihood

###############################################################################

def rsquared(result) -> float:
    """
    Calculate the R² (coefficient of determination) from a fitted WWA model.

    Computes R² as the proportion of variance in the observed data
    explained by the covariate-dependent mean (mu) of the fitted model.

    Parameters
    ----------
    result : statsmodels GenericLikelihoodModelResults
        The result object returned by model.fit().

    Returns
    -------
    float
        The R² value between 0 and 1, where 1 indicates a perfect fit.
    """
    from climattr.minimization.utils import choose_strategy

    mu_0, sigma_0, _, alpha = result.params
    model = result.model
    global_tas = model.exog[:, 0]

    mu_fitted, _ = choose_strategy(model.strategy, mu_0, sigma_0, alpha, global_tas)

    ss_res = np.sum((model.endog - mu_fitted) ** 2)
    ss_tot = np.sum((model.endog - np.mean(model.endog)) ** 2)

    return 1 - ss_res / ss_tot

###############################################################################
