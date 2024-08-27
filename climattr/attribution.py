import numpy as np
import pandas as pd
import xarray as xr

from typing import List

from climattr.utils import (
    find_nearest,
    get_percentiles_from_ci,
    get_fitted_percentiles
)
from climattr.validator import (
    validate_direction, 
    validate_ci
)

def _calc_bootstrap_ensemble(
    data: np.ndarray, 
    direction: str = "ascending", 
    boot_size: int = 1000) -> np.ndarray:
    """
    Generates bootstrap ensembles from the input data and sorts them in the 
    specified direction.
    
    This function creates multiple bootstrap samples from the input data, sorts 
    each sample, and optionally reverses the order of the sorted samples based on 
    the specified direction.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data from which 
        bootstrap samples will be drawn.
        
    direction : str, optional
        The direction in which to sort the bootstrap samples. Can be either 
        "ascending" (default) or "descending".
        
    boot_size : int, optional
        The number of bootstrap samples to generate. Default is 1000.
    
    Returns
    -------
    numpy.ndarray
        A 2D array of shape (boot_size, n_samples) where each row is a sorted 
        bootstrap sample drawn from the input data. The samples are sorted in the 
        specified direction.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([3.2, 1.5, 4.7, 2.8])
    >>> result = _calc_bootstrap_ensemble(data, direction="ascending", boot_size=3)
    >>> result
    array([[1.5, 2.8, 3.2, 4.7],
           [1.5, 2.8, 3.2, 4.7],
           [1.5, 3.2, 3.2, 4.7]])

    >>> result = _calc_bootstrap_ensemble(data, direction="descending", boot_size=2)
    >>> result
    array([[4.7, 3.2, 3.2, 1.5],
           [4.7, 3.2, 2.8, 1.5]])

    """
    # Flatten the input data
    n_samples = data.shape[0]
    
    # Generate the bootstrap samples using np.random.choice
    sample_store = np.random.choice(data, (int(boot_size), n_samples), replace=True)
    
    # Sort each row in the sample_store
    sample_store.sort(axis=1)
    
    # Reverse the rows if direction is "descending"
    if direction == "descending":
        sample_store = sample_store[:, ::-1]
    
    return sample_store

###############################################################################

def _calc_return_time_confidence(
    data: np.ndarray, 
    direction: str = "ascending", 
    bootstrap_ci: int = 95, 
    boot_size: int = 100) -> np.ndarray:
    """
    Calculates the confidence intervals for the return time of a dataset using 
    bootstrapping.

    This function generates bootstrap samples from the input data, sorts them in 
    the specified direction, and then calculates the confidence intervals for the 
    return time based on the specified confidence interval percentage.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data for which return 
        time confidence intervals will be calculated.
        
    direction : str, optional
        The direction in which to sort the bootstrap samples. Can be either 
        "ascending" (default) or "descending".
        
    bootstrap_ci : int, optional
        The confidence interval percentage to use for calculating the return time 
        confidence intervals. Default is 95, representing a 95% confidence interval.
        
    boot_size : int, optional
        The number of bootstrap samples to generate. Default is 100.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (2, n_samples) where the first row contains the lower 
        bound and the second row contains the upper bound of the confidence 
        intervals for each sample.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([3.2, 1.5, 4.7, 2.8])
    >>> result = _calc_return_time_confidence(
        data, direction="ascending", bootstrap_ci=95, boot_size=3
    )
    >>> result
    array([[1.5, 2.8, 3.2, 4.7],
           [1.5, 3.2, 3.2, 4.7]])

    >>> result = _calc_return_time_confidence(
        data, direction="descending", bootstrap_ci=90, boot_size=2
    )
    >>> result
    array([[4.7, 3.2, 3.2, 1.5],
           [4.7, 3.2, 2.8, 1.5]])
    """
    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    sample_store = _calc_bootstrap_ensemble(
        data, 
        direction=direction, 
        boot_size=boot_size
    )
    # Calculate the confidence intervals using np.percentile
    conf_inter = np.percentile(sample_store, np.array([ci_inf, ci_sup]), axis=0)
    
    return conf_inter

###############################################################################

def _rp_plot_data(
    data: np.ndarray,
    fit_function,
    color: str,
    label: str,
    ax,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000) -> List[np.ndarray]:
    """
    Plots return period data along with its confidence intervals on a given axis.

    This function generates the return period data from the input data using a 
    specified fit function. It also calculates and plots the confidence intervals 
    for the return periods based on bootstrap sampling.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data for which return 
        periods and confidence intervals will be calculated.
        
    fit_function : callable
        A function that fits the input data to a distribution and calculates the 
        return period.
        
    color : str
        The color to use for plotting the return periods and confidence intervals.
        
    label : str
        The label to use for the plot legend.
        
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to plot the data.
        
    direction : str, optional
        The direction in which to sort the bootstrap samples. Can be either 
        "ascending" or "descending" (default).
        
    bootstrap_ci : int, optional
        The confidence interval percentage to use for calculating the return time 
        confidence intervals. Default is 95.
        
    boot_size : int, optional
        The number of bootstrap samples to generate. Default is 1000.

    Returns
    -------
    List[np.ndarray]
        A list of two arrays containing the lower and upper bounds of the confidence 
        intervals for each sample.
    """
    return_period = np.array([
        _rp_calculation(data, fit_function, i, direction) for i in data]
    )

    conf_data = _calc_return_time_confidence(
        data, 
        direction=direction, 
        boot_size=boot_size, 
        bootstrap_ci=bootstrap_ci
    )
    conf_rp = _calc_return_time_confidence(
        return_period,
        direction='descending',
        boot_size=boot_size,
        bootstrap_ci=bootstrap_ci
    )

    ax.semilogx(
        return_period, data, marker='o', markersize=2,
        linestyle='None', mec=color, mfc=color,
        color=color, fillstyle='full',
        label=label, zorder=2
    )

    # plot the fitted line
    params = fit_function.fit(data)
    x = np.linspace(
        fit_function.ppf(0.001, *params), 
        fit_function.ppf(0.991, *params), 
        700
    )
    if direction == 'descending':
        fitted_rp = np.array([1 / fit_function.sf(i, *params) for i in x])
    else:
        fitted_rp = np.array([1 / fit_function.cdf(i, *params) for i in x])

    ax.semilogx(fitted_rp, x, color=color, lw=2)

    conf_data_inf = conf_data[0,:].squeeze()
    conf_data_sup = conf_data[1,:].squeeze()

    conf_rp_inf = conf_rp[0,:].squeeze()
    conf_rp_sup = conf_rp[1,:].squeeze()

    ax.fill_between(
        return_period, conf_data_inf, conf_data_sup, color=color,
        alpha=0.2,linewidth=1.,zorder=0
    )
    ax.semilogx(
        [return_period, return_period],
        [conf_data_inf, conf_data_sup],
        color=color, linewidth=1., zorder=1
    )
    ax.semilogx(
        [conf_rp_inf, conf_rp_sup],
        [data, data],
        color=color, linewidth=1., zorder=1
    )

    return conf_rp_inf, conf_rp_sup

###############################################################################

def _pr_calculation(
    all_array: np.ndarray, 
    nat_array: np.ndarray, 
    fit_function, 
    thresh: int,
    direction: str = 'descending') -> float:
    """
    Calculates the probability ratio (PR) between two datasets.

    This function fits the input datasets to a specified distribution using the 
    provided fit function and calculates the probability ratio for a given threshold.

    Parameters
    ----------
    all_array : numpy.ndarray
        An array of shape (n_samples,) containing the "all" scenario data.
        
    nat_array : numpy.ndarray
        An array of shape (n_samples,) containing the "natural" scenario data.
        
    fit_function : callable
        A function that fits the input data to a distribution.
        
    thresh : int
        The threshold value for which the probability ratio will be calculated.
        
    direction : str, optional
        The direction in which to calculate the probability ratio. Default is 
        "descending".

    Returns
    -------
    float
        The calculated probability ratio.
    """
    params_all = fit_function.fit(all_array)
    params_nat = fit_function.fit(nat_array)

    if direction == 'descending':
        pr = fit_function.sf(thresh, *params_all) \
            / fit_function.sf(thresh, *params_nat)
    else:
        pr = fit_function.cdf(thresh, *params_all) \
            / fit_function.cdf(thresh, *params_nat)

    return pr

###############################################################################

def _far_calculation(
    all_array: np.ndarray, 
    nat_array: np.ndarray, 
    fit_function, 
    thresh: float) -> float:
    """
    Calculates the Fraction of Attributable Risk (FAR) between two datasets.

    This function computes the FAR, which is a measure of the fraction of risk 
    attributable to a specific factor, by comparing the probability ratio (PR) 
    between the "all" and "natural" scenario datasets.

    Parameters
    ----------
    all_array : numpy.ndarray
        An array of shape (n_samples,) containing the "all" scenario data.
        
    nat_array : numpy.ndarray
        An array of shape (n_samples,) containing the "natural" scenario data.
        
    fit_function : callable
        A function that fits the input data to a distribution.
        
    thresh : float
        The threshold value for which the FAR will be calculated.

    Returns
    -------
    float
        The calculated Fraction of Attributable Risk (FAR).
    """
    return 1 - (1 / _pr_calculation(
        all_array, nat_array, fit_function, thresh
    ))

###############################################################################

def _rp_calculation(
    data: np.ndarray, 
    fit_function, 
    thresh: float,
    direction: str = 'descending') -> float:
    """
    Calculates the return period for a given threshold in the dataset.

    This function fits the input data to a specified distribution and calculates 
    the return period for a given threshold value.

    Parameters
    ----------
    data : numpy.ndarray
        An array of shape (n_samples,) containing the input data.
        
    fit_function : callable
        A function that fits the input data to a distribution.
        
    thresh : float
        The threshold value for which the return period will be calculated.
        
    direction : str, optional
        The direction in which to calculate the return period. Default is "descending".

    Returns
    -------
    float
        The calculated return period for the given threshold.
    """

    params = fit_function.fit(data)

    if direction == 'descending':
        rp = 1 / fit_function.sf(thresh, *params)
    else:
        rp = 1 / fit_function.cdf(thresh, *params)

    return rp

###############################################################################

def attribution_metrics(
    all: xr.DataArray, 
    nat: xr.DataArray, 
    fit_function,
    thresh: float,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000) -> pd.DataFrame:
    """
    Calculate attribution metrics including Probability Ratio (PR), 
    Fraction of Attributable Risk (FAR), and Return Periods (RP) for 
    "ALL" (observed) and "NAT" (natural) climate scenarios.

    The metrics are computed using bootstrapped samples to provide 
    confidence intervals (CI).

    Parameters
    ----------
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes 
        human influences on climate.

    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents 
        the natural climate without human influences.

    fit_function : callable 
        A statistical distribution or fitting function used to model the data.
        
    thresh : float
        The threshold value for which the attribution metrics are calculated.

    direction : str, optional, default = 'descending'
        The direction in which to assess exceedance of the threshold. 
        Can be 'descending' or 'ascending'.

    bootstrap_ci : int, optional, default = 95
        The confidence interval (CI) percentage for bootstrapping.

    boot_size : int, optional, default = 1000
        The number of bootstrap samples to generate.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated metrics (PR, FAR, RP_ALL, RP_NAT) 
        along with their confidence intervals (CI).
    """
    validate_direction(direction)

    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    all_boot = _calc_bootstrap_ensemble(all_array, boot_size=boot_size)    
    nat_boot = _calc_bootstrap_ensemble(nat_array, boot_size=boot_size)

    template_array = np.zeros(boot_size)
    metrics = {
        'PR': template_array.copy(), 
        'FAR': template_array.copy(), 
        'RP_ALL': template_array.copy(), 
        'RP_NAT': template_array.copy()
    }
    for boot in range(int(boot_size)):
        metrics['PR'][boot] = \
            _pr_calculation(
                all_boot[boot], nat_boot[boot], fit_function, thresh, direction
            )
        metrics['FAR'][boot] = \
            _far_calculation(
                all_boot[boot], nat_boot[boot], fit_function, thresh
            )
        metrics['RP_ALL'][boot] = \
            _rp_calculation(
                all_boot[boot], fit_function, thresh, direction
            )
        metrics['RP_NAT'][boot] = \
            _rp_calculation(
                nat_boot[boot], fit_function, thresh, direction
            )

    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    # create empty metrics dataframe
    metrics_result = pd.DataFrame(
        np.zeros((4, 3)), 
        columns=['value', 'ci_inf', 'ci_sup'], 
        index=['PR', 'FAR', 'RP_ALL', 'RP_NAT']
    )

    # fill dataframe with metrics
    for metric_name in ['PR', 'FAR', 'RP_ALL', 'RP_NAT']:
        metrics_result.loc[metric_name, 'value'] = np.median(metrics[metric_name])
        metrics_result.loc[metric_name, 'ci_inf'] = np.percentile(metrics[metric_name], ci_inf)
        metrics_result.loc[metric_name, 'ci_sup'] = np.percentile(metrics[metric_name], ci_sup)

    return metrics_result
        
###############################################################################

def histogram_plot(
    ax,
    all: xr.DataArray,
    nat: xr.DataArray,
    fit_function,
    thresh: float) -> None:
    """
    Plot histograms of the "ALL" and "NAT" scenarios along with their 
    fitted probability density functions (PDFs).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the histogram.
    
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes 
        human influences on climate.
    
    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents 
        the natural climate without human influences.
    
    fit_function : callable
        A statistical distribution or fitting function used to model the data.
    
    thresh : float
        The threshold value, which is plotted as a vertical dashed line.
    
    Returns
    -------
    None
        This function does not return anything; it modifies the provided 
        axes object in-place.
    """
    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    params_all = fit_function.fit(all_array)
    params_nat = fit_function.fit(nat_array)

    ax.hist(all_array, color='C0', alpha=0.5, density=True, label='ALL')
    ax.hist(nat_array, color='C1', alpha=0.5, density=True, label='NAT')

    # fit the requested distribution and plot it as a line
    percentiles = np.linspace(0, 100, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_nat = get_fitted_percentiles(percentiles, params_nat, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color='C0', lw=2)
    ax.plot(x_nat, fit_function.pdf(x_nat, *params_nat), color='C1', lw=2)

    ax.axvline(thresh, color='k', ls='--')
    ax.legend()

###############################################################################

def rp_plot(
    ax,
    all: xr.DataArray,
    nat: xr.DataArray,
    fit_function,
    thresh: float,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000) -> None:
    """
    Plot return periods for the "ALL" and "NAT" scenarios, including 
    confidence intervals (CI) for the bootstrapped return periods.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the return period plot.
    
    all : xr.DataArray
        Data array representing the "ALL" scenario, which includes 
        human influences on climate.
    
    nat : xr.DataArray
        Data array representing the "NAT" scenario, which represents 
        the natural climate without human influences.
    
    fit_function : callable
        A statistical distribution or fitting function used to model the data.
    
    thresh : float
        The threshold value, which is plotted as a horizontal dashed line.
    
    direction : str, optional, default = 'descending'
        The direction in which to assess exceedance of the threshold. 
        Can be 'descending' or 'ascending'.
    
    bootstrap_ci : int, optional, default = 95
        The confidence interval (CI) percentage for bootstrapping.
    
    boot_size : int, optional, default = 1000
        The number of bootstrap samples to generate.

    Returns
    -------
    None
        This function does not return anything; it modifies the provided 
        axes object in-place.
    """
    # validation steps
    validate_direction(direction)
    validate_ci(bootstrap_ci)

    all_array = np.sort(all.to_numpy().flatten())
    nat_array = np.sort(nat.to_numpy().flatten())

    if direction == 'descending':
        all_array = all_array[::-1]
        nat_array = nat_array[::-1]

    conf_rp_inf_all, conf_rp_sup_all = _rp_plot_data(
        all_array, fit_function, 'C0', 'ALL', ax, direction, bootstrap_ci, boot_size
    )
    conf_rp_inf_nat, conf_rp_sup_nat = _rp_plot_data(
        nat_array, fit_function, 'C1', 'NAT', ax, direction, bootstrap_ci, boot_size
    )

    ax.axhline(thresh, color='k', ls='--')

    # add return period estimate for ALL
    idx = find_nearest(thresh, all_array)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(
        conf_rp_inf_all[idx], conf_rp_sup_all[idx], 
        ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
        facecolor='silver', edgecolor='C0',
        linewidth=2., alpha=0.3, zorder=0
    )

    # add return period estimate for NAT
    idx = find_nearest(thresh, nat_array)
    ax.axvspan(
        conf_rp_inf_nat[idx], conf_rp_sup_nat[idx], 
        ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
        facecolor='silver', edgecolor='C1',
        linewidth=2., alpha=0.3, zorder=0
    )

    ax.legend()

###############################################################################
