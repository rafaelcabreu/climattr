import numpy as np
import xarray as xr

import scipy.stats

from climattr.utils import get_fitted_percentiles

def histogram_plot(
    ax,
    obs: xr.DataArray,
    all: xr.DataArray,
    fit_function,
    **kwargs) -> None:
    """
    Plot histograms for observed and model data, including their probability 
    density functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The plot axis on which to draw the histograms.
    
    obs : xr.DataArray
        The observed data.
    
    all : xr.DataArray
        The model or "all" scenario data.
    
    fit_function : statistical function
        A statistical function used to fit the data and calculate the probability 
        density function.

    Returns
    -------
    None
        The function adds the histogram and line plot to the provided axis and 
        does not return anything.
    """
    all_array = all.to_numpy().flatten()
    obs_array = obs.to_numpy().flatten()

    params_all = fit_function.fit(all_array, loc=all_array.mean(), scale=all_array.std())
    params_obs = fit_function.fit(obs_array, loc=obs_array.mean(), scale=obs_array.std())

    # getting the kwargs
    all_color = kwargs.get('all_color', 'C1')
    obs_color = kwargs.get('obs_color', 'k')

    ax.hist(all_array, color=all_color, alpha=0.5, density=True, label='ALL')
    ax.hist(obs_array, color=obs_color, alpha=0.5, density=True, label='OBS')

    percentiles = np.linspace(0.01, 99.9, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_obs = get_fitted_percentiles(percentiles, params_obs, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color=all_color, lw=2)
    ax.plot(x_obs, fit_function.pdf(x_obs, *params_obs), color=obs_color, lw=2)

    ax.legend()

###############################################################################

def qq_plot(
    ax,
    obs: xr.DataArray,
    all: xr.DataArray,
    **kwargs) -> None:
    """
    Generate a quantile-quantile plot to compare quantiles of observed data 
    against model data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The plot axis on which to draw the QQ plot.
    
    obs : xr.DataArray
        The observed data.
    
    all : xr.DataArray
        The model or simulated data.

    Returns
    -------
    None
        The function does not return anything; it directly modifies the provided axis.
    """
    percentiles = np.arange(1,101,1)

    all_array = all.to_numpy().flatten()
    obs_array = obs.to_numpy().flatten()

    all_percentiles = np.percentile(all_array, percentiles)
    obs_percentiles = np.percentile(obs_array, percentiles)

    # getting the kwargs
    color = kwargs.get('color', 'C1')

    ax.plot(obs_percentiles, all_percentiles, marker='o', ls='', color=color)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # Add a reference line
    min_val = min(xlims[0], ylims[0])
    max_val = max(xlims[1], ylims[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="Reference Line")

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

###############################################################################

def qq_plot_theoretical(
    ax,
    data: xr.DataArray,
    fit_function,
    **kwargs) -> None:
    """
    Generate a theoretical quantile-quantile plot to compare data quantiles against 
    a fitted theoretical distribution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The plot axis on which to draw the theoretical QQ plot.
    data : xr.DataArray
        The data array from which quantiles are calculated.
    fit_function : statistical function
        A statistical function used to fit the data and estimate theoretical quantiles.

    Returns
    -------
    None
        The function does not return anything; it directly modifies the provided axis.
    """
    data_array = data.to_numpy().flatten()
    percentiles = scipy.stats.percentileofscore(
        data_array,
        data_array
    )

    # getting the kwargs
    color = kwargs.get('color', 'C1')

    params = fit_function.fit(data_array, loc=data_array.mean(), scale=data_array.std())
    theor_percentiles = get_fitted_percentiles(percentiles, params, fit_function)

    ax.plot(theor_percentiles, data_array, marker='o', ls='', color=color)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # Add a reference line
    min_val = min(xlims[0], ylims[0])
    max_val = max(xlims[1], ylims[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="Reference Line")

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

###############################################################################
