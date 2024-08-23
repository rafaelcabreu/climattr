import numpy as np
import xarray as xr

import scipy.stats

from climattr.utils import get_fitted_percentiles

def histogram_plot(
    ax,
    obs: xr.DataArray,
    all: xr.DataArray,
    fit_function) -> None:

    all_array = all.to_numpy().flatten()
    obs_array = obs.to_numpy().flatten()

    params_all = fit_function.fit(all_array)
    params_obs = fit_function.fit(obs_array)

    ax.hist(all_array, color='C0', alpha=0.5, density=True, label='ALL')
    ax.hist(obs_array, color='k', alpha=0.5, density=True, label='OBS')

    percentiles = np.linspace(0, 100, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_obs = get_fitted_percentiles(percentiles, params_obs, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color='C0', lw=2)
    ax.plot(x_obs, fit_function.pdf(x_obs, *params_obs), color='k', lw=2)

    ax.legend()

###############################################################################

def qq_plot(
    ax,
    obs: xr.DataArray,
    all: xr.DataArray) -> None:

    percentiles = np.arange(1,101,1)

    all_array = all.to_numpy().flatten()
    obs_array = obs.to_numpy().flatten()

    all_percentiles = np.percentile(all_array, percentiles)
    obs_percentiles = np.percentile(obs_array, percentiles)

    ax.plot(obs_percentiles, all_percentiles, marker='o', ls='')

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
    fit_function) -> None:

    data_array = data.to_numpy().flatten()
    percentiles = scipy.stats.percentileofscore(
        data_array,
        data_array
    )

    params = fit_function.fit(data_array)
    theor_percentiles = get_fitted_percentiles(percentiles, params, fit_function)

    ax.plot(theor_percentiles, data_array, marker='o', ls='')

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # Add a reference line
    min_val = min(xlims[0], ylims[0])
    max_val = max(xlims[1], ylims[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="Reference Line")

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

###############################################################################
