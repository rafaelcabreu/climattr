import numpy as np
import pandas as pd
from typing import List
import xarray as xr

import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats

from climattr.attribution import _rp_plot_data
from climattr.utils import (
    get_xy_coords,
    find_nearest, 
    get_percentiles_from_ci,
    add_features,
    calculate_anomalies
)
from climattr.validator import (
    validate_ci,
    validate_direction
)

def timeseries_plot(
    ax: plt.Axes, 
    data: xr.DataArray,
    linear_regression: bool = True,
    highlight_year: int | None = 1999,
    percentiles: List[int] | None = [1, 5, 90, 95]) -> None:
    """
    Plot a time series on the given axis with optional linear regression, 
    highlighted year, and percentile lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the time series.
    
    data : xr.DataArray
        The data array containing the time series data to be plotted.
    
    linear_regression : bool, optional, default = True
        If True, a linear regression line is plotted on the time series.
    
    highlight_year : int or None, optional, default = 1999
        The specific year to be highlighted on the plot. If None, no year 
        is highlighted.
    
    percentiles : List[int] or None, optional, default = [1, 5, 90, 95]
        List of percentiles to be plotted as horizontal lines on the graph. 
        If None, no percentile lines are plotted.

    Returns
    -------
    None
    """
    dataframe = data.to_dataframe().reset_index()
    dataframe.plot(ax=ax, x='time', y=data.name, legend=False)

    if linear_regression:
        x = np.arange(dataframe.shape[0])
        y = dataframe[data.name].values

        res = scipy.stats.linregress(x, y)
        dataframe['linear_fit'] = res.intercept + res.slope * x

        dataframe.plot(ax=ax, x='time', y='linear_fit', color='k', legend=False)

    if percentiles:
        quantiles = dataframe[data.name].quantile(
            [p / 100 for p in percentiles]
        ).to_frame()

        for _, row in quantiles.iterrows():
            ax.axhline(row[data.name], color='r', ls='--')
            ax.text(
                dataframe['time'].min(), 
                row[data.name], 
                f'{int(_ * 100):02d}%', 
                color='r', 
                va='bottom'
            )

    if highlight_year:
        dataframe['year'] = dataframe['time'].dt.year
        dataframe_year = dataframe.loc[
            dataframe['year'] == highlight_year, [data.name, 'time']
        ]

        dataframe_year.plot(ax=ax, x='time', y=data.name, legend=False, marker='o', color='r')
        ax.text(
            dataframe_year['time'].iloc[0], 
            dataframe_year[data.name].values[0], 
            highlight_year, 
            color='r', 
            va='bottom'
        )       

###############################################################################

def rp_plot(
    ax,
    data: xr.DataArray,
    fit_function,
    highlight_year: int | None = 1999,
    direction: str = 'descending',
    bootstrap_ci: int | None = 95,
    boot_size: int = 1000) -> None:
    """
    Plot a return period graph on the given axis with optional highlighting 
    of a specific year and confidence intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the return period.
    
    data : xr.DataArray
        The data array containing the observations to be used for the plot.
    
    fit_function : function
        The function used to fit the return period distribution.
    
    highlight_year : int or None, optional, default = 1999
        The specific year to be highlighted on the plot. If None, no year 
        is highlighted.
    
    direction : str, optional, default = 'descending'
        The direction of the data ordering for the return period plot. 
        It can be 'ascending' or 'descending'.
    
    bootstrap_ci : int, optional, default = 95
        The confidence interval percentage for the bootstrap method.
    
    boot_size : int, optional, default = 1000
        The number of bootstrap samples to be used.

    Returns
    -------
    None
    """
    # validation steps
    validate_direction(direction)
    validate_ci(bootstrap_ci)

    dataframe = data.to_dataframe().reset_index() 
    data_array = np.sort(dataframe[data.name].values.flatten())

    if direction == 'descending':
        data_array = data_array[::-1]

    conf_rp_inf, conf_rp_sup = _rp_plot_data(
        data_array, fit_function, 'C0', 'OBS', ax, direction, bootstrap_ci, boot_size
    )

    if highlight_year:
        dataframe['year'] = dataframe['time'].dt.year
        dataframe_year = dataframe.loc[
            dataframe['year'] == highlight_year, [data.name, 'time']
        ]
  
        thresh = dataframe_year[data.name].iloc[0]
        ax.axhline(thresh, color='r', ls='--')
        ax.text(
            1, 
            dataframe_year[data.name].iloc[0], 
            f'th = {dataframe_year[data.name].iloc[0]:.3f}', 
            color='r', 
            va='bottom'
        ) 

        # add return period estimate for OBS
        if bootstrap_ci:
            idx = find_nearest(thresh, data_array)
            ymin, ymax = ax.get_ylim()
            ax.axvspan(
                conf_rp_inf[idx], conf_rp_sup[idx], 
                ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
                facecolor='silver', edgecolor='C0',
                linewidth=2., alpha=0.3, zorder=0
            )

###############################################################################

def climatology_timeseries_plot(
    ax: plt.Axes, 
    data: xr.DataArray,
    idate: datetime = '1980-01-01', 
    edate: datetime = '2010-12-31', 
    resample: str = 'month', 
    confidence_interval: int = 95, 
    years: tuple = (2024,)) -> None:
    """
    Plot a climatological time series with confidence intervals and optional 
    highlighted years.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the climatological time series.
    
    data : xr.DataArray
        The data array containing the time series data to be analyzed.
    
    idate : datetime, optional, default = '1980-01-01'
        The start date for computing the climatology.
    
    edate : datetime, optional, default = '2010-12-31'
        The end date for computing the climatology.
    
    resample : str, optional, default = 'month'
        The resampling frequency for computing the climatology 
        (e.g., 'day', 'month', 'year').

    confidence_interval : int, optional, default = 95
        The confidence interval percentage for shading around the 
        climatological mean.

    years : tuple of int, optional, default = (2024,)
        The specific years to be highlighted in the plot.

    Returns
    -------
    None
    """
    dataframe = data.to_dataframe().reset_index().set_index('time')

    ci_inf, ci_sup = get_percentiles_from_ci(confidence_interval)

    dataframe_clim = dataframe[idate:edate].groupby(
        getattr(dataframe[idate:edate].index, resample)
    ).mean()
    
    # confidence interval
    dataframe_cinf = dataframe.groupby(
        getattr(dataframe.index, resample)
    ).quantile(ci_inf / 100)
    dataframe_csup = dataframe.groupby(
        getattr(dataframe.index, resample)
    ).quantile(ci_sup / 100)

    dataframe_clim[data.name].plot(ax=ax, label='Climatology')
    ax.fill_between(
        dataframe_clim.index, 
        dataframe_cinf[data.name], 
        dataframe_csup[data.name], 
        alpha=0.2
    )

    for year in years:
        highlight_year = dataframe.loc[f'{year}-01-01':f'{year}-12-31']
        highlight_year.index = highlight_year.index.map(lambda x: getattr(x, resample))
        highlight_year[data.name].plot(ax=ax, label=year)

###############################################################################

def anomaly_map_plot(
    data: xr.DataArray,
    idate: datetime,
    edate: datetime,
    col_wrap=4,
    **kwargs) -> None:        
    """
    Plot anomaly maps for a range of dates that can be either standardized or not.

    Parameters
    ----------    
    data : xr.DataArray
        The data array containing the time series data to be analyzed.
    
    idate : datetime, 
        The start date for plotting the anomalies.
    
    edate : datetime, 
        The end date for plotting the anomalies.
    
    col_wrap : str, optional, default = 4
        Number of plots in each lines. Default is 4

    Returns
    -------
    None
    """    
    x, y = get_xy_coords(data)

    # extract the keyword arguments
    cbar_kwargs = kwargs.pop('cbar_kwargs', {'shrink': 0.5})
    anomaly_kwargs = kwargs.pop(
        'anomaly_kwargs', 
        {'idate': '1980-01-01', 'edate': '2010-12-31', 'standardize': False}
    )
    cmap = kwargs.pop('cmap', 'RdBu_r')

    # calculate the anomalies
    data_anomaly = calculate_anomalies(
        data,
        idate=anomaly_kwargs['idate'],
        edate=anomaly_kwargs['edate'],
        standardize=anomaly_kwargs['standardize']
    )

    plots = data_anomaly.sel(time=slice(idate, edate)).plot(
        x=x, y=y, col='time', col_wrap=col_wrap, robust=True, cmap=cmap,
        subplot_kws={"projection": ccrs.PlateCarree()}, cbar_kwargs=cbar_kwargs
    )

    # add features to the plots, like states, countries, coastlines, etc.
    for ax in plots.axs.flat:
        add_features(
            ax, 
            extent=[data[x].min(), data[x].max(), data[y].min(), data[y].max()],
            labels=False,
            states_color='k',
            country_color='k',
            coastlines_color='k',
        )

    return None

###############################################################################
