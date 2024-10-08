import numpy as np
import pandas as pd
from typing import List
import xarray as xr

import matplotlib.pyplot as plt
import scipy.stats

from climattr.attribution import _rp_plot_data
from climattr.utils import find_nearest
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
            ax.axhline(row['tx_max'], color='r', ls='--')
            ax.text(
                dataframe['time'].min(), 
                row['tx_max'], 
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

#####################################################################

def rp_plot(
    ax,
    data: xr.DataArray,
    fit_function,
    highlight_year: int | None = 1999,
    direction: str = 'descending',
    bootstrap_ci: int = 95,
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
        idx = find_nearest(thresh, data_array)
        ymin, ymax = ax.get_ylim()
        ax.axvspan(
            conf_rp_inf[idx], conf_rp_sup[idx], 
            ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
            facecolor='silver', edgecolor='C0',
            linewidth=2., alpha=0.3, zorder=0
        )

###############################################################################
