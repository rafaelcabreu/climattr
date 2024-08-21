import xarray as xr

from climattr.utils import get_percentiles_from_ci

def timeseries_plot(
    ax, 
    data: xr.DataArray):

    dataframe = data.to_dataframe().reset_index()

    dataframe.plot(ax=ax, x='time', y=data.name)

#####################################################################

def climatology_plot(
    ax, 
    data: xr.DataArray,
    confidence_interval: int = 95):

    ci_inf, ci_sup = get_percentiles_from_ci(confidence_interval)

    dataframe = data.to_dataframe().reset_index()
    dataframe['month'] = dataframe['time'].dt.month

    climatology = dataframe.groupby('month').mean()

    percentiles = dataframe.groupby('month').quantile([ci_inf / 100, ci_sup / 100])
    percentiles = percentiles.reset_index()

    climatology[data.name].plot(ax=ax, lw=2)
    ax.fill_between(
        percentiles['month'].unique(),
        percentiles[data.name][percentiles['level_1'] == ci_inf / 100],
        percentiles[data.name][percentiles['level_1'] == ci_sup / 100],
        alpha=0.5
    )

    # dataframe.plot(ax=ax, x='time', y=data.name)

#####################################################################
