import iris
import numpy as np
import pandas as pd
import re
import requests
import xarray as xr

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime
from glob import glob
import iris.cube
from typing import Union, List

from climattr.validator import validate_ci

def add_features(
    ax: cartopy.mpl.geoaxes.GeoAxes, 
    extent: Union[None, List] = None, 
    states: bool = True, 
    labels: bool =True, 
    shapename: Union[None, str] = None, 
    countries: bool = True, 
    **kwargs) -> cartopy.mpl.geoaxes.GeoAxes:
    """
    Add geographical features like countries, states, labels, and a custom 
    shapefile to a Cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The GeoAxes object to which features will be added.
    
    extent : list or None, optional
        The geographical extent to display on the map as [xmin, xmax, ymin, ymax].
    
    states : bool, optional
        Whether to add state/province borders. Default is True.
    
    labels : bool, optional
        Whether to add labels to the map. Default is True.
    
    shapename : str or None, optional
        The path to a shapefile that will be plotted. Default is None.
    
    countries : bool, optional
        Whether to add country borders. Default is True.
    **kwargs
        Additional keyword arguments to customize features, like 'country_color' 
        or 'states_color'.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        The GeoAxes object with added features.
    """
    if countries:
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none')

        # check if the user specified a color for the countries
        if 'country_color' in kwargs.keys():
            ax.add_feature(
                countries, 
                edgecolor=kwargs['country_color'], 
                facecolor='none', 
                linewidth=0.25
            )
        else:
            ax.add_feature(
                countries, 
                edgecolor='#d0d0d0', 
                facecolor='none', 
                linewidth=0.25
            )

    if states:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )

        # check if the user specified a color for the states
        if 'states_color' in kwargs.keys():
            ax.add_feature(
                states_provinces, 
                edgecolor=kwargs['states_color'], 
                facecolor='none', 
                linewidth=0.25
            )
        else:
            ax.add_feature(
                states_provinces, 
                edgecolor='#d0d0d0', 
                facecolor='none', 
                linewidth=0.25
            )

    if extent:
        ax.set_extent(extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if shapename:
        shp = Reader(shapename)
        # check if the user specified a color for the shapefiles
        ax.add_geometries(
            shp.geometries(), 
            ccrs.PlateCarree(), 
            facecolor='none',
            edgecolor='k'
        )
        
    if labels:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='none')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabels_top = False
        gl.ylabels_right = False

    # check if the user specified a color for the coastlines
    if 'coastlines_color' in kwargs.keys():
        ax.coastlines('50m', color=kwargs['coastlines_color'])
    else:
        ax.coastlines('50m', color='#d0d0d0')
    ax.set_xlabel('')
    ax.set_ylabel('')

    return ax

###############################################################################

def find_nearest(
    value: float, 
    data: np.ndarray) -> int:
    """
    Find the index of the nearest value in a numpy array.

    Parameters
    ----------
    value : float
        The value to find in the array.
    
    data : np.ndarray
        The array in which to search for the nearest value.

    Returns
    -------
    int
        The index of the nearest value in the array.
    """
    idx=(np.abs(data - value)).argmin()
    return idx

###############################################################################

def get_percentiles_from_ci(cofidence_interval: int) -> tuple:
    """
    Calculate the lower and upper percentile bounds from a given confidence 
    interval percentage.

    Parameters
    ----------
    confidence_interval : int
        The confidence interval percentage (e.g., 95 for 95% confidence interval).

    Returns
    -------
    tuple
        A tuple containing the lower and upper percentile values.

    Raises
    ------
    ValueError
        If the confidence interval is not a valid percentage.
    """
    validate_ci(cofidence_interval)

    ci_inf = (100 - cofidence_interval) / 2
    ci_sup = 100 - (100 - cofidence_interval) / 2

    return ci_inf, ci_sup

###############################################################################

def get_xy_coords(dataset: xr.Dataset) -> tuple:
    """
    Extract the coordinate names for latitude and longitude from an xarray Dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray Dataset from which to extract the latitude and longitude coordinates.

    Returns
    -------
    tuple
        A tuple (x, y) containing the names of the longitude and latitude coordinates.

    Notes
    -----
    This function assumes the latitude and longitude coordinates are named using common
    conventions ('lat', 'latitude', 'y' for latitude and 'lon', 'longitude', 'x' 
    for longitude).
    """
    latitudes = ['lat', 'latitude', 'y']
    longitudes = ['lon', 'longitude', 'x']

    for coord in dataset.coords.items():
        
        if coord[0] in latitudes:
            y = coord[0]

        if coord[0] in longitudes:
            x = coord[0]

    return x, y

###############################################################################

def multiens_netcdf(file_path: str, **kwargs) -> xr.Dataset:
    """
    Open multiple NetCDF files representing different model ensemble members and
    combine them into a single xarray Dataset.

    Parameters
    ----------
    file_path : str
        A file path pattern that matches multiple model output files.
    
    **kwargs
        Additional keyword arguments passed to xarray.open_mfdataset.

    Returns
    -------
    xr.Dataset
        A single xarray Dataset containing data from multiple ensemble members,
        concatenated along a new 'ensemble' dimension.

    Notes
    -----
    Assumes that filenames contain ensemble identifiers matching the pattern 
    'r\d+i\d+p\d+f\d+' and that all files corresponding to a single ensemble 
    should be combined.
    """
    ifiles = glob(file_path)

    try:
        ensemble_pattern = r'r\d+i\d+p\d+f\d+'
        ensembles = np.unique([
            re.search(ensemble_pattern, ifile).group() for ifile in ifiles
        ])
    except AttributeError:
        ensemble_pattern = r'r\d+i\d+p\d+'
        ensembles = np.unique([
            re.search(ensemble_pattern, ifile).group() for ifile in ifiles
        ])
    except Exception as e:
        print(f'Could not find ensemble pattern in the files: {e}')

    ds_list = []
    for ensemble in ensembles:
        ds_list.append(
            xr.open_mfdataset(
                [ifile for ifile in ifiles if ensemble in ifile],
                **kwargs
            ).expand_dims({'ensemble': [ensemble]})
        )
    return xr.concat(ds_list, dim='ensemble')

###############################################################################

def get_fitted_percentiles(
    percentiles: np.ndarray, 
    params: tuple, 
    fit_function) -> np.ndarray:
    """
    Calculate the scores at the given percentiles from a fitted statistical function.

    Parameters
    ----------
    percentiles : list of float
        The percentiles for which the scores are calculated.
    params : tuple
        The parameters of the statistical distribution function used to calculate 
        the scores.
    fit_function : callable
        A statistical function that supports the percent point function (ppf).

    Returns
    -------
    np.ndarray
        An array of scores at the specified percentiles.

    Raises
    ------
    ValueError
        If the number of parameters is greater than three or the fitting is 
        otherwise invalid.

    Notes
    -----
    The function handles distributions with up to three parameters (location, 
    scale, and shape).
    """
    if len(params) == 2:
        scores = fit_function.ppf(
            percentiles / 100, loc=params[0], scale=params[1]
        )
    elif len(params) == 3:
        scores = fit_function.ppf(
            percentiles / 100, params[0], loc=params[1], scale=params[2]
        )
    else:
        raise ValueError(
            'Could not fit the given function number of estimated parameters > 3'
        )

    return scores

###############################################################################

def reassign_longitude(
    dataset: xr.Dataset, 
    x: str = 'lon') -> xr.Dataset:

    params = {x: (((dataset[x] + 180) % 360) - 180)}
    dataset = dataset.assign_coords(**params).sortby(x)

    return dataset

###############################################################################

def regrid_dataset(
    dataarray: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
    method: str = "AreaWeighted") -> xr.DataArray:
    """
    Regrids a given xarray Dataset or DataArray based on a target grid (either 
    from another dataset or specified lat/lon coordinates).

    Parameters
    ----------
    dataset : xr.DataArray
        The dataset or data array to be regridded. It must have x and y 
        coordinates that can be interpolated.
                
    lons : np.ndarray
        A NumPy array of longitude values to use as the target grid. This option 
        should be used if `dataset_grid` is not provided.
        
    lats : np.ndarray
        A NumPy array of latitude values to use as the target grid. This option 
        should be used if `dataset_grid` is not provided.

    method : str, optional
        Method for regriding dataset using iris package, options are: Linear,
        AreaWeighted, and Nearest. Default is 'AreaWeighted'. The recommended
        method for precipitation is AreaWeighted which is conservative. 
        
    Returns
    -------
    xr.DataArray
        The regridded dataset or data array. The returned object will have the 
        same type as the input `dataset`.
        
    Examples
    --------
    Regrid a dataset using lat/lon coordinates:
    
    >>> regridded = regrid_dataset(dataset, new_lons, new_lats, method='AreaWeighted')
    """
    # create a target grid
    target_grid = target_grid = iris.cube.Cube(
        np.zeros((len(lats), len(lons))),
        dim_coords_and_dims=[
            (iris.coords.DimCoord(lats, standard_name='latitude', units='degrees'), 0),
            (iris.coords.DimCoord(lons, standard_name='longitude', units='degrees'), 1)]
    )

    # initialize regridding methods
    methods = {
        'Linear': iris.analysis.Linear(),
        'AreaWeighted': iris.analysis.AreaWeighted(),
        'Nearest': iris.analysis.Nearest()
    }

    target_grid.coord("latitude").guess_bounds()
    target_grid.coord("longitude").guess_bounds()

    # convert xarray DataArray to iris cube to regrid data
    dataarray.attrs.pop('standard_name', None)
    dataarray_cube = dataarray.to_iris()
    dataarray_cube.coord("latitude").guess_bounds()
    dataarray_cube.coord("longitude").guess_bounds()

    # regrid using iris package
    regridded_cube = dataarray_cube.regrid(target_grid, methods[method])
    dataarray_regrided = xr.DataArray.from_iris(regridded_cube)
    dataarray_regrided.name = dataarray.name

    # convert back to xarray object
    return dataarray_regrided

###############################################################################

def calculate_anomalies(
    dataarray: xr.DataArray,
    idate: datetime = '1981-01-01',
    edate: datetime = '2010-12-31',
    standardize: bool = False) -> xr.DataArray:
    """
    Computes anomalies for a given xarray DataArray based on a specified 
    reference period.

    Parameters
    ----------
    dataarray : xr.DataArray
        The input dataset containing time-series data with a "time" dimension.

    idate : str, optional
        The start date of the reference period in the format 'YYYY-MM-DD'. 
        Default is '1981-01-01'.
        
    edate : str, optional
        The end date of the reference period in the format 'YYYY-MM-DD'. 
        Default is '2010-12-31'.
        
    standardize : bool, optional
        If True, computes standardized anomalies by dividing the anomaly by the 
        standard deviation of the reference period. Default is False, which 
        returns absolute anomalies.

    Returns
    -------
    xr.DataArray
        An array containing the calculated anomalies, where each value 
        represents the deviation from the monthly climatology of the 
        reference period.

    Examples
    --------
    Compute absolute anomalies based on the 1981-2010 climatology:
    
    >>> anomalies = calculate_anomalies(dataarray, idate="1981-01-01", edate="2010-12-31")
    """
    dataarray = dataarray.load() # load data into memory

    climatology_mean = dataarray.sel(
        time=slice(idate, edate)
    ).groupby("time.month").mean("time")

    # if standardize is true than divide the anomaly by 
    # the standard deviation
    if standardize:
        climatology_std = dataarray.sel(
            time=slice(idate, edate)
        ).groupby("time.month").std("time")

        anomalies = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            dataarray.groupby("time.month"),
            climatology_mean,
            climatology_std,
        )
    else:
        anomalies = xr.apply_ufunc(
            lambda x, m: (x - m),
            dataarray.groupby("time.month"),
            climatology_mean
        )

    return anomalies

###############################################################################

def calculate_anomaly(
    dataframe: pd.DataFrame, 
    itime: datetime = '1981-01-01', 
    etime: datetime = '2010-12-31') -> pd.DataFrame:

    dataframe_anomaly = dataframe.groupby(dataframe.index.month).apply(
        lambda x: x - dataframe[itime:etime].mean()
    )

    # rename because of duplicated names
    dataframe_anomaly.index.names = ['month', 'time']

    # remove unecessary column
    dataframe_anomaly = dataframe_anomaly.reset_index().drop('month', axis=1)
    dataframe_anomaly = dataframe_anomaly.set_index('time')

    return dataframe_anomaly

###############################################################################

def get_global_temperature_anomaly(
    model: str,
    scenario: str,
    itime: Union[str, datetime] = '1981-01-01', 
    etime: Union[str, datetime] = '2010-12-31') -> pd.DataFrame:
    """
    Fetches and processes the global temperature anomaly from a CMIP6 model 
    using the Climate Explorer API.

    Parameters:
    -----------
    model : str
        The name of the climate model (e.g., 'CESM2', 'MPI-ESM1-2-HR').
    scenario : str
        The emission scenario (e.g., 'ssp126', 'ssp585').
    itime : Union[str, datetime], optional
        The start date for the anomaly calculation (default is '1981-01-01').
    etime : Union[str, datetime], optional
        The end date for the anomaly calculation (default is '2010-12-31').

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the monthly global temperature anomalies, with 
        the time index as the end of each month.

    Notes:
    ------
    The function fetches global surface temperature data from the Climate 
    Explorer's KNMI CMIP6 models repository.
    It processes the data, reshapes it, and computes the temperature anomaly 
    for the specified time period.
    """
    baseurl = 'https://climexp.knmi.nl/CMIP6/Tglobal/'

    url = f'{baseurl}/global_tas_mon_{model}_{scenario}_ave.dat'

    # get global average from the models using the climexp site and raise error
    # if not found
    response = requests.get(url)
    response.raise_for_status()

    # get rid of all the header before loading to the dataframe
    lines = response.text.split('\n')
    data = []
    for line in lines:
        if not line.startswith('#'):
            data.append(line.split())

    dataframe = pd.DataFrame(data).dropna()

    # convert month columns into lines and create time column
    dataframe = dataframe.melt(id_vars=0, var_name='time', value_name='tas')
    dataframe['time'] = pd.to_datetime(
        dataframe[0].astype(str) + '-' + dataframe['time'].astype(str) + '-1'
    )
    dataframe = dataframe.drop(0, axis=1)

    # return the labels in the end of the month
    dataframe['tas'] = dataframe['tas'].astype(float)
    dataframe = dataframe.resample('ME', on='time').mean()
    
    dataframe = calculate_anomaly(dataframe, itime=itime, etime=etime)
    dataframe = dataframe.sort_index()

    return dataframe

###############################################################################
