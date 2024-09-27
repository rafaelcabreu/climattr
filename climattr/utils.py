import numpy as np
import geopandas as gpd
import re
import xarray as xr

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from glob import glob
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
    ensemble_pattern = r'r\d+i\d+p\d+f\d+'
    ifiles = glob(file_path)

    ensembles = np.unique([
        re.search(ensemble_pattern, ifile).group() for ifile in ifiles
    ])

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
    dataset: Union[xr.Dataset, xr.DataArray],
    dataset_grid: Union[None, xr.Dataset, xr.DataArray] = None,
    lons: Union[None, np.ndarray] = None,
    lats: Union[None, np.ndarray] = None) -> Union[xr.Dataset, xr.DataArray]:
    """
    Regrids a given xarray Dataset or DataArray based on a target grid (either 
    from another dataset or specified lat/lon coordinates).

    Parameters
    ----------
    dataset : Union[xr.Dataset, xr.DataArray]
        The dataset or data array to be regridded. It must have x and y 
        coordinates that can be interpolated.
        
    dataset_grid : Union[None, xr.Dataset, xr.DataArray], optional
        An optional dataset or data array that provides the grid onto which the 
        `dataset` should be interpolated. If provided, the regridding will be 
        based on the coordinates (x, y) from this dataset.
        
    lons : Union[None, np.ndarray], optional
        A NumPy array of longitude values to use as the target grid. This option 
        should be used if `dataset_grid` is not provided.
        
    lats : Union[None, np.ndarray], optional
        A NumPy array of latitude values to use as the target grid. This option 
        should be used if `dataset_grid` is not provided.
        
    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        The regridded dataset or data array. The returned object will have the 
        same type as the input `dataset`.
        
    Raises
    ------
    ValueError
        Raised if neither a target dataset (`dataset_grid`) nor latitude/longitude 
        arrays (`lons`, `lats`) are provided.
        Raised if both a target dataset (`dataset_grid`) and latitude/longitude 
        arrays (`lons`, `lats`) are provided simultaneously.
        
    Examples
    --------
    Regrid a dataset using lat/lon coordinates:
    
    >>> regridded = regrid_dataset(dataset, lons=new_lons, lats=new_lats)
    
    Regrid a dataset using another dataset's grid:
    
    >>> regridded = regrid_dataset(dataset, dataset_grid=target_dataset)
    """
    # raise error if no option is selected
    if not dataset_regrided and not lons and not lats:
        raise ValueError('You should add either a dataset to regrid or lat,lon coordinates')
    elif dataset_grid and lons and lats:
        raise ValueError('You should choose either a dataset to regrid or lat,lon coordinates')

    x, y = get_xy_coords(dataset)
    x_grid, y_grid = get_xy_coords(dataset_grid)

    if dataset_grid:
        dataset_regrided = xr.interp(
            **{x: dataset_grid[x_grid], y: dataset_grid[y_grid]}
        )
    else:
        dataset_regrided = xr.interp(**{x: lons, y: lats})

    return dataset_regrided

###############################################################################
