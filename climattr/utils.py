import numpy as np
import geopandas as gpd
import re
import xarray as xr

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
        shapefile = gpd.read_file(shapename)
        shapefile.plot(ax=ax)

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

def find_nearest(value, data):
    idx=(np.abs(data - value)).argmin()
    return idx

###############################################################################

def get_percentiles_from_ci(cofidence_interval):

    validate_ci(cofidence_interval)

    ci_inf = (100 - cofidence_interval) / 2
    ci_sup = 100 - (100 - cofidence_interval) / 2

    return ci_inf, ci_sup

###############################################################################

def get_xy_coords(dataset):

    latitudes = ['lat', 'latitude', 'y']
    longitudes = ['lon', 'longitude', 'x']

    for coord in dataset.coords.items():
        
        if coord[0] in latitudes:
            y = coord[0]

        if coord[0] in longitudes:
            x = coord[0]

    return x, y

###############################################################################

def from_cmip6(file_path, **kwargs):
    """Open a NetCDF file and return an instance of CustomDataset."""
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
