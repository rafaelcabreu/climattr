import numpy as np
import geopandas as gpd

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from typing import Union, List

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
