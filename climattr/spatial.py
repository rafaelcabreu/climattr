import geopandas as gpd
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from typing import Union, List

from climattr.utils import (
    add_features,
    get_xy_coords
)

def _mask_area(
    dataset: xr.Dataset, 
    shapefile: gpd.GeoDataFrame) -> xr.Dataset:

    # get coords
    x, y = get_xy_coords(dataset)

    shapes = [geom for geom in shapefile.geometry]

    # Create a mask with the same dimensions as the NetCDF data
    mask = geometry_mask(
        shapes,
        transform=dataset.rio.transform(),
        invert=True,
        out_shape=(dataset.dims[y], dataset.dims[x])
    )

    # Convert the mask to a DataArray
    mask_da = xr.DataArray(
        mask, 
        coords=[dataset[y], dataset[x]], 
        dims=[y, x]
    )

    # Apply the mask to your dataset
    return dataset.where(mask_da)

#####################################################################

def _plot_area(
    spatial_sel: str,
    mask: Union[None, str] = None,
    box: Union[None, List] = None,
    ) -> None:

    if not spatial_sel:
        ValueError("Should first run 'filter_area' before plotting")
    elif spatial_sel == 'mask':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, shapename=mask)
        plt.show()
    elif spatial_sel == 'box':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, extent=box)
        plt.show()

#####################################################################

def filter_area(
    dataset: xr.Dataset, 
    mask: Union[None, str] = None,
    box: Union[None, List] = None,
    plot_area: bool = False
    ):

    # get coords
    x, y = get_xy_coords(dataset)

    # order data to ensure lat and lon coords will be
    # correclty ordered before filtering the data
    dataset = dataset.sortby([x, y])

    # raise error if no option is selected
    if not mask and not box:
        raise ValueError('You should add either a box or a mask argument')
    elif mask and box:
        raise ValueError('You should choose either a box or a mask argument')

    if box:
        spatial_sel = 'box'
        instance = dataset.sel(**{
            y: slice(box[2], box[3]), 
            x: slice(box[0], box[1])
        })

    if mask:
        spatial_sel = 'mask'
        shapefile = gpd.read_file(mask)
        instance = _mask_area(dataset, shapefile) 

    # plot to view filtered area
    if plot_area:
        _plot_area(spatial_sel, box, mask)

    return instance

#####################################################################