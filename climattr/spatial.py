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
    """
    Apply a geographical mask from a shapefile to a given xarray Dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray Dataset to be masked.
    
    shapefile : gpd.GeoDataFrame
        A GeoDataFrame containing the geometries to be used for masking the dataset.

    Returns
    -------
    xr.Dataset
        The masked xarray Dataset where data outside the geometries are set to NaN.

    Notes
    -----
    This function assumes that the dataset includes latitude and longitude dimensions.
    """
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
    """
    Plot a spatial area based on the specified selection criteria, either mask or box.

    Parameters
    ----------
    spatial_sel : str
        The type of spatial selection to plot ('mask' or 'box').
    
    mask : str or None, optional
        The filepath to the shapefile used for the mask if spatial_sel is 'mask'.
    
    box : list or None, optional
        The geographical extent [xmin, xmax, ymin, ymax] if spatial_sel is 'box'.

    Raises
    ------
    ValueError
        If no valid spatial selection is provided.

    Notes
    -----
    This function uses Cartopy for plotting and requires a specific projection.
    """
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
    mask: None | str = None,
    box: None | List = None,
    plot_area: bool = False
    ):
    """
    Filter the dataset based on a geographical mask or bounding box and optionally 
    plot the result.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray Dataset to filter.
    
    mask : str or None, optional
        The filepath to a shapefile used to mask the dataset.
    
    box : list or None, optional
        A list defining the geographical bounds [xmin, xmax, ymin, ymax].
    
    plot_area : bool, optional
        If True, plots the filtered area using the '_plot_area' function.

    Returns
    -------
    xr.Dataset
        The filtered xarray Dataset.

    Raises
    ------
    ValueError
        If neither a box nor a mask is specified, or both are specified.

    Notes
    -----
    This function either applies a geographical mask or selects a subset of the 
    dataset based on the provided bounding box coordinates.
    """
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