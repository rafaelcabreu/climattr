import geopandas as gpd
import xarray as xr

import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Union, List
import salem

from climattr.utils import (
    add_features,
    get_xy_coords,
    reassign_longitude
)

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
    if spatial_sel == 'mask':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, shapename=mask)
        plt.show()
    elif spatial_sel == 'box':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, extent=box)
        plt.show()
    else:
        raise ValueError("Invalid spatial selection type. Use 'box' or 'mask'")

#####################################################################

def filter_area(
    dataset: xr.Dataset, 
    mask: None | str = None,
    box: None | List = None,
    plot_area: bool = False,
    reassign_lon: bool = True
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
    if reassign_lon:
        dataset = reassign_longitude(dataset, x)
    
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

        # filter and subset dataset using shapefile with salem package
        instance = dataset.salem.roi(shape=shapefile) 
        instance = instance.salem.subset(shape=shapefile)

    # plot to view filtered area
    if plot_area:
        _plot_area(spatial_sel, box=box, mask=mask)

    return instance

#####################################################################

def filter_time(
    dataset: xr.Dataset,
    itime: Union[None, datetime] = None, 
    etime: Union[None, datetime] = None, 
    months: Union[None, List] = None) -> xr.Dataset:
    """
    Filters an xarray Dataset by time, with options to specify a time range 
    and/or specific months.

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray dataset that contains a 'time' dimension. This dataset 
        will be filtered based on the provided time range and/or months.
    
    itime : datetime or None, optional
        The starting date (inclusive) for the filtering. If set to None, no 
        lower bound on time will be applied. The default is None.
    
    etime : datetime or None, optional
        The ending date (inclusive) for the filtering. If set to None, no 
        upper bound on time will be applied. The default is None.
    
    months : list or None, optional
        A list of integers representing the months (e.g., [1, 2, 12] for January, 
        February, and December) to filter by. If set to None, no month-based 
        filtering will be applied. The default is None.
    
    Returns
    -------
    xr.Dataset
        The filtered xarray Dataset, based on the provided time range and/or 
        specific months.
    """
    if itime and etime:
        dataset = dataset.sel(time=slice(itime, etime))
    if months:
        dataset = dataset.where(dataset['time.month'].isin(months))

    return dataset

#####################################################################
