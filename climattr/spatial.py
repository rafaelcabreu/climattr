import geopandas as gpd
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from typing import Union, List

from climattr.main import ClimAttr
from climattr.utils import add_features

def _mask_area(
    self, 
    shapefile: gpd.GeoDataFrame):

    shapes = [geom for geom in shapefile.geometry]

    # Create a mask with the same dimensions as the NetCDF data
    mask = geometry_mask(
        shapes,
        transform=self.rio.transform(),
        invert=True,
        out_shape=(self.dims[self.y], self.dims[self.x])
    )

    # Convert the mask to a DataArray
    mask_da = xr.DataArray(
        mask, 
        coords=[self[self.y], self[self.x]], 
        dims=[self.y, self.x]
    )

    # Apply the mask to your dataset
    return self.where(mask_da)

#####################################################################

def _plot_area(self) -> None:

    if not self.spatial_sel:
        ValueError("Should first run 'filter_area' before plotting")
    elif self.spatial_sel == 'mask':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, shapename=self.mask)
        plt.show()
    elif self.spatial_sel == 'box':
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        add_features(ax, extent=self.box)
        plt.show()

#####################################################################

def filter_area(
    self, 
    mask: Union[None, str] = None,
    box: Union[None, List] = None,
    plot_area: bool = False
    ):
    
    # raise error if no option is selected
    if not mask and not box:
        raise ValueError('You should add either a box or a mask argument')
    elif mask and box:
        raise ValueError('You should choose either a box or a mask argument')

    if box:
        self.spatial_sel = 'box'
        self.box = box
        instance = self.dataset.sel(**{
            f'{self.y}': slice(box[2], box[3]), 
            f'{self.x}': slice(box[0], box[1])
        })

    if mask:
        self.dataset.spatial_sel = 'mask'
        self.mask = mask
        shapefile = gpd.read_file(mask)
        instance = self._mask_area(shapefile) 

    # plot to view filtered area
    if plot_area:
        self._plot_area()

    return instance

#####################################################################