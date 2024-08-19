import numpy as np
import re
import xarray as xr

from glob import glob

class ClimAttr(xr.Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize custom attributes
        self.spatial_sel = None
        self.box = None
        self.mask = None

        latitudes = ['lat', 'latitude', 'y']
        longitudes = ['lon', 'longitude', 'x']

        for coord in self.coords.items():
            
            if coord[0] in latitudes:
                self.y = coord[0]

            if coord[0] in longitudes:
                self.x = coord[0]

    #####################################################################
                
    @classmethod
    def from_netcdf(cls, file_path, **kwargs):
        """Open a NetCDF file and return an instance of CustomDataset."""
        ds = xr.open_mfdataset(file_path, **kwargs)
        return cls(ds)

    #####################################################################

    @classmethod
    def from_cmip6(cls, file_path, **kwargs):
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
        return cls(xr.concat(ds_list, dim='ensemble'))

    #####################################################################

    def _copy_attrs_to_new_instance(self, new_instance):
        """Copy custom attributes to the new instance."""
        new_instance.spatial_sel = self.spatial_sel
        new_instance.box = self.box
        new_instance.mask = self.mask
        new_instance.x = self.x
        new_instance.y = self.y
        
        return new_instance

    #####################################################################

    def mean(self, dim=None, keep_attrs=True):
        mean_data = super(self.__class__, self).mean(dim=dim, keep_attrs=keep_attrs)
        # Create a new instance of ClimAttrSpatial and copy attributes
        new_instance = self.__class__(mean_data)
        return self._copy_attrs_to_new_instance(new_instance)
    
    #####################################################################

    def max(self, dim=None, keep_attrs=True):
        max_data = super(self.__class__, self).max(dim=dim, keep_attrs=keep_attrs)
        # Create a new instance of ClimAttrSpatial and copy attributes
        new_instance = self.__class__(max_data)
        return self._copy_attrs_to_new_instance(new_instance)
    
    #####################################################################

    def min(self, dim=None, keep_attrs=True):
        min_data = super(self.__class__, self).min(dim=dim, keep_attrs=keep_attrs)
        # Create a new instance of ClimAttrSpatial and copy attributes
        new_instance = self.__class__(min_data)
        return self._copy_attrs_to_new_instance(new_instance)
    
    #####################################################################
