import numpy as np
import re
import xarray as xr

from glob import glob

from climattr.utils import get_xy_coords

@xr.register_dataset_accessor("eea")
class ClimAttr:

    def __init__(self, dataset, *args, **kwargs):
        # initialize custom attributes
        self.dataset = dataset
        self.spatial_sel = None
        self.box = None
        self.mask = None

        self.x, self.y = get_xy_coords(self.dataset)

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
