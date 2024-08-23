import xarray as xr

from datetime import datetime

from climattr.validator import validate_correction_method

def scaling(
    data: xr.DataArray,
    clim: xr.DataArray,
    idate: datetime,
    edate: datetime,
    method: str = 'add') -> xr.DataArray:

    # validate method
    validate_correction_method(method)

    # calculate climate mean
    clim = clim.sel(time=slice(idate, edate))
    clim = clim.to_numpy().flatten().mean()

    if method == 'add':
        scaled_data = data - clim
    else:
        scaled_data = data / clim

    return scaled_data

###############################################################################

