import xarray as xr

from datetime import datetime

from climattr.validator import validate_correction_method

def scaling(
    data: xr.DataArray,
    clim: xr.DataArray,
    idate: datetime,
    edate: datetime,
    method: str = 'add') -> xr.DataArray:
    """
    Scale the input data by adjusting it based on the climate mean over a 
    specified period. The scaling can be done either by subtracting the mean 
    (additive scaling) or by dividing by the mean (multiplicative scaling).

    Parameters
    ----------
    data : xr.DataArray
        The data to be scaled, typically representing climate variables 
        (e.g., temperature, precipitation).
    
    clim : xr.DataArray
        The climatology data used to calculate the mean for scaling. This 
        should cover the same variable as 'data' over a baseline period.
    
    idate : datetime
        The start date of the period over which the climatology mean is calculated.
    
    edate : datetime
        The end date of the period over which the climatology mean is calculated.
    
    method : str, optional, default = 'add'
        The method of scaling. If 'add', the climate mean is subtracted 
        from the data (additive scaling). If 'mult', the data is divided 
        by the climate mean (multiplicative scaling).

    Returns
    -------
    xr.DataArray
        The scaled data array, with adjustments applied based on the 
        specified method and climatology mean.
    """
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

