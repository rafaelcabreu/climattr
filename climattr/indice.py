import xarray as xr
import xclim

def xclim_indice(
    dataset: xr.Dataset,
    xclim_function: str,
    **kwargs) -> xr.Dataset:
    """
    Applies a specified xclim climate indicator function to an xarray dataset, 
    with configurable options for missing data tolerance and validation checks.

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray dataset containing the climate data to be processed. 
        This dataset must have the required variables and metadata for the 
        selected xclim function.
    
    xclim_function : str
        The name of the xclim climate indicator function to apply. The function 
        must be part of the `xclim.indicators.atmos` module.
    
    **kwargs : keyword arguments
        Additional arguments required by the specified xclim function. These 
        should be passed in a key-value format as expected by the chosen indicator 
        function.
    
    Returns
    -------
    xr.Dataset
        The output xarray dataset after applying the xclim climate indicator 
        function, which contains the calculated climate indices or derived metrics.
        
    Examples
    --------
    1. Apply the `tg_mean` (mean temperature) xclim indicator function to a dataset:
    
        >>> result = xclim_indice(
        ...    dataset=ds, xclim_function="tx_max", tasmax='tasmax', freq='YS'
        ... )

    See Also
    --------
    - `xclim.indicators.atmos`: xclim atmospheric indicators module, where 
       available functions can be found.
    
    References
    ----------
    - Xclim Documentation: https://xclim.readthedocs.io/
    
    """
    with xclim.set_options(
        check_missing="pct",
        missing_options={"pct": dict(tolerance=1)},
        cf_compliance="log",
        data_validation='log'
    ):

        xclim_function = getattr(
            xclim.indicators.atmos, 
            xclim_function
        )

        indice = xclim_function(
            **kwargs,
            ds=dataset
        )

    return indice

###############################################################################
