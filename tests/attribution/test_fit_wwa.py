import pytest
import xarray as xr
import pandas as pd
import numpy as np

from climattr.attribution import fit_wwa_data

def test_fit_wwa_data():
    # Create a sample xr.DataArray for 'all'
    times = pd.date_range('1900-01-01', periods=120, freq='AS')
    data_values = np.random.rand(120)
    all_data = xr.DataArray(data_values, coords=[times], dims=['time'], name='test_variable')

    # Create a sample pd.DataFrame for 'global_tas'
    tas_times = pd.date_range('1900-01-01', periods=120, freq='AS')
    tas_values = np.random.rand(120)
    global_tas = pd.DataFrame({'tas': tas_values}, index=tas_times)

    # Set other parameters
    fit_function_name = 'norm'
    all_date = '2015-01-01'
    nat_date = '1900-01-01'
    strategy = 'linear'
    verbose = False

    # Set return values for the mocks
    # Create a mock result object for fit_data
    mock_params = np.array([0.601697,  0.283071, -2.347771, -0.240005])

    # Call the function
    params, all_wwa, nat_wwa = fit_wwa_data(
        all=all_data,
        global_tas=global_tas,
        fit_function_name=fit_function_name,
        all_date=all_date,
        nat_date=nat_date,
        strategy=strategy,
        verbose=verbose
    )

    # Assertions to check outputs
    assert isinstance(params, np.ndarray)
    np.testing.assert_allclose(params, mock_params, atol=0.1)

    assert isinstance(all_wwa, xr.DataArray)
    assert all_wwa.name == 'test_variable'

    assert isinstance(nat_wwa, xr.DataArray)
    assert nat_wwa.name == 'test_variable'

###############################################################################
