import pandas as pd
import pytest
import numpy as np
import xarray as xr

from datetime import datetime

from climattr.correction import scaling


def test_scaling_additive():
    # Create test data
    times = pd.date_range("2023-01-01", periods=10)
    data_values = np.random.rand(10) * 10  # Random data
    clim_values = np.ones(10) * 5  # Constant climatology

    data = xr.DataArray(data_values, dims="time", coords={"time": times}, name="data")
    clim = xr.DataArray(clim_values, dims="time", coords={"time": times}, name="clim")

    # Expected result
    expected_result = data_values - clim_values.mean()

    # Apply scaling
    idate = datetime(2023, 1, 1)
    edate = datetime(2023, 1, 10)
    result = scaling(data, clim, idate, edate, method="add")

    # Assert that the result is as expected
    np.testing.assert_allclose(result.values, expected_result, rtol=1e-6)

###############################################################################

def test_scaling_multiplicative():
    # Create test data
    times = pd.date_range("2023-01-01", periods=10)
    data_values = np.random.rand(10) * 10  # Random data
    clim_values = np.ones(10) * 5  # Constant climatology

    data = xr.DataArray(data_values, dims="time", coords={"time": times}, name="data")
    clim = xr.DataArray(clim_values, dims="time", coords={"time": times}, name="clim")

    # Expected result
    expected_result = data_values / clim_values.mean()

    # Apply scaling
    idate = datetime(2023, 1, 1)
    edate = datetime(2023, 1, 10)
    result = scaling(data, clim, idate, edate, method="mult")

    # Assert that the result is as expected
    np.testing.assert_allclose(result.values, expected_result, rtol=1e-6)

###############################################################################

def test_scaling_invalid_method():
    # Create test data
    times = pd.date_range("2023-01-01", periods=10)
    data_values = np.random.rand(10) * 10  # Random data
    clim_values = np.ones(10) * 5  # Constant climatology

    data = xr.DataArray(data_values, dims="time", coords={"time": times}, name="data")
    clim = xr.DataArray(clim_values, dims="time", coords={"time": times}, name="clim")

    # Apply scaling with invalid method
    idate = datetime(2023, 1, 1)
    edate = datetime(2023, 1, 10)

    with pytest.raises(ValueError):
        scaling(data, clim, idate, edate, method="invalid")

###############################################################################
