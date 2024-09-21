import pytest
import numpy as np
import pandas as pd
import xarray as xr

import scipy.stats

from climattr.attribution import attribution_metrics

# Set the random seed
np.random.seed(42)

# Fixture to create sample xarray DataArrays
@pytest.fixture
def sample_data():
    time = pd.date_range("2000-01-01", periods=100, freq="D")
    all_array = np.random.normal(10, 2, size=(100,))
    nat_array = np.random.normal(8, 1.5, size=(100,))
    all_data = xr.DataArray(all_array, coords={"time": time}, dims="time")
    nat_data = xr.DataArray(nat_array, coords={"time": time}, dims="time")
    return all_data, nat_data

def test_attribution_metrics(sample_data):
    """Test attribution_metrics function."""
    all_data, nat_data = sample_data
    fit_function = scipy.stats.norm

    # Call the attribution_metrics function
    result = attribution_metrics(
        all=all_data,
        nat=nat_data,
        fit_function=fit_function,
        thresh=9.5,
        direction='descending',
        bootstrap_ci=95,
        boot_size=100
    )

    # Check if the result is a DataFrame with correct shape
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 3)
    
    # Check that the values are correctly filled in the DataFrame
    assert np.isclose(result.loc['PR', 'value'], 5.954527849810254)
    assert np.isclose(result.loc['FAR', 'value'], 0.832060412685886)
    assert np.isclose(result.loc['RP_ALL', 'value'], 1.530539847952145)
    assert np.isclose(result.loc['RP_NAT', 'value'], 8.98900315155591)
