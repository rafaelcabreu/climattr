import pytest
import xarray as xr
import numpy as np
import xclim

from datetime import datetime
from climattr.indice import xclim_indice

# Fixture to create a sample xarray dataset
@pytest.fixture
def sample_dataset():
    time = np.array([
        datetime(2000, 1, 1), 
        datetime(2000, 1, 2), 
        datetime(2000, 1, 3)
    ], dtype='datetime64[ns]')
    tasmax_data = np.array([30, 32, 31], dtype=np.float32)

    ds = xr.Dataset(
        {
            "tasmax": (("time",), tasmax_data, {"units": "degC"}),
        },
        coords={"time": time}
    )
    return ds

###############################################################################

def test_xclim_indice_tx_max(sample_dataset):
    """Test applying the tx_max indicator (maximum temperature)"""
    
    result = xclim_indice(
        dataset=sample_dataset,
        xclim_function="tx_max",
        tasmax=sample_dataset.tasmax,
        freq='YS'
    )

    assert isinstance(result, xr.DataArray)
    assert "tx_max" == result.name

###############################################################################

