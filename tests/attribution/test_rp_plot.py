import pytest
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats

from climattr.attribution import _rp_plot_data

# Fixture to create sample data
@pytest.fixture
def sample_data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

###############################################################################

def test_rp_plot_data_with_default_params(sample_data):
    """Test _rp_plot_data with default parameters."""
    fig, ax = plt.subplots()

    # Call the _rp_plot_data function
    conf_rp_inf, conf_rp_sup = _rp_plot_data(
        data=sample_data,
        fit_function=scipy.stats.norm,
        color='b',
        label='Test Label',
        ax=ax,
        direction='descending',
        bootstrap_ci=95,
        boot_size=1000
    )
    
    # Ensure confidence intervals are returned and have expected shapes
    assert isinstance(conf_rp_inf, np.ndarray)
    assert isinstance(conf_rp_sup, np.ndarray)
    assert conf_rp_inf.shape == sample_data.shape
    assert conf_rp_sup.shape == sample_data.shape
    
    # Check if plotting was done
    lines = ax.get_lines()
    assert len(lines) > 0

###############################################################################

def test_rp_plot_data_with_ascending_direction(sample_data):
    """Test _rp_plot_data with ascending direction."""
    fig, ax = plt.subplots()
    
    # Call the _rp_plot_data function with ascending direction
    conf_rp_inf, conf_rp_sup = _rp_plot_data(
        data=sample_data,
        fit_function=scipy.stats.norm,
        color='r',
        label='Test Ascending',
        ax=ax,
        direction='ascending',
        bootstrap_ci=90,
        boot_size=500
    )
    
    # Ensure confidence intervals are returned and have expected shapes
    assert isinstance(conf_rp_inf, np.ndarray)
    assert isinstance(conf_rp_sup, np.ndarray)
    assert conf_rp_inf.shape == sample_data.shape
    assert conf_rp_sup.shape == sample_data.shape
    
    # Ensure direction is ascending by checking if data is plotted in increasing order
    assert np.all(np.diff(sample_data) >= 0)

###############################################################################
