import pytest
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt

from climattr.exploratory import timeseries_plot

# Fixture to create a sample xarray DataArray
@pytest.fixture
def sample_dataarray():
    time = pd.date_range("2000-01-01", periods=10, freq="YE")
    values = np.random.rand(10) * 30
    data = xr.DataArray(values, coords=[time], dims="time", name="tx_max")
    return data

###############################################################################

def test_timeseries_plot_with_linear_regression(sample_dataarray):
    """Test timeseries_plot with linear regression enabled"""
    fig, ax = plt.subplots()
    timeseries_plot(ax, sample_dataarray, linear_regression=True, highlight_year=2005)

    # Check that a line was plotted
    lines = ax.get_lines()
    assert len(lines) > 0, "There should be at least one line in the plot"

    # Check for the linear regression line
    assert any(line.get_label() == 'linear_fit' for line in lines)

###############################################################################

def test_timeseries_plot_without_linear_regression(sample_dataarray):
    """Test timeseries_plot without linear regression"""
    fig, ax = plt.subplots()
    timeseries_plot(ax, sample_dataarray, linear_regression=False, highlight_year=2005)

    # Check that a line was plotted
    lines = ax.get_lines()
    assert len(lines) > 0, "There should be at least one line in the plot"

    # Ensure there's no linear regression line
    assert not any(line.get_label() == 'linear_fit' for line in lines)

###############################################################################
