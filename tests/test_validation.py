import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import scipy.stats

from climattr.validation import (
    histogram_plot, 
    qq_plot, 
    qq_plot_theoretical 
)

def test_histogram_plot():
    # Create sample data
    times = pd.date_range("2023-01-01", periods=10)
    lat = np.linspace(-10, 10, 10)
    lon = np.linspace(-20, 20, 10)
    obs_data = np.random.rand(len(times), len(lat), len(lon))
    all_data = np.random.rand(len(times), len(lat), len(lon))

    obs = xr.DataArray(
        obs_data, 
        dims=["time", "lat", "lon"], 
        coords={"time": times, "lat": lat, "lon": lon}, 
        name="obs"
    )
    all = xr.DataArray(
        all_data, 
        dims=["time", "lat", "lon"], 
        coords={"time": times, "lat": lat, "lon": lon}, 
        name="all"
    )

    fit_function = scipy.stats.genextreme  # Using normal distribution for fitting

    # Create a plot
    fig, ax = plt.subplots()
    histogram_plot(ax, obs, all, fit_function)

    # Check that the histogram and lines are plotted
    assert len(ax.patches) > 0  # Histograms are drawn
    assert len(ax.lines) > 0  # Lines are drawn

    # Check legend entries
    assert len(ax.get_legend().get_texts()) == 2  # Should be 'ALL' and 'OBS'

###############################################################################

def test_qq_plot():
    # Create sample data
    times = pd.date_range("2023-01-01", periods=10)
    lat = np.linspace(-10, 10, 10)
    lon = np.linspace(-20, 20, 10)
    obs_data = np.random.rand(len(times), len(lat), len(lon))
    all_data = np.random.rand(len(times), len(lat), len(lon))

    obs = xr.DataArray(
        obs_data, 
        dims=["time", "lat", "lon"], 
        coords={"time": times, "lat": lat, "lon": lon}, 
        name="obs"
    )
    all = xr.DataArray(
        all_data, 
        dims=["time", "lat", "lon"], 
        coords={"time": times, "lat": lat, "lon": lon}, 
        name="all"
    )

    # Create a plot
    fig, ax = plt.subplots()
    qq_plot(ax, obs, all)

    # Check that the QQ plot is drawn
    assert len(ax.lines) > 0  # QQ plot and reference line are drawn

    # Check that the reference line exists
    ref_line = ax.get_lines()[-1]
    assert np.array_equal(ref_line.get_xdata(), ref_line.get_ydata())  # The reference line should be y = x

###############################################################################

def test_qq_plot_theoretical():
    # Create sample data
    times = pd.date_range("2023-01-01", periods=10)
    lat = np.linspace(-10, 10, 10)
    lon = np.linspace(-20, 20, 10)
    data = xr.DataArray(
        np.random.rand(len(times), len(lat), len(lon)), 
        dims=["time", "lat", "lon"], 
        coords={"time": times, "lat": lat, "lon": lon}, 
        name="data"
    )

    fit_function = scipy.stats.norm  # Using normal distribution for fitting

    # Create a plot
    fig, ax = plt.subplots()
    qq_plot_theoretical(ax, data, fit_function)

    # Check that the theoretical QQ plot is drawn
    assert len(ax.lines) > 0  # QQ plot and reference line are drawn

    # Check that the reference line exists
    ref_line = ax.get_lines()[-1]
    assert np.array_equal(ref_line.get_xdata(), ref_line.get_ydata())  # The reference line should be y = x

###############################################################################
