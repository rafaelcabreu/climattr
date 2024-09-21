import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd

from datetime import datetime
from shapely.geometry import Polygon

from climattr.filter import (
    _plot_area, 
    filter_area,
    filter_time
)

# Fixture to create a sample xarray Dataset
@pytest.fixture
def sample_dataset():
    time = pd.date_range("2000-01-01", periods=365, freq="D")  # Daily data for 1 year
    values = np.random.rand(365)
    dataset = xr.Dataset(
        {"data_var": (("time",), values)},
        coords={"time": time}
    )
    return dataset

###############################################################################

def test_plot_area_box(monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    box = [-10, 10, -10, 10]

    # Mock the add_features function and plt.subplots to prevent actual plotting
    _plot_area(spatial_sel='box', box=box)

###############################################################################

def test_filter_area_with_mask(monkeypatch):
    # Create a sample dataset with lat/lon coordinates
    lat = np.linspace(-10, 10, 20)
    lon = np.linspace(-20, 20, 40)
    time = np.arange(0, 10)
    data = np.random.rand(len(time), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "variable": (["time", "lat", "lon"], data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    # Mock shapefile reading
    mock_shapefile = gpd.GeoDataFrame({
        "geometry": [Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)])]}, 
        crs="EPSG:4326"
    )
    monkeypatch.setattr(gpd, "read_file", lambda _: mock_shapefile)

    filtered_dataset = filter_area(dataset, mask="fake/path/to/shapefile.shp")

    assert isinstance(filtered_dataset, xr.Dataset)

###############################################################################

def test_filter_area_with_box():
    # Create a sample dataset with lat/lon coordinates
    lat = np.linspace(-10, 10, 20)
    lon = np.linspace(-20, 20, 40)
    time = np.arange(0, 10)
    data = np.random.rand(len(time), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "variable": (["time", "lat", "lon"], data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    box = [-5, 5, -5, 5]
    filtered_dataset = filter_area(dataset, box=box)

    assert isinstance(filtered_dataset, xr.Dataset)
    assert filtered_dataset["lat"].min() >= box[2]
    assert filtered_dataset["lat"].max() <= box[3]
    assert filtered_dataset["lon"].min() >= box[0]
    assert filtered_dataset["lon"].max() <= box[1]

###############################################################################

def test_filter_area_invalid_arguments():
    # Create a sample dataset with lat/lon coordinates
    lat = np.linspace(-10, 10, 20)
    lon = np.linspace(-20, 20, 40)
    time = np.arange(0, 10)
    data = np.random.rand(len(time), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "variable": (["time", "lat", "lon"], data)
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )

    # Test no mask or box provided
    with pytest.raises(ValueError):
        filter_area(dataset)

    # Test both mask and box provided
    with pytest.raises(ValueError):
        filter_area(dataset, mask="fake/path/to/shapefile.shp", box=[-5, 5, -5, 5])

###############################################################################

def test_filter_time_by_range(sample_dataset):
    """Test filter_time by specifying a time range."""
    itime = datetime(2000, 2, 1)
    etime = datetime(2000, 3, 1)
    
    # Call the filter_time function
    filtered_ds = filter_time(sample_dataset, itime=itime, etime=etime)
    
    # Check if time range is correctly filtered
    assert pd.to_datetime(filtered_ds.time.values).min() == np.datetime64(itime)
    assert pd.to_datetime(filtered_ds.time.values).max() == np.datetime64(etime)
    
###############################################################################

def test_filter_time_by_month(sample_dataset):
    """Test filter_time by specifying specific months."""
    months = [1, 2]  # January and February
    
    # Call the filter_time function
    filtered_ds = filter_time(sample_dataset, months=months)
    
    for dim in filtered_ds.dims:
        filtered_ds = filtered_ds.dropna(dim=dim, how="any")

    # Ensure all months in the filtered dataset are within the specified list
    filtered_months = np.unique(filtered_ds['time.month'].values)
    assert np.array_equal(filtered_months, months)

###############################################################################
