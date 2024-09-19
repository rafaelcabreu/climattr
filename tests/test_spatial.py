import numpy as np
import pytest
import rioxarray
import xarray as xr
import geopandas as gpd

from shapely.geometry import box, Polygon

from climattr.spatial import _mask_area, _plot_area, filter_area


def test_mask_area():
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

    dataset = dataset.rio.write_crs("EPSG:4674")
    dataset.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Create a simple square shapefile (GeoDataFrame)
    geom = [box(-5, -5, 5, 5)]
    shapefile = gpd.GeoDataFrame({"geometry": geom}, crs="EPSG:4326")

    # Apply mask
    masked_dataset = _mask_area(dataset, shapefile)

    # Check if the mask was applied correctly
    mask_area = (dataset["lat"] >= -5) & (dataset["lat"] <= 5) & \
        (dataset["lon"] >= -5) & (dataset["lon"] <= 5)
    expected_data = np.where(mask_area, dataset["variable"], np.nan)
    assert np.allclose(masked_dataset["variable"].values, expected_data, equal_nan=True)

###############################################################################

def test_plot_area_mask(monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during tests
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    mock_shapefile = gpd.GeoDataFrame({
        "geometry": [Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)])]}, 
        crs="EPSG:4326"
    )
    monkeypatch.setattr(gpd, "read_file", lambda _: mock_shapefile)

    mask_path = "path/to/fake/shapefile.shp"

    # Mock the add_features function and plt.subplots to prevent actual plotting
    with pytest.raises(ValueError):
        _plot_area(spatial_sel='invalid')

    with pytest.raises(ValueError):
        _plot_area(spatial_sel=None)

    _plot_area(spatial_sel='mask', mask=mask_path)

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
