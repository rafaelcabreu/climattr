import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch, MagicMock

from climattr.utils import (
    add_features,
    find_nearest, 
    get_percentiles_from_ci, 
    get_xy_coords, 
    get_fitted_percentiles
)

def test_add_features():
    ax = MagicMock()  # Mock Cartopy GeoAxes
    ax.set_extent = MagicMock()
    ax.set_xlim = MagicMock()
    ax.set_ylim = MagicMock()
    ax.gridlines = MagicMock()
    ax.coastlines = MagicMock()
    ax.set_xlabel = MagicMock()
    ax.set_ylabel = MagicMock()
    
    # Call add_features with mocked GeoAxes
    result = add_features(ax, extent=[-10, 10, -10, 10], labels=True)
    
    # Assert the functions have been called with the correct parameters
    ax.set_extent.assert_called_once_with([-10, 10, -10, 10])
    ax.set_xlim.assert_called_once_with(-10, 10)
    ax.set_ylim.assert_called_once_with(-10, 10)
    ax.gridlines.assert_called_once()
    ax.coastlines.assert_called_once()
    
    # Ensure the function returns the modified ax object
    assert result == ax

###############################################################################

def test_find_nearest():
    data = np.array([1, 3, 5, 7, 9])
    value = 6
    result = find_nearest(value, data)
    
    # The nearest value to 6 in the array is 5 (index 2)
    assert result == 2

###############################################################################

def test_get_percentiles_from_ci_without_mock():
    result = get_percentiles_from_ci(95)
    assert result == (2.5, 97.5)
    
###############################################################################

def test_get_xy_coords():
    mock_dataset = MagicMock(spec=xr.Dataset)
    
    # Simulate the coordinates of the dataset
    mock_dataset.coords.items.return_value = [('lat', None), ('lon', None)]
    
    x, y = get_xy_coords(mock_dataset)
    
    # Assert the function correctly identifies latitude and longitude
    assert x == 'lon'
    assert y == 'lat'

###############################################################################

def test_get_fitted_percentiles():
    fit_function = MagicMock()
    fit_function.ppf.side_effect = lambda p, loc, scale: np.array([p * loc * scale])
    
    params = (1, 2)  # Example parameters
    percentiles = np.array([10, 50, 90])
    
    result = get_fitted_percentiles(percentiles, params, fit_function)
    
    # Verify that the fit_function's ppf was called
    fit_function.ppf.assert_called()
    
    # Check the resulting scores array has the expected structure
    assert isinstance(result, np.ndarray)

###############################################################################
