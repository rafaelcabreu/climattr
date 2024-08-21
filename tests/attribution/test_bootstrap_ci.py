import numpy as np
from climattr.attribution import (
    _calc_return_time_confidence
)

def test_return_type_and_shape():
    """Test that the function returns a numpy array with the correct shape."""
    data = np.array([3.2, 1.5, 4.7, 2.8])
    result = _calc_return_time_confidence(data)
    assert isinstance(result, np.ndarray), "The result should be a numpy array."
    assert result.shape == (2, len(data)), f"Expected shape (2, {len(data)}), but got {result.shape}."

###############################################################################

def test_default_parameters():
    """Test the function with default parameters."""
    data = np.array([3.2, 1.5, 4.7, 2.8])
    result = _calc_return_time_confidence(data)
    assert result.shape == (2, len(data)), "The result shape should be (2, n_samples) with default parameters."
    assert np.all(result[0] <= result[1]), "Lower confidence interval values should be less than or equal to upper values."

###############################################################################

def test_ascending_direction():
    """Test the function with the 'ascending' direction."""
    data = np.array([3.2, 1.5, 4.7, 2.8])
    result = _calc_return_time_confidence(data, direction="ascending")
    assert np.all(result[0] <= result[1]), "In ascending order, lower bounds should be <= upper bounds."

###############################################################################

def test_small_bootstrap_size():
    """Test the function with a small bootstrap size."""
    data = np.array([3.2, 1.5, 4.7, 2.8])
    result = _calc_return_time_confidence(data, boot_size=10)
    assert result.shape == (2, len(data)), "The result shape should be (2, n_samples) with a small bootstrap size."

###############################################################################

def test_identical_data():
    """Test the function when the data array contains identical values."""
    data = np.array([2.0, 2.0, 2.0, 2.0])
    result = _calc_return_time_confidence(data)
    assert np.all(result[0] == result[1]), "For identical data, the lower and upper bounds should be equal."

###############################################################################
