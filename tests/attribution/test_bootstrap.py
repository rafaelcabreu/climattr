import numpy as np

from climattr.attribution import (
    _calc_bootstrap_ensemble
)

def test_output_shape():
    """Test if the output shape is correct."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    boot_size = 1000
    result = _calc_bootstrap_ensemble(data, boot_size=boot_size)
    assert result.shape == (boot_size, len(data))

###############################################################################

def test_sorting_ascending():
    """Test if sorting direction 'ascending' is applied correctly."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = _calc_bootstrap_ensemble(data, direction="ascending", boot_size=10)
    assert np.all(np.diff(result, axis=1) >= 0)

###############################################################################

def test_sorting_descending():
    """Test if sorting direction 'descending' is applied correctly."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = _calc_bootstrap_ensemble(data, direction="descending", boot_size=10)
    assert np.all(np.diff(result, axis=1) <= 0)

###############################################################################

def test_bootstrap_sampling():
    """Test if the function produces bootstrap samples with replacement."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = _calc_bootstrap_ensemble(data, boot_size=10)
    unique_samples = np.unique(result, axis=0)
    assert unique_samples.shape[0] <= 10  # There should be duplicates

###############################################################################

def test_edge_case_empty_data():
    """Test if the function handles empty data."""
    data = np.array([])
    result = _calc_bootstrap_ensemble(data, boot_size=1000)
    assert result.shape == (1000, 0)

###############################################################################

def test_edge_case_zero_boot_size():
    """Test if the function handles zero boot_size."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = _calc_bootstrap_ensemble(data, boot_size=0)
    assert result.shape == (0, len(data))

###############################################################################
