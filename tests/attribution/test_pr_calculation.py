import pytest
import numpy as np
import xarray as xr
import pandas as pd

import scipy.stats

from climattr.attribution import (
    _pr_calculation,
    _far_calculation,
    _rp_calculation
)


# Fixture to create sample data arrays
@pytest.fixture
def sample_data():
    all_array = np.array([10, 15, 20, 25, 30])
    nat_array = np.array([8, 12, 18, 22, 28])
    return all_array, nat_array

###############################################################################

def test_pr_calculation_descending(sample_data):
    """Test _pr_calculation with 'descending' direction."""
    all_array, nat_array = sample_data
    
    fit_function = scipy.stats.norm

    # Call the _pr_calculation function with descending direction
    pr = _pr_calculation(
        all_array=all_array,
        nat_array=nat_array,
        fit_function=fit_function,
        thresh=20,
        direction='descending'
    )
    
    # Check if the probability ratio is calculated correctly
    expected_pr = fit_function.sf(20, *fit_function.fit(all_array)) \
        / fit_function.sf(20, *fit_function.fit(nat_array))
    assert np.isclose(pr, expected_pr)

###############################################################################

def test_pr_calculation_ascending(sample_data):
    """Test _pr_calculation with 'ascending' direction."""
    all_array, nat_array = sample_data
    fit_function = scipy.stats.norm
    
    # Call the _pr_calculation function with ascending direction
    pr = _pr_calculation(
        all_array=all_array,
        nat_array=nat_array,
        fit_function=fit_function,
        thresh=20,
        direction='ascending'
    )
    
    # Check if the probability ratio is calculated correctly
    expected_pr = fit_function.cdf(20, *fit_function.fit(all_array)) \
        / fit_function.cdf(20, *fit_function.fit(nat_array))
    assert np.isclose(pr, expected_pr)

###############################################################################

def test_far_calculation_descending(sample_data):
    """Test _far_calculation with descending direction."""
    all_array, nat_array = sample_data
    fit_function = scipy.stats.norm
    
    # Call the _far_calculation function
    far = _far_calculation(
        all_array=all_array,
        nat_array=nat_array,
        fit_function=fit_function,
        thresh=20
    )
    
    # FAR should be 1 - (1 / PR), and with PR = 2, FAR = 1 - (1 / 2) = 0.5
    expected_far = 0.2650886075803144
    assert np.isclose(far, expected_far), f"Expected FAR: {expected_far}, but got {far}"

###############################################################################

def test_rp_calculation_descending(sample_data):
    """Test _rp_calculation with descending direction."""
    all_array, _ = sample_data  # Only need one dataset for RP calculation
    fit_function = scipy.stats.norm

    # Call the _rp_calculation function
    rp = _rp_calculation(
        data=all_array,
        fit_function=fit_function,
        thresh=20,
        direction='descending'
    )

    # RP should be 1 / SF(thresh), with SF(thresh) = 1 / (thresh + 1)
    expected_rp = 1 / fit_function.sf(20, *fit_function.fit(all_array))
    assert np.isclose(rp, expected_rp)

###############################################################################
