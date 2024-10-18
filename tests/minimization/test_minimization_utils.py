import numpy as np
import pandas as pd
import pytest

from climattr.minimization.utils import (
    linear_function, 
    exponential_function, 
    choose_strategy
)

# Helper data for testing
global_tas = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
mu_0 = 1.0
sigma_0 = 0.5
alpha = 0.2

def test_linear_function():
    mu_tas_est, sigma_tas_est = linear_function(
        mu_0, sigma_0, alpha, global_tas
    )

    # Expected results based on the linear model
    expected_mu = mu_0 + alpha * global_tas
    expected_sigma = sigma_0  # σ does not change with tas

    pd.testing.assert_series_equal(mu_tas_est, expected_mu)
    assert sigma_tas_est == expected_sigma

###############################################################################

def test_exponential_function():
    mu_tas_est, sigma_tas_est = exponential_function(
        mu_0, sigma_0, alpha, global_tas
    )

    # Expected results based on the exponential model
    exp_term = np.exp(alpha * global_tas / mu_0)
    expected_mu = mu_0 * exp_term
    expected_sigma = sigma_0 * exp_term

    pd.testing.assert_series_equal(mu_tas_est, expected_mu)
    pd.testing.assert_series_equal(sigma_tas_est, expected_sigma)

###############################################################################

def test_choose_strategy_linear():
    mu_tas_est, sigma_tas_est = choose_strategy(
        'linear', mu_0, sigma_0, alpha, global_tas
    )

    # Expected results from linear function
    expected_mu, expected_sigma = linear_function(
        mu_0, sigma_0, alpha, global_tas
    )

    pd.testing.assert_series_equal(mu_tas_est, expected_mu)
    assert sigma_tas_est == expected_sigma

###############################################################################

def test_choose_strategy_exponential():
    mu_tas_est, sigma_tas_est = choose_strategy(
        'exponential', mu_0, sigma_0, alpha, global_tas
    )

    # Expected results from exponential function
    expected_mu, expected_sigma = exponential_function(
        mu_0, sigma_0, alpha, global_tas
    )

    pd.testing.assert_series_equal(mu_tas_est, expected_mu)
    pd.testing.assert_series_equal(sigma_tas_est, expected_sigma)

###############################################################################
