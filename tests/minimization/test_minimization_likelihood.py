import pytest
import numpy as np
import pandas as pd
from climattr.minimization.utils import choose_strategy
from climattr.minimization.likelihood import (
    GEVModel,
    NormModel,
    GammaModel,
    GPDModel
)

# Mock data for testing
np.random.seed(42)
endog = np.random.rand(100)  # Random response variable (e.g., extremes)
exog = np.random.rand(100, 1)  # Random explanatory variable (e.g., temperature anomalies)

def test_gev_model_initialization():
    model = GEVModel(endog, exog, strategy='linear')

    assert model.strategy == 'linear'
    assert model.endog is not None
    assert model.exog is not None

###############################################################################

def test_gev_nloglikeobs():
    model = GEVModel(endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that output is not NaN or infinity (valid log-likelihood values)
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gev_model_fit():
    model = GEVModel(endog, exog, strategy='linear')
    
    # Fit the model with the default starting parameters
    result = model.fit()

    # Ensure that the model successfully fits and returns a result object
    assert result is not None
    assert hasattr(result, 'params')  # Ensure the result contains parameters

    # Check that the parameter array has the expected length (mu_0, sigma_0, c, alpha)
    assert len(result.params) == 4

###############################################################################

def test_gev_model_with_constant_exog():
    constant_exog = np.zeros((100, 1))  # Constant exogenous variable (e.g., no temperature anomaly changes)
    model = GEVModel(endog, constant_exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that the log-likelihood is valid and not -inf due to zero anomalies
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gev_model_with_constant_endog():
    
    constant_endog = np.ones(100)
    model = GEVModel(constant_endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])

    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)

    # Ensure that the log-likelihood is valid
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gev_invalid_arguments_in_nloglikeobs():
    model = GEVModel(endog, exog, strategy='linear')

    # Use parameters that should result in invalid log-likelihood values
    # For example, c is too large leading to invalid arg values in nloglikeobs
    invalid_params = np.array([1.0, 0.5, 10.0, 0.2])

    nloglike_values = model.nloglikeobs(invalid_params)

    # Ensure that the log-likelihood returns -inf when invalid arguments are used
    assert np.all(nloglike_values == -np.inf)

###############################################################################

def test_norm_model_initialization():
    model = NormModel(endog, exog, strategy='linear')

    assert model.strategy == 'linear'
    assert model.endog is not None
    assert model.exog is not None

###############################################################################

def test_norm_nloglikeobs():
    model = NormModel(endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that output is not NaN or infinity (valid log-likelihood values)
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_norm_model_fit():
    model = NormModel(endog, exog, strategy='linear')
    
    # Fit the model with the default starting parameters
    result = model.fit()

    # Ensure that the model successfully fits and returns a result object
    assert result is not None
    assert hasattr(result, 'params')  # Ensure the result contains parameters

    # Check that the parameter array has the expected length (mu_0, sigma_0, c, alpha)
    assert len(result.params) == 4

###############################################################################

def test_norm_model_with_constant_exog():
    # Constant exogenous variable (e.g., no temperature anomaly changes)
    constant_exog = np.zeros((100, 1))
    model = NormModel(endog, constant_exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that the log-likelihood is valid and not -inf due to zero anomalies
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_norm_model_with_constant_endog():
     # Constant response variable (e.g., no variation in data)
    constant_endog = np.ones(100)
    model = NormModel(constant_endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])

    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)

    # Ensure that the log-likelihood is valid
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_norm_invalid_arguments_in_nloglikeobs():
    model = NormModel(endog, exog, strategy='linear')

    # Use parameters that should result in invalid log-likelihood values
    # For example, set parameters such that the model would encounter an issue
    invalid_params = np.array([1.0, 0.5, 10.0, 0.2])

    nloglike_values = model.nloglikeobs(invalid_params)

    # Ensure that the log-likelihood returns reasonable values
    assert np.all(np.isfinite(nloglike_values))  # No -inf in the log-likelihood

###############################################################################

def test_gamma_model_initialization():
    model = GammaModel(endog, exog, strategy='linear')

    assert model.strategy == 'linear'
    assert model.endog is not None
    assert model.exog is not None

###############################################################################

def test_gamma_nloglikeobs():
    model = GammaModel(endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)

    print(nloglike_values)
    
    # Ensure that output is valid log-likelihood values
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gamma_model_fit():
    model = GammaModel(endog, exog, strategy='linear')
    
    # Fit the model with default starting parameters
    result = model.fit()

    # Ensure that the model successfully fits and returns a result object
    assert result is not None
    assert hasattr(result, 'params')  # Ensure the result contains parameters

    # Check that the parameter array has the expected length (mu_0, sigma_0, c, alpha)
    assert len(result.params) == 4

###############################################################################

def test_gamma_model_with_constant_exog():
    # Constant exogenous variable (e.g., no temperature anomaly changes)
    constant_exog = np.zeros((100, 1))
    model = GammaModel(endog, constant_exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that the log-likelihood is valid and not inf due to constant anomalies
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gamma_invalid_parameters_in_nloglikeobs():
    model = GammaModel(endog, exog, strategy='linear')

    # Use parameters that should result in invalid log-likelihood values
    # For example, xi <= 0 should return inf
    invalid_params = np.array([1.0, 0.5, 0.1, 0.2])  # Invalid sigma and xi

    nloglike_values = model.nloglikeobs(invalid_params)

    # Ensure that the log-likelihood returns inf when invalid parameters are used
    assert np.all(nloglike_values == - np.inf)

###############################################################################

def test_gpd_model_initialization():
    model = GPDModel(endog, exog, strategy='linear')

    assert model.strategy == 'linear'
    assert model.endog is not None
    assert model.exog is not None

###############################################################################

def test_gpd_nloglikeobs():
    model = GPDModel(endog, exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that output is valid log-likelihood values
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gpd_nloglikeobs_xi_zero():
    model = GPDModel(endog, exog, strategy='linear')

    # Set c such that xi = -c = 0
    params = np.array([1.0, 0.5, 0.0, 0.2])

    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)

    # Ensure that the log-likelihood is valid for the exponential case
    assert np.all(np.isfinite(nloglike_values))

###############################################################################

def test_gpd_nloglikeobs_invalid_t():
    model = GPDModel(endog, exog, strategy='linear')

    # Use test parameters where xi and shifted_data / sigma_tas_est will result in t <= 0
    params = np.array([1.0, 0.5, -10.0, 0.2])  # Large positive c -> xi = -c -> negative xi

    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)

    # Ensure that the log-likelihood returns inf when t <= 0
    assert np.all(nloglike_values == - np.inf)

###############################################################################

def test_gpd_model_fit():
    model = GPDModel(endog, exog, strategy='linear')
    
    # Fit the model with default starting parameters
    result = model.fit()

    print(result.params)

    # Ensure that the model successfully fits and returns a result object
    assert result is not None
    assert hasattr(result, 'params')  # Ensure the result contains parameters

    # Check that the parameter array has the expected length (mu_0, sigma_0, c, alpha)
    assert len(result.params) == 4

###############################################################################

def test_gpd_model_with_constant_exog():
    # Constant exogenous variable (e.g., no temperature anomaly changes)
    constant_exog = np.zeros((100, 1))
    model = GPDModel(endog, constant_exog, strategy='linear')

    # Use arbitrary test parameters
    params = np.array([1.0, 0.5, -0.1, 0.2])
    
    # Compute the negative log-likelihood
    nloglike_values = model.nloglikeobs(params)
    
    # Ensure that the log-likelihood is valid and not inf due to constant anomalies
    assert np.all(np.isfinite(nloglike_values))

###############################################################################
