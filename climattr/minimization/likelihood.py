import numpy as np
import pandas as pd

from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel

from climattr.minimization.utils import choose_strategy

class GEVModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GEVModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c, alpha = params

        xi = - c

        global_tas = self.exog[:, 0]
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alpha, global_tas
        )

        t = (x - mu_tas_est) / sigma_tas_est
        arg = 1 - xi * t
        if np.any(arg <= 0):
            return -np.inf * np.ones_like(global_tas)

        log_arg = np.log(arg)
        log_likelihood = -np.log(sigma_tas_est) - (1 - 1 / xi) * log_arg - arg ** (1 / xi)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            alpha_start = 0.0
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, alpha_start])

        # Call the superclass fit method
        return super(GEVModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class NormModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(NormModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c, alpha = params

        global_tas = self.exog[:, 0]
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alpha, global_tas
        )
        
        log_likelihood = - np.log(sigma_tas_est ** 2) - (x - mu_tas_est) ** 2 / sigma_tas_est ** 2

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            alpha_start = 0.0
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, alpha_start])

        # Call the superclass fit method
        return super(NormModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class GammaModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GammaModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c, alpha = params

        xi = - c

        global_tas = self.exog[:, 0]
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alpha, global_tas
        )

        if xi <= 0:
            return -np.inf * np.ones_like(global_tas)
        
        # Compute the log-likelihood
        log_likelihood = (xi - 1) * np.log(x) - \
            x / sigma_tas_est - \
            xi * np.log(sigma_tas_est) - \
            gammaln(xi)

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            alpha_start = 0.0
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, alpha_start])

        # Call the superclass fit method
        return super(GammaModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################

class GPDModel(GenericLikelihoodModel):

    def __init__(self, endog, exog, strategy='linear', **kwds):
        self.strategy = strategy

        super(GPDModel, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):

        mu_0, sigma_0, c, alpha = params

        xi = - c

        global_tas = self.exog[:, 0]
        x = self.endog

        mu_tas_est, sigma_tas_est = choose_strategy(
            self.strategy, mu_0, sigma_0, alpha, global_tas
        )

        # Compute shifted data
        shifted_data = x - mu_tas_est

        if xi != 0:
            # Compute the term t = 1 + xi * (x_i - mu) / sigma
            t = 1 + xi * shifted_data / sigma_tas_est

            # Check the constraint t > 0 for all data points
            if np.any(t <= 0):
                return -np.inf * np.ones_like(global_tas)

            # Compute the log-likelihood
            log_likelihood = np.log(sigma_tas_est) - (1 / xi + 1) * t
        else:
            # When xi = 0, GPD reduces to exponential distribution
            log_likelihood = np.log(sigma_tas_est) - (1 / sigma_tas_est) * shifted_data

        return - log_likelihood

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        # Provide starting values if not set
        if start_params is None:
            mu_0_start = np.mean(self.endog)
            sigma_0_start = np.std(self.endog)
            alpha_start = 0.0
            c_start = - 0.1
            start_params = np.array([mu_0_start, sigma_0_start, c_start, alpha_start])

        # Call the superclass fit method
        return super(GPDModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

###############################################################################