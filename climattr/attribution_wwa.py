import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from typing import List, Union

from climattr.attribution import (
    _pr_calculation,
    _far_calculation,
    _rp_calculation,
    _calc_bootstrap_ensemble,
    _rp_plot_data
)
from climattr.minimization import likelihood
from climattr.minimization.fit import (
    fit_data,
    extrapolate_data,
    aic,
    bic,
    rsquared
)
from climattr.utils import (
    find_nearest,
    get_percentiles_from_ci,
    get_fitted_percentiles
)
from climattr.validator import (
    validate_direction, 
    validate_ci
)


def fit_summary(
    all: xr.DataArray,
    global_tas: pd.DataFrame,
    fit_function_name: str,
    strategy: str = 'linear',
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False) -> pd.DataFrame:
    """
    Fit a WWA model and return a summary table with parameters,
    goodness-of-fit metrics (AIC, BIC, R²), and bootstrap confidence intervals.

    Parameters
    ----------
    all : xr.DataArray
        Data array with the climate variable to fit.
    global_tas : pd.DataFrame
        DataFrame with global mean temperature anomalies indexed by time.
    fit_function_name : str
        Name of the distribution to fit ('genextreme', 'norm', 'gamma', 'genpareto').
    strategy : str, optional
        Covariate strategy ('linear' by default).
    bootstrap_ci : int, optional
        Confidence interval percentage (default 95).
    boot_size : int, optional
        Number of bootstrap iterations (default 1000).
    seed : int, optional
        Random seed for reproducibility (default 42).
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores, default -1).
    verbose : bool, optional
        If True, print fit details (default False).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'estimate', 'ci_inf', 'ci_sup' for each
        parameter (mu_0, sigma_0, c, alpha) and goodness-of-fit metric
        (log_likelihood, aic, bic, r2, n_obs).
    """
    import warnings

    all_dataframe = all.to_dataframe().reset_index()
    dataframe = all_dataframe.set_index('time').join(global_tas).dropna()

    fit_functions = {
        'genextreme': likelihood.GEVModel,
        'norm': likelihood.NormModel,
        'gamma': likelihood.GammaModel,
        'genpareto': likelihood.GPDModel
    }

    # Point estimate on full dataset
    model = fit_functions[fit_function_name](dataframe[all.name], dataframe['tas'], strategy=strategy)
    result = model.fit(disp=verbose)

    param_names = ['mu_0', 'sigma_0', 'c', 'alpha']
    k = len(result.params)
    n = int(result.nobs)

    estimates = {
        'mu_0': result.params[0],
        'sigma_0': result.params[1],
        'c': result.params[2],
        'alpha': result.params[3],
        'log_likelihood': result.llf,
        'aic': aic(- result.llf, k),
        'bic': bic(- result.llf, k, n),
        'r2': rsquared(result),
        'n_obs': n,
    }

    # Bootstrap
    indices = np.arange(len(dataframe))
    indices_boot = _calc_bootstrap_ensemble(indices, boot_size=boot_size, seed=seed)

    def compute_fit(boot):
        dataframe_boot = dataframe.iloc[indices_boot[boot]].sort_index().dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_boot = fit_functions[fit_function_name](
                dataframe_boot[all.name], dataframe_boot['tas'], strategy=strategy
            )
            try:
                result_boot = model_boot.fit(disp=False)
            except Exception:
                return None

        k_boot = len(result_boot.params)
        n_boot = int(result_boot.nobs)

        return {
            'mu_0': result_boot.params[0],
            'sigma_0': result_boot.params[1],
            'c': result_boot.params[2],
            'alpha': result_boot.params[3],
            'log_likelihood': result_boot.llf,
            'aic': aic(- result_boot.llf, k_boot),
            'bic': bic(- result_boot.llf, k_boot, n_boot),
            'r2': rsquared(result_boot),
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_fit)(boot) for boot in range(int(boot_size))
    )

    # Filter out failed fits
    results = [r for r in results if r is not None]

    ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

    all_names = param_names + ['log_likelihood', 'aic', 'bic', 'r2', 'n_obs']
    summary = pd.DataFrame(
        np.zeros((len(all_names), 3)),
        index=all_names,
        columns=['estimate', 'ci_inf', 'ci_sup']
    )

    for name in all_names:
        summary.loc[name, 'estimate'] = estimates[name]

    for name in param_names + ['log_likelihood', 'aic', 'bic', 'r2']:
        values = np.array([r[name] for r in results])
        values = values[~np.isnan(values)]
        summary.loc[name, 'ci_inf'] = np.percentile(values, ci_inf)
        summary.loc[name, 'ci_sup'] = np.percentile(values, ci_sup)

    # n_obs has no CI
    summary.loc['n_obs', 'ci_inf'] = np.nan
    summary.loc['n_obs', 'ci_sup'] = np.nan

    return summary

############################################################################### 

def attribution_metrics(
    all: xr.DataArray, 
    global_tas: pd.DataFrame, 
    fit_function_name: str,
    all_date: Union[datetime, str] = '2015-11-30',
    nat_date: Union[datetime, str] = '1900-11-30',
    strategy: str = 'linear',
    direction: str = 'descending',
    bootstrap_ci: int = 95,
    boot_size: int = 1000,
    seed: int = 42,
    summary_statistics: bool = True,
    verbose: bool = False,
    n_jobs: int = -1) -> pd.DataFrame:
    """
    Compute attribution metrics (PR, FAR, RP) using the WWA framework
    with bootstrap confidence intervals.

    Fits a statistical distribution to the climate variable conditioned on
    global mean temperature, then extrapolates to "ALL" (factual) and "NAT"
    (counterfactual) climates to calculate probability ratio, fraction of
    attributable risk, and return periods.

    Parameters
    ----------
    all : xr.DataArray
        Data array with the climate variable to analyse.
    global_tas : pd.DataFrame
        DataFrame with global mean temperature anomalies indexed by time,
        containing a 'tas' column.
    fit_function_name : str
        Name of the scipy.stats distribution to fit
        (e.g. 'genextreme', 'norm', 'gamma', 'genpareto').
    all_date : Union[datetime, str], optional
        Date representing the factual (ALL) climate (default '2015-11-30').
        Also used to determine the event threshold.
    nat_date : Union[datetime, str], optional
        Date representing the counterfactual (NAT) climate (default '1900-11-30').
    strategy : str, optional
        Covariate strategy for the fit (default 'linear').
    direction : str, optional
        Tail direction for exceedance probabilities: 'descending' for upper
        tail (e.g. heat extremes) or 'ascending' for lower tail
        (e.g. cold extremes). Default 'descending'.
    bootstrap_ci : int, optional
        Confidence interval percentage for bootstrap (default 95).
    boot_size : int, optional
        Number of bootstrap iterations (default 1000).
    seed : int, optional
        Random seed for reproducibility (default 42).
    summary_statistics : bool, optional
        If True, return median and confidence intervals. If False, return
        the full bootstrap ensemble (default True).
    verbose : bool, optional
        If True, print fit details (default False).
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores, default -1).

    Returns
    -------
    pd.DataFrame
        If summary_statistics is True: a DataFrame with index
        ['value', 'ci_inf', 'ci_sup'] and columns ['PR', 'FAR', 'RP_ALL', 'RP_NAT'].
        If summary_statistics is False: a DataFrame with one row per bootstrap
        iteration and columns ['PR', 'FAR', 'RP_ALL', 'RP_NAT'].
    """

    all_dataframe = all.to_dataframe().reset_index()

    #global_tas['tas'] = global_tas['tas'].rolling(4, center=True).mean()
    dataframe = all_dataframe.set_index('time').join(global_tas).dropna()

    # get the threshold value for the selected date
    thresh = dataframe.loc[all_date, all.name]

    validate_direction(direction)

    # Bootstrap for confidence intervals
    boot_size = 1000
    indices = np.arange(len(dataframe))
    indices_boot = _calc_bootstrap_ensemble(indices, boot_size=boot_size, seed=seed)

    fit_function = getattr(stats, fit_function_name)

    def compute_metrics(boot):
        """Function to compute PR, FAR, RP_ALL, RP_NAT for a single bootstrap iteration."""
        dataframe_boot = dataframe.iloc[indices_boot[boot]].sort_index().dropna()
        params_boot = fit_data(
            dataframe_boot[all.name], dataframe_boot['tas'], fit_function_name, strategy, verbose=verbose
        )

        # extrapolate data to the selected dates
        params_all, all_wwa = extrapolate_data(global_tas, params_boot, all_date, fit_function_name, strategy)
        params_nat, nat_wwa = extrapolate_data(global_tas, params_boot, nat_date, fit_function_name, strategy)

        return {
            'PR': float(_pr_calculation(all_wwa, nat_wwa, fit_function, thresh, direction, params_all, params_nat)),
            'FAR': float(_far_calculation(all_wwa, nat_wwa, fit_function, thresh, direction, params_all, params_nat)),
            'RP_ALL': float(_rp_calculation(all_wwa, fit_function, thresh, direction, params_all)),
            'RP_NAT': float(_rp_calculation(nat_wwa, fit_function, thresh, direction, params_nat))
        }

    # Run computations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_metrics)(boot) for boot in range(int(boot_size))
    )

    # Convert results to structured arrays
    metrics = {
        'PR': np.array([res['PR'] for res in results]),
        'FAR': np.array([res['FAR'] for res in results]),
        'RP_ALL': np.array([res['RP_ALL'] for res in results]),
        'RP_NAT': np.array([res['RP_NAT'] for res in results])
    }

    if summary_statistics:
        ci_inf, ci_sup = get_percentiles_from_ci(bootstrap_ci)

        # Create empty metrics dataframe
        metrics_result = pd.DataFrame(
            np.zeros((3, 4)), 
            index=['value', 'ci_inf', 'ci_sup'], 
            columns=['PR', 'FAR', 'RP_ALL', 'RP_NAT']
        )

        # Fill dataframe with metrics
        for metric_name in ['PR', 'FAR', 'RP_ALL', 'RP_NAT']:
            metric_without_nan = metrics[metric_name][~np.isnan(metrics[metric_name])]

            metrics_result.loc['value', metric_name] = np.median(metric_without_nan)
            metrics_result.loc['ci_inf', metric_name] = np.percentile(metric_without_nan, ci_inf)
            metrics_result.loc['ci_sup', metric_name] = np.percentile(metric_without_nan, ci_sup)
    else:
        metrics_result = pd.DataFrame(metrics)

    return metrics_result
        
###############################################################################

def histogram_plot(
    ax,
    all: xr.DataArray, 
    global_tas: pd.DataFrame,
    fit_function_name: str,
    all_date: Union[datetime, str] = '2015-11-30',
    nat_date: Union[datetime, str] = '1900-11-30',
    strategy: str = 'linear',
    verbose: bool = False,
    **kwargs) -> None:
    """
    Plot histograms and fitted PDFs for the factual (ALL) and
    counterfactual (NAT) climate scenarios.

    Fits a statistical distribution to the climate variable conditioned on
    global mean temperature, extrapolates to the ALL and NAT dates, and
    plots the resulting histograms and fitted PDFs. A vertical dashed line
    marks the observed event threshold.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw the plot.
    all : xr.DataArray
        Data array with the climate variable to analyse.
    global_tas : pd.DataFrame
        DataFrame with global mean temperature anomalies indexed by time,
        containing a 'tas' column.
    fit_function_name : str
        Name of the scipy.stats distribution to fit
        (e.g. 'genextreme', 'norm', 'gamma', 'genpareto').
    all_date : Union[datetime, str], optional
        Date representing the factual (ALL) climate (default '2015-11-30').
        Also used to determine the event threshold.
    nat_date : Union[datetime, str], optional
        Date representing the counterfactual (NAT) climate (default '1900-11-30').
    strategy : str, optional
        Covariate strategy for the fit (default 'linear').
    verbose : bool, optional
        If True, print fit details (default False).
    **kwargs
        Additional keyword arguments:
        - all_color : str, color for the ALL scenario (default 'C1').
        - nat_color : str, color for the NAT scenario (default 'C0').
        - alpha : float, histogram transparency (default 0.5).

    Returns
    -------
    None
        Modifies the provided axes object in-place.
    """
    all_dataframe = all.to_dataframe().reset_index()

    # global_tas['tas'] = global_tas['tas'].rolling(4, center=True).mean()
    dataframe = all_dataframe.set_index('time').join(global_tas).dropna()

    # fit function using MLE
    params = fit_data(
        dataframe[all.name], dataframe['tas'], fit_function_name, strategy, verbose=verbose
    )

    # extrapolate data to the selected dates
    params_all, all_wwa = extrapolate_data(global_tas, params, all_date, fit_function_name, strategy)
    params_nat, nat_wwa = extrapolate_data(global_tas, params, nat_date, fit_function_name, strategy)

    fit_function = getattr(stats, fit_function_name)

    # get the threshold value for the selected date
    thresh = dataframe.loc[all_date, all.name]

    # getting the kwargs
    all_color = kwargs.get('all_color', 'C1')
    nat_color = kwargs.get('nat_color', 'C0')
    alpha = kwargs.get('alpha', 0.5)

    ax.hist(all_wwa, color=all_color, alpha=alpha, density=True, label='ALL')
    ax.hist(nat_wwa, color=nat_color, alpha=alpha, density=True, label='NAT')

    # fit the requested distribution and plot it as a line
    percentiles = np.linspace(0.01, 99.9, 700)
    x_all = get_fitted_percentiles(percentiles, params_all, fit_function)
    x_nat = get_fitted_percentiles(percentiles, params_nat, fit_function)

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color=all_color, lw=2)
    ax.plot(x_nat, fit_function.pdf(x_nat, *params_nat), color=nat_color, lw=2)

    ax.axvline(thresh, color='k', ls='--')
    ax.legend()

###############################################################################

# def rp_plot(
#     ax,
#     all: xr.DataArray, 
#     global_tas: pd.DataFrame,
#     fit_function_name: str,
#     all_date: Union[datetime, str] = '2015-11-30',
#     nat_date: Union[datetime, str] = '1900-11-30',
#     strategy: str = 'linear',
#     verbose: bool = False,
#     direction: str = 'descending',
#     bootstrap_ci: int = 95,
#     boot_size: int = 1000,
#     **kwargs) -> None:
#     """
#     Plot return periods for the "ALL" and "NAT" scenarios, including 
#     confidence intervals (CI) for the bootstrapped return periods.

#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes object on which to draw the return period plot.
    
#     all : xr.DataArray
#         Data array representing the "ALL" scenario, which includes 
#         human influences on climate.
    
#     nat : xr.DataArray
#         Data array representing the "NAT" scenario, which represents 
#         the natural climate without human influences.
    
#     fit_function : callable
#         A statistical distribution or fitting function used to model the data.
    
#     thresh : float
#         The threshold value, which is plotted as a horizontal dashed line.
    
#     direction : str, optional, default = 'descending'
#         The direction in which to assess exceedance of the threshold. 
#         Can be 'descending' or 'ascending'.
    
#     bootstrap_ci : int, optional, default = 95
#         The confidence interval (CI) percentage for bootstrapping.
    
#     boot_size : int, optional, default = 1000
#         The number of bootstrap samples to generate.

#     Returns
#     -------
#     None
#         This function does not return anything; it modifies the provided 
#         axes object in-place.
#     """
#     # validation steps
#     validate_direction(direction)
#     validate_ci(bootstrap_ci)

#     all_dataframe = all.to_dataframe().reset_index()

#     # global_tas['tas'] = global_tas['tas'].rolling(4, center=True).mean()
#     dataframe = all_dataframe.set_index('time').join(global_tas).dropna()

#     # fit function using MLE
#     params = fit_data(
#         dataframe[all.name], dataframe['tas'], fit_function_name, strategy, verbose=verbose
#     )

#     # extrapolate data to the selected dates
#     params_all, all_wwa = extrapolate_data(global_tas, params, all_date, fit_function_name, strategy)
#     params_nat, nat_wwa = extrapolate_data(global_tas, params, nat_date, fit_function_name, strategy)

#     fit_function = getattr(stats, fit_function_name)

#     # get the threshold value for the selected date
#     thresh = dataframe.loc[all_date, all.name]

#     if direction == 'descending':
#         all_wwa = all_wwa[::-1]
#         nat_wwa = nat_wwa[::-1]

#         all_span_checker = all_wwa.max() >= thresh
#         nat_span_checker = nat_wwa.max() >= thresh
#     else:
#         all_span_checker = all_wwa.min() <= thresh
#         nat_span_checker = nat_wwa.min() <= thresh

#     # getting the kwargs
#     all_color = kwargs.get('all_color', 'C1')
#     nat_color = kwargs.get('nat_color', 'C0')

#     conf_rp_inf_all, conf_rp_sup_all = _rp_plot_data(
#         all_wwa, fit_function, all_color, 'ALL', ax, direction, bootstrap_ci, boot_size, params_all
#     )
#     conf_rp_inf_nat, conf_rp_sup_nat = _rp_plot_data(
#         nat_wwa, fit_function, nat_color, 'NAT', ax, direction, bootstrap_ci, boot_size, params_nat
#     )

#     ax.axhline(thresh, color='k', ls='--')

#     # add return period estimate for ALL
#     idx = find_nearest(thresh, all_wwa)

#     ymin, ymax = ax.get_ylim()

#     if all_span_checker:
#         ax.axvspan(
#             conf_rp_inf_all[idx], conf_rp_sup_all[idx], 
#             ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
#             facecolor='silver', edgecolor=all_color,
#             linewidth=2., alpha=0.3, zorder=0
#         )

#     # add return period estimate for NAT
#     idx = find_nearest(thresh, nat_wwa)

#     if nat_span_checker:
#         ax.axvspan(
#             conf_rp_inf_nat[idx], conf_rp_sup_nat[idx], 
#             ymin=0, ymax=(thresh - ymin)/ (ymax - ymin),
#             facecolor='silver', edgecolor=nat_color,
#             linewidth=2., alpha=0.3, zorder=0
#         )

#     ax.legend()

# ############################################################################### 