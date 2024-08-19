import numpy as np
import pandas as pd

from climattr.main import ClimAttr
from climattr.utils import find_nearest
from climattr.validator import (
    validate_bootstrap_ci,
    validate_direction
)


def _calc_return_time_confidence(
    data, 
    direction="ascending", 
    bootstrap_ci=95, 
    boot_size=100):

    n_samples = data.shape[0]
    boot_size = int(boot_size)

    ci_inf = (100 - bootstrap_ci) / 2
    ci_sup = 100 - (100 - bootstrap_ci) / 2

    # Use np.random.choice to perform the resampling in a vectorized way
    # to select the n boot_size samples of the bootstrap algorithm
    sample_store = np.random.choice(data, (boot_size, n_samples), replace=True)

    # Sort the resampled data along each row
    sample_store.sort(axis=1)

    # Reverse the data if direction is descending
    if direction == "descending":
        sample_store = sample_store[:, ::-1]

    # Calculate the confidence intervals using np.percentile
    conf_inter = np.percentile(sample_store, np.array([ci_inf, ci_sup]), axis=0)
    
    return conf_inter

###############################################################################

def _rp_plot_data(
    data,
    fit_function,
    color,
    label,
    ax,
    direction='descending',
    bootstrap_ci=95,
    boot_size=1000):

    return_period = np.array([_rp_calculation(data, fit_function, i, direction) for i in data])

    conf_data = _calc_return_time_confidence(
        data, 
        direction=direction, 
        boot_size=boot_size, 
        bootstrap_ci=bootstrap_ci
    )
    conf_rp = _calc_return_time_confidence(
        return_period,
        direction='descending',
        boot_size=boot_size,
        bootstrap_ci=bootstrap_ci
    )

    ax.semilogx(
        return_period, data, marker='o', markersize=2,
        linestyle='None', mec=color, mfc=color,
        color=color, fillstyle='full',
        label=label, zorder=2
    )

    # plot the fitted line
    params = fit_function.fit(data)
    x = np.linspace(
        fit_function.ppf(0.001, *params), 
        fit_function.ppf(0.991, *params), 
        700
    )
    if direction == 'descending':
        fitted_rp = np.array([1 / fit_function.sf(i, *params) for i in x])
    else:
        fitted_rp = np.array([1 / fit_function.cdf(i, *params) for i in x])

    ax.semilogx(fitted_rp, x, color=color, lw=2)

    conf_data_inf = conf_data[0,:].squeeze()
    conf_data_sup = conf_data[1,:].squeeze()

    conf_rp_inf = conf_rp[0,:].squeeze()
    conf_rp_sup = conf_rp[1,:].squeeze()

    ax.fill_between(
        return_period, conf_data_inf, conf_data_sup, color=color,
        alpha=0.2,linewidth=1.,zorder=0
    )
    ax.semilogx(
        [return_period, return_period],
        [conf_data_inf, conf_data_sup],
        color=color, linewidth=1., zorder=1
    )
    ax.semilogx(
        [conf_rp_inf, conf_rp_sup],
        [data, data],
        color=color, linewidth=1., zorder=1
    )

    return conf_rp_inf, conf_rp_sup

###############################################################################

def _calc_bootstrap_ensemble(
    data, 
    direction="ascending", 
    boot_size=1000):
    
    # Flatten the input data
    shape = data.shape[0]
    
    # Generate the bootstrap samples using np.random.choice
    sample_store = np.random.choice(data, (int(boot_size), shape), replace=True)
    
    # Sort each row in the sample_store
    sample_store.sort(axis=1)
    
    # Reverse the rows if direction is "descending"
    if direction == "descending":
        sample_store = sample_store[:, ::-1]
    
    return sample_store

###############################################################################

def _pr_calculation(
    all_array, 
    nat_array, 
    fit_function, 
    threshold,
    direction='descending'):

    params_all = fit_function.fit(all_array)
    params_nat = fit_function.fit(nat_array)

    if direction == 'descending':
        pr = fit_function.sf(threshold, *params_all) \
            / fit_function.sf(threshold, *params_nat)
    else:
        pr = fit_function.cdf(threshold, *params_all) \
            / fit_function.cdf(threshold, *params_nat)

    return pr

###############################################################################

def _far_calculation(
    all_array, 
    nat_array, 
    fit_function, 
    threshold):

    return 1 - (1 / _pr_calculation(
        all_array, nat_array, fit_function, threshold
    ))

###############################################################################

def _rp_calculation(
    data, 
    fit_function, 
    threshold,
    direction='descending'):

    params = fit_function.fit(data)

    if direction == 'descending':
        rp = 1 / fit_function.sf(threshold, *params)
    else:
        rp = 1 / fit_function.cdf(threshold, *params)

    return rp

###############################################################################

@staticmethod
def attribution_metrics(
    all, 
    nat, 
    fit_function,
    threshold,
    direction='descending',
    bootstrap_ci=95,
    boot_size=1000):

    validate_bootstrap_ci(bootstrap_ci)
    validate_direction(direction)

    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    all_boot = _calc_bootstrap_ensemble(all_array, boot_size=boot_size)    
    nat_boot = _calc_bootstrap_ensemble(nat_array, boot_size=boot_size)

    template_array = np.zeros(boot_size)
    metrics = {
        'PR': template_array.copy(), 
        'FAR': template_array.copy(), 
        'RP_ALL': template_array.copy(), 
        'RP_NAT': template_array.copy()
    }
    for boot in range(int(boot_size)):
        metrics['PR'][boot] = \
            _pr_calculation(
                all_boot[boot], nat_boot[boot], fit_function, threshold, direction
            )
        metrics['FAR'][boot] = \
            _far_calculation(
                all_boot[boot], nat_boot[boot], fit_function, threshold
            )
        metrics['RP_ALL'][boot] = \
            _rp_calculation(
                all_boot[boot], fit_function, threshold, direction
            )
        metrics['RP_NAT'][boot] = \
            _rp_calculation(
                nat_boot[boot], fit_function, threshold, direction
            )

    ci_inf = (100 - bootstrap_ci) / 2
    ci_sup = 100 - (100 - bootstrap_ci) / 2

    # create empty metrics dataframe
    metrics_result = pd.DataFrame(
        np.zeros((4, 3)), 
        columns=['value', 'ci_inf', 'ci_sup'], 
        index=['PR', 'FAR', 'RP_ALL', 'RP_NAT']
    )

    # fill dataframe with metrics
    metrics_result.loc['PR', 'value'] = np.median(metrics['PR'])
    metrics_result.loc['PR', 'ci_inf'] = np.percentile(metrics['PR'], ci_inf)
    metrics_result.loc['PR', 'ci_sup'] = np.percentile(metrics['PR'], ci_sup)
    metrics_result.loc['FAR', 'value'] = np.median(metrics['FAR'])
    metrics_result.loc['FAR', 'ci_inf'] = np.percentile(metrics['FAR'], ci_inf)
    metrics_result.loc['FAR', 'ci_sup'] = np.percentile(metrics['FAR'], ci_sup)
    metrics_result.loc['RP_ALL', 'value'] = np.median(metrics['RP_ALL'])
    metrics_result.loc['RP_ALL', 'ci_inf'] = np.percentile(metrics['RP_ALL'], ci_inf)
    metrics_result.loc['RP_ALL', 'ci_sup'] = np.percentile(metrics['RP_ALL'], ci_sup)
    metrics_result.loc['RP_NAT', 'value'] = np.median(metrics['RP_NAT'])
    metrics_result.loc['RP_NAT', 'ci_inf'] = np.percentile(metrics['RP_NAT'], ci_inf)
    metrics_result.loc['RP_NAT', 'ci_sup'] = np.percentile(metrics['RP_NAT'], ci_sup)

    return metrics_result
        
###############################################################################

@staticmethod
def histogram_plot(
    ax,
    all,
    nat,
    fit_function,
    threshold):

    all_array = all.to_numpy().flatten()
    nat_array = nat.to_numpy().flatten()

    params_all = fit_function.fit(all_array)
    params_nat = fit_function.fit(nat_array)

    ax.hist(all_array, color='C0', alpha=0.5, density=True, label='ALL')
    ax.hist(nat_array, color='C1', alpha=0.5, density=True, label='NAT')

    # fit the requested distribution and plot it as a line
    x_all = np.linspace(
        fit_function.ppf(0.01, params_all[0]), 
        fit_function.ppf(0.99, params_all[0]), 
        700
    )
    x_nat = np.linspace(
        fit_function.ppf(0.01, params_nat[0]), 
        fit_function.ppf(0.99, params_nat[0]), 
        700
    )

    ax.plot(x_all, fit_function.pdf(x_all, *params_all), color='C0', lw=2)
    ax.plot(x_nat, fit_function.pdf(x_nat, *params_nat), color='C1', lw=2)

    ax.axvline(threshold, color='k', ls='--')
    ax.legend()

###############################################################################

@staticmethod
def rp_plot(
    ax,
    all,
    nat,
    fit_function,
    threshold,
    direction='descending',
    bootstrap_ci=95,
    boot_size=1000):

    # validation steps
    validate_bootstrap_ci(bootstrap_ci)
    validate_direction(direction)

    all_array = np.sort(all.to_numpy().flatten())
    nat_array = np.sort(nat.to_numpy().flatten())

    if direction == 'descending':
        all_array = all_array[::-1]
        nat_array = nat_array[::-1]

    conf_rp_inf_all, conf_rp_sup_all = _rp_plot_data(
        all_array, fit_function, 'C0', 'ALL', ax, direction, bootstrap_ci, boot_size
    )
    conf_rp_inf_nat, conf_rp_sup_nat = _rp_plot_data(
        nat_array, fit_function, 'C1', 'NAT', ax, direction, bootstrap_ci, boot_size
    )

    ax.axhline(threshold, color='k', ls='--')

    # add return period estimate for ALL
    idx = find_nearest(threshold, all_array)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(
        conf_rp_inf_all[idx], conf_rp_sup_all[idx], 
        ymin=0, ymax=(threshold - ymin)/ (ymax - ymin),
        facecolor='silver', edgecolor='C0',
        linewidth=2., alpha=0.3, zorder=0
    )

    # add return period estimate for NAT
    idx = find_nearest(threshold, nat_array)
    ax.axvspan(
        conf_rp_inf_nat[idx], conf_rp_sup_nat[idx], 
        ymin=0, ymax=(threshold - ymin)/ (ymax - ymin),
        facecolor='silver', edgecolor='C1',
        linewidth=2., alpha=0.3, zorder=0
    )

###############################################################################
