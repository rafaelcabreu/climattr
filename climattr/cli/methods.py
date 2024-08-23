import matplotlib.pyplot as plt
import xarray as xr
import xclim

import scipy.stats

import climattr as eea

def _read_file(args, option):

    # Load datasets based on the selected data source
    if args.data_source == 'cmip6':
        data = eea.utils.from_cmip6(option)
    elif args.data_source == 'netcdf':
        data = xr.open_mfdataset(option)

    return data

#####################################################################

def method_filter_time(args):

    data = _read_file(args, args.ifile)

    if args.itime and args.etime:
        data = data.sel(time=slice(args.itime, args.etime))
    elif args.months:
        months = [int(month) for month in args.months]
        data = data.where(data['time.month'].isin(months))
    else:
        raise ValueError('You should either add months argument of itime,etime')

    data[[args.variable]].to_netcdf(
        args.ofile, 
        encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    )

#####################################################################

def method_filter_area(args):

    data = _read_file(args, args.ifile)

    if args.box:
        box = [float(coord) for coord in args.box]
        data = eea.spatial.filter_area(data, box=box)
    elif args.mask:
        data = eea.spatial.filter_area(data, mask=args.mask)
    else:
        raise ValueError('You should add either a box or a mask argument')

    if args.reduce:
        data = getattr(data, args.reduce)(dim=eea.utils.get_xy_coords(data))
    data[[args.variable]].to_netcdf(
        args.ofile, 
        encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    )

#####################################################################

def method_attribution_metrics(args):

    all_data = _read_file(args, args.all)
    nat_data = _read_file(args, args.nat)

    fit_function = getattr(scipy.stats, args.fit_function)

    metrics = eea.attribution.attribution_metrics(
        all_data[args.variable], 
        nat_data[args.variable], 
        fit_function, 
        args.thresh, 
        bootstrap_ci=95,
        direction='descending'
    )
    
    # print metrics
    print(metrics)

    metrics.to_csv(args.ofile)

#####################################################################

def method_attribution_plot(args):

    all_data = _read_file(args, args.all)
    nat_data = _read_file(args, args.nat)

    fit_function = getattr(scipy.stats, args.fit_function)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    eea.attribution.histogram_plot(
        ax1,
        all_data[args.variable], 
        nat_data[args.variable], 
        fit_function, 
        args.thresh, 
    )

    eea.attribution.rp_plot(
        ax2,
        all_data[args.variable], 
        nat_data[args.variable], 
        fit_function, 
        args.thresh, 
        direction=args.direction,
        bootstrap_ci=95
    )

    ax1.set_xlabel(args.variable)
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('Return Period (years)')
    ax2.set_ylabel(args.variable)

    plt.tight_layout()
    fig.savefig(args.ofile, dpi=300, bbox_inches='tight')

#####################################################################

def method_qq_plot(args):

    all_data = _read_file(args, args.all)
    obs_data = _read_file(args, args.obs)

    fit_function = getattr(scipy.stats, args.fit_function)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    #eea.validation.qq_plot(ax, nat['tas'], all['tas'])
    eea.validation.qq_plot_theoretical(ax1, obs_data[args.variable], fit_function)
    eea.validation.qq_plot_theoretical(ax2, all_data[args.variable], fit_function)

    ax1.set_title('OBS')
    ax2.set_title('ALL')

    ax1.set_xlabel('Theoretical Percentiles')
    ax1.set_ylabel(args.variable)
    ax2.set_xlabel('Theoretical Percentiles')
    ax2.set_ylabel(args.variable)

    plt.tight_layout()
    fig.savefig(args.ofile, dpi=300, bbox_inches='tight')

#####################################################################

def method_validation_plot(args):

    all_data = _read_file(args, args.all)
    obs_data = _read_file(args, args.obs)

    fit_function = getattr(scipy.stats, args.fit_function)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    #eea.validation.qq_plot(ax, nat['tas'], all['tas'])
    eea.validation.histogram_plot(ax1, obs_data[args.variable], all_data[args.variable], fit_function)
    eea.validation.qq_plot(ax2, obs_data[args.variable], all_data[args.variable])

    ax1.set_xlabel(args.variable)
    ax1.set_ylabel('PDF')
    ax2.set_xlabel('OBS')
    ax2.set_ylabel('ALL')

    plt.tight_layout()
    fig.savefig(args.ofile, dpi=300, bbox_inches='tight')

#####################################################################

def method_exploratory_plot(args):

    data = _read_file(args, args.ifile)

    fit_function = getattr(scipy.stats, args.fit_function)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))

    eea.exploratory.timeseries_plot(
        ax1, 
        data[args.variable].isel(ensemble=0), 
        highlight_year=args.year
    )
    eea.exploratory.rp_plot(
        ax2, 
        data[args.variable].isel(ensemble=0), 
        fit_function, 
        highlight_year=args.year
    )

    ax1.set_xlabel('Time')
    ax1.set_ylabel(args.variable)
    ax2.set_xlabel('Return Period (years)')
    ax2.set_ylabel(args.variable)

    plt.tight_layout()
    fig.savefig(args.ofile, dpi=300, bbox_inches='tight')

#####################################################################

def method_xclim(args):

    data = _read_file(args, args.ifile)

    with xclim.set_options(
        check_missing="pct",
        missing_options={"pct": dict(tolerance=1)},
        cf_compliance="log",
        data_validation='log'
    ):

        xclim_function = getattr(
            xclim.indicators.atmos, 
            args.xclim_function
        )

        indice = xclim_function(
            **args.kwargs,
            ds=data
        )

        indice.to_netcdf(
            args.ofile, 
            encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
        )

#####################################################################

def method_scaling(args):

    data = _read_file(args, args.ifile)
    clim = _read_file(args, args.clim)

    scaled_data = eea.correction.scaling(
        data[args.var],
        clim[args.var],
        args.idate,
        args.edate,
        method = args.method
    )

    scaled_data.to_netcdf(
        args.ofile, 
        encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    )

#####################################################################
