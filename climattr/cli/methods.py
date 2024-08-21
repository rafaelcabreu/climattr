import xarray as xr

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

    data = data.sel(time=slice(args.itime, args.etime))
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
