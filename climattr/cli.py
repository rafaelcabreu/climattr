import argparse
import scipy
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

def filter_area(parser, subparsers):

    spatial_parser = subparsers.add_parser('filter-area', help='Help for spatial filter')
    spatial_parser.add_argument('--box', required=False, help='Box helper')
    spatial_parser.add_argument('--mask', required=False, help='Box helper')

#####################################################################

def filter_time(parser, subparsers):

    time_parser = subparsers.add_parser('filter-time', help='Help for time filter')
    time_parser.add_argument('--itime', required=False, help='Box helper')
    time_parser.add_argument('--etime', required=False, help='Box helper')

    args = parser.parse_args()

#####################################################################


def main():
    parser = argparse.ArgumentParser(description="Run attribution metrics from CMIP6 data")
    subparsers = parser.add_subparsers()

    filter_area(parser, subparsers)
    filter_time(parser, subparsers)

    # # Adding arguments
    # parser.add_argument(
    #     '--data', 
    #     required=False,
    #     help="Path to the dataset files"
    # )
    # parser.add_argument(
    #     '-a', '--all', 
    #     required=False,
    #     help="Path to the CMIP6 'all' dataset files"
    # )
    # parser.add_argument(
    #     '-n', '--nat', 
    #     required=False,
    #     help="Path to the CMIP6 'nat' dataset files"
    # )
    # parser.add_argument(
    #     '-d', '--data-source', 
    #     choices=['cmip6', 'netcdf'],
    #     required=True,
    #     help="Specify the data source type: 'cmip6' or 'netcdf'"
    # )
    # parser.add_argument(
    #     '--filter-area',
    #     nargs='+',  # Accepts multiple values (either for box coordinates or shapefile)
    #     help="Filter the data for a specific area: provide 4 values for a box (lon_min, lon_max, lat_min, lat_max) or a shapefile path"
    # )
    # parser.add_argument(
    #     '--filter-time', 
    #     required=False,
    #     help="Filter dataset based on initial and final time"
    # )
    # parser.add_argument(
    #     '--stat-function',
    #     choices=['mean', 'min', 'max'],
    #     help="Statistical function to apply after area filtering (mean, min, or max)"
    # )
    # parser.add_argument(
    #     '-o', '--ofile',
    #     required=False,
    #     help="Output file name"
    # )
    # parser.add_argument(
    #     '-m', '--metrics', 
    #     action='store_true',
    #     help="Calculate and print the attribution metrics"
    # )
    # parser.add_argument(
    #     '-v', '--variable',
    #     required=True,
    #     type=str,
    #     default='tas',
    #     help="Variable name to use for attribution metrics (default: 'tas')"
    # )
    # parser.add_argument(
    #     '-f', '--fit_function',
    #     type=str,
    #     default='norm',
    #     help="Scipy fit function to use for attribution metrics (default: 'norm')"
    # )
    # parser.add_argument(
    #     '-t', '--threshold',
    #     type=int,
    #     default=301,
    #     help="Threshold value for the attribution metrics (default: 301)"
    # )

    #args = parser.parse_args()

    # # time filter based on initial and final time
    # if args.filter_time:
    #     data = _read_file(args, args.data)

    #     data = data.sel(time=slice(args.itime, args.etime))
    #     data[[args.variable]].to_netcdf(
    #         args.ofile, 
    #         encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    #     )

    # # spatial filter based on bounding box or shapefile, and apply a method
    # if args.filter_area:
    #     data = _read_file(args, args.data)

    #     # Filter by bounding box
    #     if len(args.filter_area) == 4:
    #         box = [float(coord) for coord in args.filter_area]
    #         data = eea.spatial.filter_area(data, box=box)
    #         data = getattr(data, args.stat_function)(dim=eea.utils.get_xy_coords(data))
    #         data[[args.variable]].to_netcdf(
    #             args.ofile, 
    #             encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    #         )
    #     # Filter by shapefile
    #     elif len(args.filter_area) == 1:
    #         mask = args.filter_area[0]
    #         data = eea.spatial.filter_area(data, mask=mask)
    #         data = getattr(data, args.stat_function)(dim=eea.utils.get_xy_coords(data))
    #         data.to_netcdf(args.ofile)
    #         data[[args.variable]].to_netcdf(
    #             args.ofile, 
    #             encoding={'time':{'units':'days since 1850-01-01', 'dtype': 'float64'}}
    #         )
    #     else:
    #         raise ValueError(
    #             "Invalid --filter-area argument. Provide either 4 coordinates for a box or 1 shapefile path."
    #         )
        
    # # Calculate metrics if selected
    # elif args.metrics:
    #     if args.all and args.nat: 
    #         all_data = _read_file(args, args.all)
    #         nat_data = _read_file(args, args.nat)
    #     else:
    #         raise ValueError(
    #             "To calculate the metrics you should provide both ALL and NAT files paths."
    #         )

    #     fit_function = getattr(scipy.stats, args.fit_function)

    #     metrics = eea.attribution_metrics(
    #         all_data[args.variable], 
    #         nat_data[args.variable], 
    #         fit_function, 
    #         args.threshold, 
    #         bootstrap_ci=95,
    #         direction='descending'
    #     )
        
    #     # print metrics
    #     print(metrics)

#####################################################################

if __name__ == '__main__':
    main()