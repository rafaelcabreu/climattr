import argparse

from climattr.cli.parsers import (
    parser_filter_area,
    parser_filter_time,
    parser_attribution_metrics,
    parser_attribution_plot
)
from climattr.cli.methods import (
    method_filter_area,
    method_filter_time,
    method_attribution_metrics,
    method_attribution_plot
)

def main():
    parser = argparse.ArgumentParser(description="Run attribution metrics from CMIP6 data")
    subparsers = parser.add_subparsers(dest='command')

    parser_filter_area(subparsers)
    parser_filter_time(subparsers)
    parser_attribution_metrics(subparsers)
    parser_attribution_plot(subparsers)

    parser.add_argument(
        '-i', '--ifile', 
        required=False,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['cmip6', 'netcdf'],
        required=True,
        help="Specify the data source type: 'cmip6' or 'netcdf'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    
    args = parser.parse_args()
    
    if args.command == 'filter-time':
        method_filter_time(args)
    elif args.command == 'filter-area':
        method_filter_area(args)
    elif args.command == 'attr-metrics':
        method_attribution_metrics(args)
    elif args.command == 'attr-plot':
        method_attribution_plot(args)

if __name__ == '__main__':
    main()