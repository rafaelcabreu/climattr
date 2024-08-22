import argparse

from climattr.cli.parsers import (
    parser_filter_area,
    parser_filter_time,
    parser_attribution_metrics,
    parser_attribution_plot,
    parser_qq_plot,
    parser_validation_plot,
    parser_xclim
)
from climattr.cli.methods import (
    method_filter_area,
    method_filter_time,
    method_attribution_metrics,
    method_attribution_plot,
    method_qq_plot,
    method_validation_plot,
    method_xclim
)

def main():
    parser = argparse.ArgumentParser(description="Run attribution metrics from CMIP6 data")
    subparsers = parser.add_subparsers(dest='command')

    parser_filter_area(subparsers)
    parser_filter_time(subparsers)
    parser_attribution_metrics(subparsers)
    parser_attribution_plot(subparsers)
    parser_qq_plot(subparsers)
    parser_validation_plot(subparsers)
    parser_xclim(subparsers)
    
    args = parser.parse_args()
    
    if args.command == 'filter-time':
        method_filter_time(args)
    elif args.command == 'filter-area':
        method_filter_area(args)
    elif args.command == 'attr-metrics':
        method_attribution_metrics(args)
    elif args.command == 'attr-plot':
        method_attribution_plot(args)
    elif args.command == 'qq-plot':
        method_qq_plot(args)
    elif args.command == 'validation-plot':
        method_validation_plot(args)
    elif args.command == 'xclim-indice':
        method_xclim(args)


if __name__ == '__main__':
    main()