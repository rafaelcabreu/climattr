import argparse

from climattr.cli.parsers import (
    parser_filter_area,
    parser_filter_time,
    parser_attribution_metrics,
    parser_attribution_plot,
    parser_qq_plot,
    parser_validation_plot,
    parser_exploratory_plot,
    parser_xclim,
    parser_scaling
)
from climattr.cli.methods import (
    method_filter_area,
    method_filter_time,
    method_attribution_metrics,
    method_attribution_plot,
    method_qq_plot,
    method_validation_plot,
    method_exploratory_plot,
    method_xclim,
    method_scaling
)

def main():
    parser = argparse.ArgumentParser(description="Run Extreme Event Attribution")
    subparsers = parser.add_subparsers(dest='command')

    parser_filter_area(subparsers)
    parser_filter_time(subparsers)
    parser_attribution_metrics(subparsers)
    parser_attribution_plot(subparsers)
    parser_qq_plot(subparsers)
    parser_validation_plot(subparsers)
    parser_exploratory_plot(subparsers)
    parser_xclim(subparsers)
    parser_scaling(subparsers)
    
    args = parser.parse_args()
    
    methods = {
        'filter-time': method_filter_time,
        'filter-area': method_filter_area,
        'attr-metrics': method_attribution_metrics,
        'attr-plot': method_attribution_plot,
        'qq-plot': method_qq_plot,
        'validation-plot': method_validation_plot,
        'exploratory-plot': method_exploratory_plot,
        'xclim-indice': method_xclim,
        'scale': method_scaling
    }

    methods[args.command](args)

if __name__ == '__main__':
    main()