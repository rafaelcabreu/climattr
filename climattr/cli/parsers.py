
def parser_filter_area(subparsers):

    spatial_parser = subparsers.add_parser(
        'filter-area', 
        help='Function used to filter the dataset based on a spatial filter',
    )
    spatial_parser.add_argument(
        '--box', 
        nargs='+',
        required=False, 
        help='Box used to filter the dataset latitude and longitude'
    )
    spatial_parser.add_argument(
        '--mask', 
        required=False, 
        help='Path for the shapefile name used to filter the dataset'
    )
    spatial_parser.add_argument(
        '--reduce',
        required=False,
        choices=['mean', 'min', 'max'],
        help="Statistical function to apply after area filtering (mean, min, or max)"
    )


#####################################################################

def parser_filter_time(subparsers):

    time_parser = subparsers.add_parser(
        'filter-time', 
        help='Function used to filter the dataset based on initial and final time'
    )
    time_parser.add_argument(
        '--itime', 
        required=True, 
        help='Initial time used to filter the dataset in the format YYYY-mm-dd'
    )
    time_parser.add_argument(
        '--etime', 
        required=True, 
        help='End time used to filter the dataset in the format YYYY-mm-dd'
    )

#####################################################################

def parser_attribution_metrics(subparsers):

    attribution_parser = subparsers.add_parser(
        'attr-metrics', 
        help='Function used calculate the attribution metrics (PR, RP, FAR)'
    )
    attribution_parser.add_argument(
        '-a', '--all', 
        required=False,
        help="Path to the CMIP6 'all' dataset files"
    )
    attribution_parser.add_argument(
        '-n', '--nat', 
        required=False,
        help="Path to the CMIP6 'nat' dataset files"
    )
    attribution_parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )
    attribution_parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=301,
        help="Threshold value for the attribution metrics (default: 301)"
    )
    attribution_parser.add_argument(
        '--direction',
        choices=['descending', 'ascending'],
        default='descending',
        help="Direction for the bootstrap ordering that will be used to calculate RP"
    )

#####################################################################

def parser_attribution_plot(subparsers):

    attribution_parser = subparsers.add_parser(
        'attr-plot', 
        help='Function used calculate the attribution metrics (PR, RP, FAR)'
    )
    attribution_parser.add_argument(
        '-a', '--all', 
        required=False,
        help="Path to the CMIP6 'all' dataset files"
    )
    attribution_parser.add_argument(
        '-n', '--nat', 
        required=False,
        help="Path to the CMIP6 'nat' dataset files"
    )
    attribution_parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )
    attribution_parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=301,
        help="Threshold value for the attribution metrics (default: 301)"
    )
    attribution_parser.add_argument(
        '--direction',
        choices=['descending', 'ascending'],
        default='descending',
        help="Direction for the bootstrap ordering that will be used to calculate RP"
    )

#####################################################################
