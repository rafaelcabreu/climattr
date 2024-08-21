
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
