import argparse

class ParseKwargs(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

#####################################################################

def parser_filter_area(subparsers):

    parser = subparsers.add_parser(
        'filter-area', 
        help='Function used to filter the dataset based on a spatial filter',
    )
    parser.add_argument(
        '-i', '--ifile', 
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '--box', 
        nargs='+',
        required=False, 
        help='Box used to filter the dataset latitude and longitude'
    )
    parser.add_argument(
        '--mask', 
        required=False, 
        help='Path for the shapefile name used to filter the dataset'
    )
    parser.add_argument(
        '--reduce',
        required=False,
        choices=['mean', 'min', 'max'],
        help="Statistical function to apply after area filtering (mean, min, or max)"
    )

#####################################################################

def parser_filter_time(subparsers):

    parser = subparsers.add_parser(
        'filter-time', 
        help='Function used to filter the dataset based on initial and final time'
    )
    parser.add_argument(
        '-i', '--ifile', 
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '--itime', 
        required=False, 
        help='Initial time used to filter the dataset in the format YYYY-mm-dd'
    )
    parser.add_argument(
        '--etime', 
        required=False, 
        help='End time used to filter the dataset in the format YYYY-mm-dd'
    )
    parser.add_argument(
        '--months', 
        nargs='+',
        required=False, 
        help='Select just specific months from the dataset'
    )

#####################################################################

def parser_attribution_metrics(subparsers):

    parser = subparsers.add_parser(
        'attr-metrics', 
        help='Function used calculate the attribution metrics (PR, RP, FAR)'
    )
    parser.add_argument(
        '-a', '--all', 
        required=True,
        help="Path to the 'all' dataset files"
    )
    parser.add_argument(
        '-n', '--nat', 
        required=True,
        help="Path to the 'nat' dataset files"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )
    parser.add_argument(
        '-t', '--thresh',
        type=float,
        default=301,
        help="Threshold value for the attribution metrics (default: 301)"
    )
    parser.add_argument(
        '--direction',
        choices=['descending', 'ascending'],
        default='descending',
        help="Direction for the bootstrap ordering that will be used to calculate RP"
    )

#####################################################################

def parser_attribution_plot(subparsers):

    parser = subparsers.add_parser(
        'attr-plot', 
        help='Function used to plot attribution histogram and RP'
    )
    parser.add_argument(
        '-a', '--all', 
        required=True,
        help="Path to the 'all' dataset files"
    )
    parser.add_argument(
        '-n', '--nat', 
        required=True,
        help="Path to the 'nat' dataset files"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )
    parser.add_argument(
        '-t', '--thresh',
        type=float,
        default=301,
        help="Threshold value for the attribution metrics (default: 301)"
    )
    parser.add_argument(
        '--direction',
        choices=['descending', 'ascending'],
        default='descending',
        help="Direction for the bootstrap ordering that will be used to calculate RP"
    )

#####################################################################

def parser_qq_plot(subparsers):

    parser = subparsers.add_parser(
        'qq-plot', 
        help='Function used for QQ-Plot against theoretical quantiles'
    )
    parser.add_argument(
        '-a', '--all', 
        required=True,
        help="Path to the 'all' dataset files"
    )
    parser.add_argument(
        '--obs', 
        required=True,
        help="Path to the observational dataset files"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )

#####################################################################

def parser_validation_plot(subparsers):

    parser = subparsers.add_parser(
        'validation-plot', 
        help='Function used for validation plot of OBS and ALL'
    )
    parser.add_argument(
        '-a', '--all', 
        required=True,
        help="Path to the 'all' dataset files"
    )
    parser.add_argument(
        '--obs', 
        required=True,
        help="Path to the observational dataset files"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )

#####################################################################

def parser_exploratory_plot(subparsers):

    parser = subparsers.add_parser(
        'exploratory-plot', 
        help='Function used to explore the plots for OBS'
    )
    parser.add_argument(
        '-i', '--ifile', 
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '-f', '--fit_function',
        type=str,
        default='norm',
        help="Scipy fit function to use for attribution metrics (default: 'norm')"
    )
    parser.add_argument(
        '-y', '--year',
        type=int,
        help="Year of the extreme event that is being analyzed"
    )

#####################################################################

def parser_xclim(subparsers):

    parser = subparsers.add_parser(
        'xclim-indice', 
        help='Function used to wrap xclim library and calculate the indice'
    )
    parser.add_argument(
        '-i', '--ifile', 
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '--xclim-function', 
        required=True,
        help="Name of the function that will calculate the indice. See: https://xclim.readthedocs.io/en/stable/api_indicators.html"
    )
    parser.add_argument(
        '-k', '--kwargs', 
        nargs='*', 
        action=ParseKwargs,
        help='Arguments used in the xclim function. See: https://xclim.readthedocs.io/en/stable/api_indicators.html'
    )

#####################################################################

def parser_scaling(subparsers):

    parser = subparsers.add_parser(
        'scale', 
        help='Function used to scale data by a given climatology (add or divide)'
    )
    parser.add_argument(
        '-i', '--ifile', 
        required=True,
        help="Path to the input dataset"
    )
    parser.add_argument(
        '-c', '--clim', 
        required=True,
        help="Path to the climatology dataset"
    )
    parser.add_argument(
        '-o', '--ofile',
        required=True,
        help="Path to the output dataset"
    )
    parser.add_argument(
        '-d', '--data-source', 
        choices=['multi-file', 'single-file'],
        required=True,
        help="Specify the data source type: 'multi-file' or 'single-file'"
    )
    parser.add_argument(
        '-v', '--variable',
        required=True,
        type=str,
        default='tas',
        help="Variable name to use for attribution metrics (default: 'tas')"
    )
    parser.add_argument(
        '--idate',
        type=str,
        help="Initial date used to calculate the climatology"
    )
    parser.add_argument(
        '--edate',
        type=str,
        help="End date used to calculate the climatology"
    )
    parser.add_argument(
        '--method',
        choices=['add', 'mult'],
        default='add',
        help="Method used to scale the data (add or mult)"
    )

#####################################################################
