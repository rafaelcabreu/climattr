import geopandas as gpd
import pandas as pd

from unidecode import unidecode


def geolocate_dataframe(
    dataframe: pd.DataFrame, 
    location: gpd.GeoDataFrame, 
    dataframe_column: str = 'city_name', 
    location_column: str = 'NM_MUN') -> gpd.GeoDataFrame:
    """
    Geolocates a Pandas DataFrame by joining it with a GeoPandas GeoDataFrame, 
    matching based on city names or other spatial feature, and returns a 
    GeoDataFrame containing the geometries for each spatial feature in the 
    original dataframe.

    This function is useful for integrating geospatial data into a regular DataFrame 
    by adding geometry information (e.g., polygons or points) for locations such as 
    cities, states, countries, etc..

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input Pandas DataFrame containing data for the spatial feature.
    
    location : gpd.GeoDataFrame
        The GeoDataFrame containing geometry information for the spatial feature, 
        which will be joined with the input dataframe.
    
    dataframe_column : str, optional
        The name of the column in the dataframe that contains the spatial feature 
        name to be used for joining. The default is 'city_name'.
    
    location_column : str, optional
        The name of the column in the location GeoDataFrame that contains the 
        spatial feature names to be used for joining. The default is 'NM_MUN'.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the original data from the input dataframe along 
        with the corresponding geometries from the location GeoDataFrame.
    
    Notes
    -----
    - The function assumes that the location names in both the dataframe and 
      GeoDataFrame may contain accentuation, which is removed for consistent joining.
    - The final result will contain all columns from the original dataframe and 
      the geometry column from the location GeoDataFrame.
    - If a match is not found for a particular city or municipality in the location 
      GeoDataFrame, the corresponding geometry will be NaN.
    
    Example
    -------
    Suppose we have a dataframe with dengue cases by city and a GeoDataFrame 
    with geometries for cities:

    >>> dataframe = pd.DataFrame({
    ...     'city_name': ['São Paulo', 'Rio de Janeiro', 'Brasília'],
    ...     'dengue_cases': [500, 300, 150]
    ... })
    
    >>> location = gpd.read_file('cities_shapefile.shp')  # A GeoDataFrame with city geometries
    
    >>> result = geolocate_dataframe(dataframe, location, 'city_name', 'NM_MUN')
    
    The resulting GeoDataFrame will contain the dengue cases along with the geometries 
    for each city.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame resulting from the join of the original dataframe and the 
        geolocation information, including a geometry column.
    """
    # remove accentuation from the cities name to join with the datasets
    location = location.rename(columns={location_column: dataframe_column})
    location[dataframe_column] = location[dataframe_column].apply(
        lambda x: str(unidecode(x)).upper()
    )

    # do the same for the dataframe
    dataframe[dataframe_column] = dataframe[dataframe_column].apply(
        lambda x: str(unidecode(x)).upper()
    )

    # join with impacts dataset
    location_dataframe = location.set_index([dataframe_column]).join(
        dataframe.set_index([dataframe_column]), how='left'
    ).reset_index()

    location_dataframe = location_dataframe[
        list(dataframe.columns) + ['geometry']
    ]

    return location_dataframe

#####################################################################

def aggregate_time_dataframe(
    dataframe: pd.DataFrame, 
    method: str, 
    date: str = 'date', 
    freq: str = 'YE', 
    keep_location: bool = True,
    location_column: str = 'city_name') -> pd.DataFrame:
    """
    Aggregates the data in a Pandas DataFrame based on a time frequency, using a 
    specified aggregation method, and optionally keeping a location indicator 
    (e.g., city or region).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input Pandas DataFrame containing the data to be aggregated. The DataFrame 
        should include a date column and, optionally, a location column.
    
    method : str
        The aggregation method to be applied, such as 'mean', 'sum', 'min', 'max', 
        'count' etc. This should be a valid Pandas aggregation method.
    
    date : str, optional, default = 'date'
        The column name in the dataframe that contains the date information. 
        This column will be converted to a datetime object if it is not already.
    
    freq : str, optional, default = 'YE'
        The frequency at which the data should be aggregated. The default is 
        'YE' (year-end). This can be any valid Pandas offset alias (e.g., 'M' 
        for monthly, 'Q' for quarterly, 'D' for daily, 'A' for annually, etc.).
    
    keep_location : bool, optional, default = True
        Whether to keep the location column (e.g., city or region) during the 
        aggregation. If True, the data will be aggregated by both time and location. 
        If False, the data will be aggregated only by time.
    
    location_column : str, optional, default = 'city_name'
        The name of the column in the dataframe that represents the location 
        (e.g., city or region). This column is used if `keep_location` is set to True.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with the aggregated data, based on the specified frequency 
        and method. The resulting DataFrame will include a date column (with the 
        aggregated time periods) and the aggregated values, along with the location 
        column if `keep_location` is True.
    
    
    Example
    -------
    Suppose you have a dataframe containing city-level data for daily temperatures, 
    and you want to aggregate it to annual averages, while keeping the city information.

    >>> dataframe = pd.DataFrame({
    ...     'city_name': ['São Paulo', 'São Paulo', 'Rio de Janeiro', 'Rio de Janeiro'],
    ...     'date': ['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-02'],
    ...     'temperature': [25.0, 26.0, 30.0, 31.0]
    ... })
        
    >>> result = aggregate_time_dataframe(dataframe, method='mean', freq='YE', keep_location=True)

    The result will be a DataFrame where the temperatures are aggregated by city 
    and year, with the average temperature for each city over that period.
    
    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame containing the date, aggregated values, and 
        location (if `keep_location` is True).
    """
    # make sure the date is formatted as datetime object
    dataframe[date] = pd.to_datetime(dataframe[date])
    dataframe = dataframe.dropna(subset=[date])

    # aggregate data by time keeping the location indicator
    if keep_location:
        aggregated_dataframe = getattr(
            dataframe.groupby(location_column).resample(freq, on=date), method
        )().drop(location_column, axis=1).reset_index()
    else:
        aggregated_dataframe = getattr(
            dataframe.resample(freq, on=date), method
        )().reset_index() 

    return aggregated_dataframe

#####################################################################

def aggregate_spatial_dataframe(
    geodataframe: gpd.GeoDataFrame, 
    location: gpd.GeoDataFrame,
    method: str,
    location_column: str = 'NM_MESO',
    keep_date: bool = True,
    date_column: str = 'date') -> gpd.GeoDataFrame:
    """
    Aggregates a GeoDataFrame spatially by joining it with another GeoDataFrame 
    representing locations (e.g., regions or administrative boundaries) and 
    applying a specified aggregation method.

    This function performs a spatial join between the input GeoDataFrame and a 
    GeoDataFrame containing location geometries (e.g., administrative boundaries) 
    to associate each point or polygon with a specific location. The resulting 
    GeoDataFrame is then aggregated based on the chosen method (e.g., 'mean', 'sum') 
    and either by location alone or by both location and date.

    Parameters
    ----------
    geodataframe : gpd.GeoDataFrame
        The input GeoDataFrame containing spatial data that needs to be aggregated.
    
    location : gpd.GeoDataFrame
        The GeoDataFrame containing location boundaries (e.g., regions, municipalities) 
        to spatially join with the input `geodataframe`. It should have a geometry 
        column and a location identifier column.
    
    method : str
        The aggregation method to be applied, such as 'mean', 'sum', 'min', 'max', 
        etc. This should be a valid Pandas aggregation method.
    
    location_column : str, optional, default = 'NM_MESO'
        The name of the column in the `location` GeoDataFrame that represents the 
        location identifier (e.g., region name, district name).
    
    keep_date : bool, optional, default = True
        Whether to keep the date column during aggregation. If `True`, the data 
        will be aggregated by both location and date. If `False`, the data will 
        be aggregated only by location.
    
    date_column : str, optional, default = 'date'
        The name of the column in the `geodataframe` that represents the date 
        (if `keep_date` is True).

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the data aggregated spatially by location 
        (and by date, if applicable), based on the specified aggregation method.
    
    Example
    -------
    Suppose you have a GeoDataFrame of temperature readings and a GeoDataFrame 
    of region boundaries. You want to aggregate the temperature readings by region 
    and year.

    >>> geodataframe = gpd.GeoDataFrame({
    ...     'temperature': [30.0, 32.5, 28.0, 29.5],
    ...     'geometry': [Point(1, 1), Point(1, 2), Point(2, 1), Point(2, 2)],
    ...     'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']
    ... })
    
    >>> location = gpd.read_file('regions_shapefile.shp')  # GeoDataFrame with region geometries

    >>> result = aggregate_spatial_dataframe(
    ...     geodataframe, location, method='mean', location_column='region_name', keep_date=True
    ... )

    This will return a GeoDataFrame with the average temperature for each region 
    and date.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame with aggregated data by location and optionally by date.
    """
    # spatial join geodataframe with location based on the geometry from gdf
    # that are within location
    geodataframe_location = gpd.sjoin(
        geodataframe, location[[location_column, 'geometry']], predicate='within'
    ).drop('index_right', axis=1)

    if keep_date:
        aggregators = [location_column, date_column]
    else:
        aggregators = [location_column]

    geodataframe_location = getattr(
        geodataframe_location.groupby(aggregators), method
    )()

    return geodataframe_location

#####################################################################
