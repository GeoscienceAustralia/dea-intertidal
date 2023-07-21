import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

from shapely.geometry import Point
from shapely.ops import unary_union
import sunriset
from math import ceil
import datetime
from datetime import timedelta
import pytz
from pyproj import CRS
from pyproj import Transformer
from dea_tools.coastal import pixel_tides
from intertidal.utils import round_date_strings

def exposure(
    start_date,
    end_date,
    dem,
    time_range,
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
    timezones = None,
    filters = None, ## Currently designed for a single output eg winter, low-tide. Needs some reworking to consider multiple outputs
):
    """
    Calculate exposure percentage for each pixel based on tide-height
    differences between the elevation value and percentile values of the
    tide model for a given time range.

    Parameters
    ----------
    start_date  :  str
        A four-digit single year string, should match the query and
        start_date used for the elevation calculation e.g. '2019'
    end_date  :  str
        A four-digit single year string, should match the query and
        end_date used for the elevation calculation e.g. '2021'
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data
        and coordinates and attributes metadata.
    time_range : tuple
        Tuple containing start and end time of time range to be used for
        tide model in the format of "YYYY-MM-DD".
    tide_model : str, optional
        The tide model used to model tides, as supported by the `pyTMD`
        Python package. Options include:
        - "FES2014" (default; pre-configured on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `dea_tools.coastal.model_tides`.
    filters  :  list of strings, optional
        A list of customisation options to input into the tidal
        modelling to calculate exposure. Selections currently combine
        to produce a single exposure output e.g. winter, low-tide
        TODO: rework to product multiple outputs
        NOTE: do not input multiple temporal options as code is likely
        to fail e.g summer, June
    timezones  :  dict, optional
        For calculation of day and night exposure, timezones is a 
        dictionary of paths to relevant timezone shapefiles for your
        area of interest. Defaults to None

    Returns
    -------
    exposure : xarray.DataArray
        An xarray.DataArray containing the percentage time 'exposure' of
        each pixel from seawater for the duration of the modelling
        period `timerange`.
    tide_cq : xarray.DataArray
        An xarray.DataArray containing the quantiled high temporal
        resolution tide modelling for each pixel. Dimesions should be
        'quantile', 'x' and 'y'.

    Notes
    -----
    - The tide-height percentiles range from 0 to 100, divided into 101
    equally spaced values.
    - The 'diff' variable is calculated as the absolute difference
    between tide model percentile value and the DEM value at each pixel.
    - The 'idxmin' variable is the index of the smallest tide-height
    difference (i.e., maximum similarity) per pixel and is equivalent
    to the exposure percent.
    """

    # Filter the input timerange to include only dates or tide ranges of interest
    if filters is not None:
        for x in filters:
            if x == 'dry':
                time_range = time_range.drop(time_range[(time_range.month == 10) ## Wet season: Oct-Mar
                        |(time_range.month == 11)
                        |(time_range.month == 12)
                        |(time_range.month == 1)
                        |(time_range.month == 2)
                        |(time_range.month == 3)
                        ])
            elif x == 'wet':
                time_range = time_range.drop(time_range[(time_range.month == 4) ## Dry season: Apr-Sep
                        |(time_range.month == 5)
                        |(time_range.month == 6)
                        |(time_range.month == 7)
                        |(time_range.month == 8)
                        |(time_range.month == 9)
                        ])
            elif x == 'summer':
                time_range = time_range.drop(time_range[time_range.quarter != 1])
            elif x == 'autumn':
                time_range = time_range.drop(time_range[time_range.quarter != 2])
            elif x == 'winter':
                time_range = time_range.drop(time_range[time_range.quarter != 3])
            elif x == 'spring':
                time_range = time_range.drop(time_range[time_range.quarter != 4])
            elif x == 'Jan':
                time_range = time_range.drop(time_range[time_range.month != 1])
            elif x == 'Feb':
                time_range = time_range.drop(time_range[time_range.month != 2])
            elif x == 'Mar':
                time_range = time_range.drop(time_range[time_range.month != 3])
            elif x == 'Apr':
                time_range = time_range.drop(time_range[time_range.month != 4])
            elif x == 'May':
                time_range = time_range.drop(time_range[time_range.month != 5])
            elif x == 'Jun':
                time_range = time_range.drop(time_range[time_range.month != 6])
            elif x == 'Jul':
                time_range = time_range.drop(time_range[time_range.month != 7])
            elif x == 'Aug':
                time_range = time_range.drop(time_range[time_range.month != 8])
            elif x == 'Sep':
                time_range = time_range.drop(time_range[time_range.month != 9])
            elif x == 'Oct':
                time_range = time_range.drop(time_range[time_range.month != 10])
            elif x == 'Nov':
                time_range = time_range.drop(time_range[time_range.month != 11])
            elif x == 'Dec':
                time_range = time_range.drop(time_range[time_range.month != 12])
            elif x == ('Daylight' and 'Night') or ('Daylight' or 'Night'):
                ## Pip install sunriset module for calculate local sunrise and sunset times
                # !pip install sunriset
                
                ## Determine the timezone of the pixels
                ## Bring in the state polygons (note: native crs = epsg:4283)
                wa = gpd.read_file(timezones['wa'])
                nt = gpd.read_file(timezones['nt'])
                sa = gpd.read_file(timezones['sa'])
                qld = gpd.read_file(timezones['qld'])
                nsw = gpd.read_file(timezones['nsw'])
                vic = gpd.read_file(timezones['vic'])
                tas = gpd.read_file(timezones['tas'])

                # Merge the polygons to create single state/territory boundaries
                wa = gpd.GeoSeries(unary_union(wa.geometry))
                nt = gpd.GeoSeries(unary_union(nt.geometry))
                sa = gpd.GeoSeries(unary_union(sa.geometry))
                qld = gpd.GeoSeries(unary_union(qld.geometry))
                nsw = gpd.GeoSeries(unary_union(nsw.geometry))
                vic = gpd.GeoSeries(unary_union(vic.geometry))
                tas = gpd.GeoSeries(unary_union(tas.geometry))
                
                ## Note: day and night times will be calculated once per area-of-interest(ds)
                ## for the median latitude and longitude position of the aoi
                tidepost_lat_3577 = dem.x.median(dim='x').values
                tidepost_lon_3577 = dem.y.median(dim='y').values
                
                ## Translate the crs of the tidepost to determine (1) local timezone
                ## and (2) the local sunrise and sunset times
                
                ## Set the Datacube native crs (GDA/Aus Albers (meters))
                crs_3577 = CRS.from_epsg(3577)
                
                ## (1) Create a transform to convert default epsg3577 coords to epsg4283 to compare 
                ## against state/territory boundary polygons and assign a timezone

                ## GDA94 CRS (degrees)
                crs_4283 = CRS.from_epsg(4283)
                ## Transfer coords from/to
                transformer_4283 = Transformer.from_crs(crs_3577, crs_4283) 
                ## Translate tidepost coords
                tidepost_lat_4283, tidepost_lon_4283 = transformer_4283.transform(tidepost_lat_3577,
                                                                                  tidepost_lon_3577)
                ## Coordinate point to test for timezone   
                point_4283 = Point(tidepost_lon_4283, tidepost_lat_4283)
                
                ## (2) Create a transform to convert default epsg3577 coords to epsg4326 for use in 
                ## sunise/sunset library

                ## World WGS84 (degrees)
                crs_4326 = CRS.from_epsg(4326) 
                ## Transfer coords from/to
                transformer_4326 = Transformer.from_crs(crs_3577, crs_4326)
                ## Translate the tidepost coords
                tidepost_lat_4326, tidepost_lon_4326 = transformer_4326.transform(tidepost_lat_3577,
                                                                                  tidepost_lon_3577)
                ## Coordinate point to locate the sunriset calculation
                point_4326 = Point(tidepost_lon_4326, tidepost_lat_4326)
                
                ## Set the local timezone for the analysis area of interest
                if wa.contains(point_4283)[0] == True:
                    timezone = 'Australia/West'
                    local_tz = 8
                    
                elif nt.contains(point_4283)[0] == True:
                    timezone = 'Australia/North'
                    local_tz = 9.5
                    
                elif sa.contains(point_4283)[0] == True:
                    timezone = 'Australia/South'
                    local_tz = 9.5
                    
                elif qld.contains(point_4283)[0] == True:
                    timezone = 'Australia/Queensland'
                    local_tz = 10
                    
                elif nsw.contains(point_4283)[0] == True:
                    timezone = 'Australia/NSW'
                    local_tz = 10
                    
                elif vic.contains(point_4283)[0] == True:
                    timezone = 'Australia/Victoria'
                    local_tz = 10
                    
                elif tas.contains(point_4283)[0] == True:
                    timezone = 'Australia/Tasmania'
                    local_tz = 10
                
                ## Calculate the local sunrise and sunset times
                # Place start and end dates in correct format
                start=round_date_strings(start_date, round_type="start")
                end=round_date_strings(end_date, round_type="end")
                startdate = datetime.date(pd.to_datetime(start).year, 
                                          pd.to_datetime(start).month, 
                                          pd.to_datetime(start).day)

                # Make 'all_timerange' time-zone aware
                localtides = time_range.tz_localize(tz=pytz.UTC).tz_convert(timezone)

                # Replace the UTC datetimes from all_timerange with local times
                modelledtides = pd.DataFrame(index = localtides)

                # Return the difference in years for the time-period. 
                # Round up to ensure all modelledtide datetimes are captured in the solar model
                diff = pd.to_datetime(end) - pd.to_datetime(start)
                diff = int(ceil(diff.days/365))
                
                ## Model sunrise and sunset
                sun_df = sunriset.to_pandas(startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff)

                ## Set the index as a datetimeindex to match the modelledtide df
                sun_df = sun_df.set_index(pd.DatetimeIndex(sun_df.index))

                ## Append the date to each Sunrise and Sunset time
                sun_df['Sunrise dt'] = sun_df.index + sun_df['Sunrise']
                sun_df['Sunset dt'] = sun_df.index + (sun_df['Sunset'])

                ## Create new dataframes where daytime and nightime datetimes are recorded, then merged 
                ## on a new `Sunlight` column
                daytime=pd.DataFrame(data = 'Sunrise', index=sun_df['Sunrise dt'], columns=['Sunlight'])
                nighttime=pd.DataFrame(data = 'Sunset', index=sun_df['Sunset dt'], columns=['Sunlight'])
                DayNight = pd.concat([daytime, nighttime], join='outer')
                DayNight.sort_index(inplace=True)
                DayNight.index.rename('Datetime', inplace=True)

                ## Create an xarray object from the merged day/night dataframe
                day_night = xr.Dataset.from_dataframe(DayNight)

                ## Remove local timezone timestamp column in modelledtides dataframe. Xarray doesn't handle 
                ## timezone aware datetimeindexes 'from_dataframe' very well.
                modelledtides.index = modelledtides.index.tz_localize(tz=None)

                ## Create an xr Dataset from the modelledtides pd.dataframe
                mt = modelledtides.to_xarray()

                ## Filter the modelledtides (mt) by the daytime, nighttime datetimes from the sunriset module
                ## Modelled tides are designated as either day or night by propogation of the last valid index 
                ## value forward
                Solar=day_night.sel(Datetime=mt.index, method='ffill')

                ## Assign the day and night tideheight datasets
                SolarDayTides = mt.where(Solar.Sunlight=='Sunrise', drop=True)
                SolarNightTides = mt.where(Solar.Sunlight=='Sunset', drop=True)

                ## Extract DatetimeIndexes to use in exposure calculations
                all_timerange_day = pd.DatetimeIndex(SolarDayTides.index)
                all_timerange_night = pd.DatetimeIndex(SolarNightTides.index)
                
                if 'Daylight':
                    time_range = all_timerange_day
                if 'Night':
                    time_range = all_timerange_night
                
    
    # Create the tide-height percentiles from which to calculate
    # exposure statistics
    pc_range = np.linspace(0, 1, 101)

    # Run the pixel_tides function with the calculate_quantiles option.
    # For each pixel, an array of tideheights is returned, corresponding
    # to the percentiles from pc_range of the timerange-tide model that
    # each tideheight appears in the model.
    tide_cq, _ = pixel_tides(
        dem,
        resample=True,
        calculate_quantiles=pc_range,
        times=time_range,
        model=tide_model,
        directory=tide_model_dir,
        cutoff=np.inf,
    )

    # Calculate the tide-height difference between the elevation value and
    # each percentile value per pixel
    diff = abs(tide_cq - dem)

    # Take the percentile of the smallest tide-height difference as the
    # exposure % per pixel
    idxmin = diff.idxmin(dim="quantile")

    # Convert to percentage
    exposure = idxmin * 100

    return exposure, tide_cq, time_range
