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
import re
import pytz
from pyproj import CRS, Transformer
from scipy.signal import argrelmax, argrelmin
from numpy import interp

from dea_tools.coastal import pixel_tides, model_tides
from intertidal.tide_modelling import pixel_tides_ensemble
from intertidal.utils import configure_logging, round_date_strings

def temporal_filters(x,
                     timeranges,
                     time_range,
                     dem):
    
    if x == 'dry':
        timeranges['dry'] = time_range.drop(time_range[(time_range.month == 10) ## Wet season: Oct-Mar
                |(time_range.month == 11)
                |(time_range.month == 12)
                |(time_range.month == 1)
                |(time_range.month == 2)
                |(time_range.month == 3)
                ])
    elif x == 'wet':
        timeranges['wet'] = time_range.drop(time_range[(time_range.month == 4) ## Dry season: Apr-Sep
                |(time_range.month == 5)
                |(time_range.month == 6)
                |(time_range.month == 7)
                |(time_range.month == 8)
                |(time_range.month == 9)
                ])
    elif x == 'summer':
        timeranges['summer'] = time_range.drop(time_range[time_range.quarter != 1])
    elif x == 'autumn':
        timeranges['autumn'] = time_range.drop(time_range[time_range.quarter != 2])
    elif x == 'winter':
        timeranges['winter'] = time_range.drop(time_range[time_range.quarter != 3])
    elif x == 'spring':
        timeranges['spring'] = time_range.drop(time_range[time_range.quarter != 4])
    elif x == 'Jan':
        timeranges['Jan'] = time_range.drop(time_range[time_range.month != 1])
    elif x == 'Feb':
        timeranges['Feb'] = time_range.drop(time_range[time_range.month != 2])
    elif x == 'Mar':
        timeranges['Mar'] = time_range.drop(time_range[time_range.month != 3])
    elif x == 'Apr':
        timeranges['Apr'] = time_range.drop(time_range[time_range.month != 4])
    elif x == 'May':
        timeranges['May'] = time_range.drop(time_range[time_range.month != 5])
    elif x == 'Jun':
        timeranges['Jun'] = time_range.drop(time_range[time_range.month != 6])
    elif x == 'Jul':
        timeranges['Jul'] = time_range.drop(time_range[time_range.month != 7])
    elif x == 'Aug':
        timeranges['Aug'] = time_range.drop(time_range[time_range.month != 8])
    elif x == 'Sep':
        timeranges['Sep'] = time_range.drop(time_range[time_range.month != 9])
    elif x == 'Oct':
        timeranges['Oct'] = time_range.drop(time_range[time_range.month != 10])
    elif x == 'Nov':
        timeranges['Nov'] = time_range.drop(time_range[time_range.month != 11])
    elif x == 'Dec':
        timeranges['Dec'] = time_range.drop(time_range[time_range.month != 12])
    elif x in ['Daylight', 'Night']:
        print (f'Calculating {x}')

#         timezones = {'wa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/western_australia/cstwacd_r.shp',
#                      'nt':'../../gdata1/data/boundaries/GEODATA_COAST_100K/northern_territory/cstntcd_r.shp',
#                      'sa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/south_australia/cstsacd_r.shp',
#                      'qld':'../../gdata1/data/boundaries/GEODATA_COAST_100K/queensland/cstqldmd_r.shp',
#                      'nsw':'../../gdata1/data/boundaries/GEODATA_COAST_100K/new_south_wales/cstnswcd_r.shp',
#                      'vic':'../../gdata1/data/boundaries/GEODATA_COAST_100K/victoria/cstviccd_r.shp',
#                      'tas':'../../gdata1/data/boundaries/GEODATA_COAST_100K/tasmania/csttascd_r.shp'
#                      }

#         ## Determine the timezone of the pixels
#         ## Bring in the state polygons (note: native crs = epsg:4283)
#         wa = gpd.read_file(timezones['wa'])
#         nt = gpd.read_file(timezones['nt'])
#         sa = gpd.read_file(timezones['sa'])
#         qld = gpd.read_file(timezones['qld'])
#         nsw = gpd.read_file(timezones['nsw'])
#         vic = gpd.read_file(timezones['vic'])
#         tas = gpd.read_file(timezones['tas'])

#         # Merge the polygons to create single state/territory boundaries
#         wa = gpd.GeoSeries(unary_union(wa.geometry))
#         nt = gpd.GeoSeries(unary_union(nt.geometry))
#         sa = gpd.GeoSeries(unary_union(sa.geometry))
#         qld = gpd.GeoSeries(unary_union(qld.geometry))
#         nsw = gpd.GeoSeries(unary_union(nsw.geometry))
#         vic = gpd.GeoSeries(unary_union(vic.geometry))
#         tas = gpd.GeoSeries(unary_union(tas.geometry))

#         ## Note: day and night times will be calculated once per area-of-interest(ds)
#         ## for the median latitude and longitude position of the aoi
#         tidepost_lat_3577 = dem.x.median(dim='x').values
#         tidepost_lon_3577 = dem.y.median(dim='y').values

#         ## Set the Datacube native crs (GDA/Aus Albers (meters))
#         crs_3577 = CRS.from_epsg(3577)

#         ## Translate the crs of the tidepost to determine (1) local timezone
#         ## and (2) the local sunrise and sunset times:

#         ## (1) Create a transform to convert default epsg3577 coords to epsg4283 to compare 
#         ## against state/territory boundary polygons and assign a timezone

#         ## GDA94 CRS (degrees)
#         crs_4283 = CRS.from_epsg(4283)
#         ## Transfer coords from/to
#         transformer_4283 = Transformer.from_crs(crs_3577, crs_4283) 
#         ## Translate tidepost coords
#         tidepost_lat_4283, tidepost_lon_4283 = transformer_4283.transform(tidepost_lat_3577,
#                                                                           tidepost_lon_3577)
#         ## Coordinate point to test for timezone   
#         point_4283 = Point(tidepost_lon_4283, tidepost_lat_4283)

#         ## (2) Create a transform to convert default epsg3577 coords to epsg4326 for use in 
#         ## sunise/sunset library

#         ## World WGS84 (degrees)
#         crs_4326 = CRS.from_epsg(4326) 
#         ## Transfer coords from/to
#         transformer_4326 = Transformer.from_crs(crs_3577, crs_4326)
#         ## Translate the tidepost coords
#         tidepost_lat_4326, tidepost_lon_4326 = transformer_4326.transform(tidepost_lat_3577,
#                                                                           tidepost_lon_3577)

#         # Identify the central coordinate directly from the dem GeoBox
#         tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs("EPSG:4326").coords[0]
        
#         ## Coordinate point to locate the sunriset calculation
#         point_4326 = Point(tidepost_lon_4326, tidepost_lat_4326)

#         ## Set the local timezone for the analysis area of interest
#         if wa.contains(point_4283)[0] == True:
#             timezone = 'Australia/West'
#             local_tz = 8

#         elif nt.contains(point_4283)[0] == True:
#             timezone = 'Australia/North'
#             local_tz = 9.5

#         elif sa.contains(point_4283)[0] == True:
#             timezone = 'Australia/South'
#             local_tz = 9.5

#         elif qld.contains(point_4283)[0] == True:
#             timezone = 'Australia/Queensland'
#             local_tz = 10

#         elif nsw.contains(point_4283)[0] == True:
#             timezone = 'Australia/NSW'
#             local_tz = 10

#         elif vic.contains(point_4283)[0] == True:
#             timezone = 'Australia/Victoria'
#             local_tz = 10

#         elif tas.contains(point_4283)[0] == True:
#             timezone = 'Australia/Tasmania'
#             local_tz = 10
        # Identify the central coordinate directly from the dem GeoBox
        tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs("EPSG:4326").coords[0]

        ## Coordinate point to locate the sunriset calculation
        point_4326 = Point(tidepost_lon_4326, tidepost_lat_4326)

        ## Calculate the local sunrise and sunset times
        # Place start and end dates in correct format
        start = time_range[0]
        end = time_range[-1]
        startdate = datetime.date(pd.to_datetime(start).year, 
                                  pd.to_datetime(start).month, 
                                  pd.to_datetime(start).day)

        # Make 'all_timerange' time-zone aware
        localtides = time_range.tz_localize(tz=pytz.UTC)#.tz_convert(timezone)

        # Replace the UTC datetimes from all_timerange with local times
        modelledtides = pd.DataFrame(index = localtides)

        # Return the difference in years for the time-period. 
        # Round up to ensure all modelledtide datetimes are captured in the solar model
        diff = pd.to_datetime(end) - pd.to_datetime(start)
        diff = int(ceil(diff.days/365))

        # Set UTC time as the local_tz
        local_tz=0
        
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

        if x == 'Daylight':
            timeranges['Daylight'] = all_timerange_day
        if x == 'Night':
            timeranges['Night'] = all_timerange_night
    
    return timeranges

def spatial_filters(
                    modelled_freq,
                    x,
                    stacked_everything,
                    ModelledTides,
                    timeranges,
                    calculate_quantiles,
                    tide_cq_dict,
                    dem,
                    exposure
                    ):
    
    # Extract the modelling freq units
    # Split the number and text characters in modelled_freq
    freq_time = int(re.findall(r'(\d+)(\w+)', modelled_freq)[0][0])
    freq_unit = str(re.findall(r'(\d+)(\w+)', modelled_freq)[0][-1])

    # Extract the number of modelled timesteps per 14 days (half lunar cycle) for neap/spring calcs
    mod_timesteps = pd.Timedelta((29.5/2),"d")/pd.Timedelta(freq_time, freq_unit)

    ## Identify kwargs for peak detection algorithm
    order=(int(mod_timesteps/2))

    ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle

    if x in ['Spring_high', 'Spring_low', 'Neap_high', 'Neap_low']:

        print (f'Calculating {x}')

        #1D tide modelling workflow

        ## apply the peak detection routine
        if x in ['Spring_high', 'Neap_high']:
            stacked_everything_peaks = argrelmax(stacked_everything.values, order=order)[0]
        if x in ['Spring_low', 'Neap_low']:
            stacked_everything_peaks = argrelmin(stacked_everything.values, order=order)[0]
        if x == 'Neap_high':       
            ## apply the peak detection routine to calculate all the high tide maxima
            Max_testarray = argrelmax(stacked_everything.values)[0]

            Max_testarray = stacked_everything.isel(time=Max_testarray)
            ## extract all hightide peaks
            Max_testarray = ModelledTides.to_dataset().sel(time=Max_testarray.time)
            ## repeat the peak detection to identify neap high tides (minima in the high tide maxima)
            stacked_everything2 = Max_testarray.mean(dim=["x","y"])
            ## extract neap high tides based on a half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
            order_nh = int(ceil((len(Max_testarray.time)/(len(stacked_everything_peaks))/2)))
            ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
            neap_peaks = argrelmin(stacked_everything2.tide_m.values, order=order_nh)[0] 

        if x == 'Neap_low':       
            ## apply the peak detection routine to calculate all the high tide maxima
            Max_testarray = argrelmin(stacked_everything.values)[0]

            Max_testarray = stacked_everything.isel(time=Max_testarray)
            ## extract all hightide peaks
            Max_testarray = ModelledTides.to_dataset().sel(time=Max_testarray.time)
            ## repeat the peak detection to identify neap high tides (maxima in the low tide minima)
            stacked_everything2 = Max_testarray.mean(dim=["x","y"])
            ## extract neap low tides based on 14 day half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
            order_nh = int(ceil((len(Max_testarray.time)/(len(stacked_everything_peaks))/2)))
            ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
            neap_peaks = argrelmax(stacked_everything2.tide_m.values, order=order_nh)[0]

        if x in ['Neap_high', 'Neap_low']: 
            ## extract neap high tides
            neappeaks = Max_testarray.isel(time=neap_peaks)

            timeranges[str(x)]=pd.to_datetime(neappeaks.time)

            # Extract the peak height dates
            tide_cq = neappeaks.quantile(q=calculate_quantiles,dim='time')

        if x in ['Spring_high', 'Spring_low']:  
            # select for indices associated with peaks
            springpeaks = stacked_everything.isel(time=stacked_everything_peaks)

            # Select dates associated with detected peaks
            springpeaks = ModelledTides.to_dataset().sel(time=springpeaks.time)

            # Save datetimes for calculation of combined filter exposure
            timeranges[str(x)]=pd.to_datetime(springpeaks.time)

            # Extract the peak height dates
            tide_cq = springpeaks.quantile(q=calculate_quantiles,dim='time')

    if x == 'Hightide':
        print ('Calculating Hightide')
        
        # calculate all the high tide maxima
        high_peaks = argrelmax(stacked_everything.values)[0]

        # extract all hightide peaks
        high_peaks2 = stacked_everything.isel(time=high_peaks)

        # identify all lower hightide peaks
        lowhigh_peaks = argrelmin(high_peaks2.values)[0]

        # extract all lower hightide peaks
        lowhigh_peaks2 = high_peaks2.isel(time=lowhigh_peaks)

        # interpolate the lower hightide curve
        low_high_linear = interp(np.arange(0,len(stacked_everything)),
                                 high_peaks[lowhigh_peaks],
                                 lowhigh_peaks2.values)
        # Extract all tides higher than/equal to the extrapolated lowest high tide line
        hightide = stacked_everything.where(stacked_everything >= low_high_linear, drop=True)
        
        ## Save datetimes for calculation of combined filter exposure
        timeranges[str(x)] = pd.to_datetime(hightide.time)

        tide_cq = hightide.quantile(q=calculate_quantiles,dim='time').to_dataset()

    if x == 'Lowtide':
        print ('Calculating Lowtide')
        
        # calculate all the low tide maxima
        low_peaks = argrelmin(stacked_everything.values)[0]

        # extract all lowtide peaks
        low_peaks2 = stacked_everything.isel(time=low_peaks)

        # identify all higher lowtide peaks
        highlow_peaks = argrelmax(low_peaks2.values)[0]

        # extract all higher lowtide peaks
        highlow_peaks2 = low_peaks2.isel(time=highlow_peaks)

        # interpolate the higher lowtide curve
        high_low_linear = interp(np.arange(0,len(stacked_everything)),
                                 low_peaks[highlow_peaks],
                                 highlow_peaks2.values)
        # Extract all tides lower than/equal to the extrapolated higher lowtide line
        lowtide = stacked_everything.where(stacked_everything <= high_low_linear, drop=True)
        
        ## Save datetimes for calculation of combined filter exposure
        timeranges[str(x)] = pd.to_datetime(lowtide.time)

        tide_cq = lowtide.quantile(q=calculate_quantiles,dim='time').to_dataset()

    # Add tide_cq to output dict
    tide_cq_dict[str(x)]=tide_cq.tide_m

    # Calculate the tide-height difference between the elevation value and
    # each percentile value per pixel
    diff = abs(tide_cq.tide_m - dem)

    # Take the percentile of the smallest tide-height difference as the
    # exposure % per pixel
    idxmin = diff.idxmin(dim="quantile")

    # Convert to percentage
    exposure[str(x)] = idxmin * 100 
    
    return timeranges, tide_cq_dict, exposure

def exposure(
            dem,
  			# times,
            start_date,
            end_date,
            modelled_freq = "30min",
            tide_model="FES2014",
            tide_model_dir="/var/share/tide_models",
            filters = ['unfiltered'], 
            filters_combined = None,
  			run_id=None,
  			log=None,
            ):
        
    """
    Calculate intertidal exposure for each pixel, indicating the 
    proportion of time that each pixel was "exposed" from tidal
    inundation during the time period of interest.
    
    The exposure calculation is based on tide-height differences between
    the elevation value and modelled tide height percentiles.
    
    Calculate exposure percentage for each pixel based on tide-height
    differences between the elevation value and percentile values of the
    tide model for a given time range.

    Parameters
    ----------
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data
        and coordinates and attributes metadata.
    time_range : tuple
        Tuple containing start and end time of time range to be used for
        tide model in the format of "YYYY-MM-DD".
    modelled_freq  :  str
        A pandas time offset alias for the frequency with which to 
        calculate the tide model during exposure calculations. Examples
        include '30min' for 30 minute cadence or '1h' for a one-hourly 
        cadence. Defaults to '30min'.
    tide_model : str, optional
        The tide model or a list of models used to model tides, as
        supported by the `pyTMD` Python package. Options include:
        - "FES2014" (default; pre-configured on DEA Sandbox)
        - "TPXO9-atlas-v5"
        - "TPXO8-atlas"
        - "EOT20"
        - "HAMTIDE11"
        - "GOT4.10"
        - "ensemble" (experimental: combine all above into single ensemble)
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `dea_tools.coastal.model_tides`.
    filters  :  list of strings, optional
        A list of customisation options to input into the tidal
        modelling to calculate exposure. Defaults to ['unfiltered']
    filters_combined  :  list of two-object tuples, optional
        Defaults to None.
	run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    exposure : xarray.DataArray
        An xarray.Dataset containing an array for each filter of
        the percentage time exposure of each pixel from seawater for 
        the duration of the modelling period `timerange`.
    tide_cq : xarray.DataArray ##Revise to dataset
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
    - filters = 'unfiltered' produces exposure for the full input time 
    period.
    - temporal filters include any of: 'dry', 'wet', 'summer', 'autumn', 
    'winter', 'spring', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Daylight', 'Night'
    - spatial filters include any of: 'Spring_high', 'Spring_low', 
    'Neap_high', 'Neap_low', 'Hightide', 'Lowtide'
    - filters_combined can be any combination of one temporal and one 
    spatial filter
    - if filters is set to `None`, no exposure will be calculated and
    the program will fail unless a tuple is nominated in `filters_combined`

    """
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id
    
    # Create the tide-height percentiles from which to calculate
    # exposure statistics
    calculate_quantiles = np.linspace(0, 1, 101) #nb formerly 'pc_range'

    # Generate range of times covering entire period of satellite record for exposure and bias/offset calculation
    time_range = pd.date_range(
        start=round_date_strings(start_date, round_type="start"),
        end=round_date_strings(end_date, round_type="end"),
        freq=modelled_freq,
    )    
    # Separate 'filters' into spatial and temporal categories to define
    # which exposure workflow to use
    temp_filters = ['dry', 'wet', 'summer', 'autumn', 'winter', 'spring', 'Jan', 'Feb', 'Mar', 'Apr', 
                        'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Daylight', 'Night']
    sptl_filters = ['Spring_high', 'Spring_low', 'Neap_high', 'Neap_low', 'Hightide', 'Lowtide']
        
    ## Create empty datasets to store outputs into
    exposure = xr.Dataset(coords=dict(y=(['y'], dem.y.values),
                                      x=(['x'], dem.x.values)))
    tide_cq_dict = xr.Dataset(coords=dict(y=(['y'], dem.y.values),
                                      x=(['x'], dem.x.values)))

    ## Create an empty dict to store temporal `time_range` variables into
    timeranges = {}   
    
    ## If filter combinations are desired, make sure each filter is calculated individually for later combination
    if filters_combined is not None:
        for x in filters_combined:
            filters.append(str(x[0])) if str(x[0]) not in filters else next
            filters.append(str(x[1])) if x[1] not in filters else next
    
    # if filters is None:
    #     filters.append('unfiltered')
    
    if 'unfiltered' in filters:
        print('Calculating unfiltered exposure')

        if (tide_model[0] == "ensemble") or (tide_model == "ensemble"):
            # Use ensemble model combining multiple input ocean tide models
            tide_cq, _ = pixel_tides_ensemble(
                                    dem,
                                    calculate_quantiles=calculate_quantiles,
                                    times=time_range,
                                    directory=tide_model_dir,
                                    ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
                                    top_n=3,
                                    reduce_method='mean',
                                    resolution=3000,
                                    )

        else:
            # Use single input ocean tide model
            tide_cq, _ = pixel_tides(
                                    dem,
                                    resample=True,
                                    calculate_quantiles=calculate_quantiles,
                                    times=time_range,
                                    model=tide_model,
                                    directory=tide_model_dir,
                                    )       

        # Add tide_cq to output dict
        tide_cq_dict['unfiltered']=tide_cq

        # Calculate the tide-height difference between the elevation value and
        # each percentile value per pixel
        diff = abs(tide_cq - dem)

        # Take the percentile of the smallest tide-height difference as the
        # exposure % per pixel
        idxmin = diff.idxmin(dim="quantile")

        # Convert to percentage
        exposure['unfiltered'] = idxmin * 100

        # return exposure
                
    if any (x in sptl_filters for x in filters):
        if (tide_model[0] == "ensemble") or (tide_model == "ensemble"):
            # Use ensemble model combining multiple input ocean tide models
            ModelledTides, _ = pixel_tides_ensemble(
                dem,
                # calculate_quantiles=calculate_quantiles,
                times=time_range,
                directory=tide_model_dir,
                ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
                top_n=3,
                reduce_method='mean',
                resolution=3000,
            )

        else:
            # Use single input ocean tide model
            ModelledTides, _ = pixel_tides(
                dem,
                # calculate_quantiles=calculate_quantiles,
                times=time_range,
                resample=True,
                model=tide_model,
                directory=tide_model_dir,
            )
           
        ## For use with spatial filter options
        ## stack the y and x dimensions
        # stacked_everything = ModelledTides.stack(z=['y','x']).groupby('z')
        stacked_everything = ModelledTides.mean(dim=["x","y"])
    
    # Filter the input timerange to include only dates or tide ranges of interest
    # if filters is not None:
    for x in filters:
        if x in temp_filters:
            print(f'Calculating temporal filter: {x}')
            
            timeranges = temporal_filters(x,
                                         timeranges,
                                         time_range,
                                         dem)

#             if x == 'dry':
#                 timeranges['dry'] = time_range.drop(time_range[(time_range.month == 10) ## Wet season: Oct-Mar
#                         |(time_range.month == 11)
#                         |(time_range.month == 12)
#                         |(time_range.month == 1)
#                         |(time_range.month == 2)
#                         |(time_range.month == 3)
#                         ])
#             elif x == 'wet':
#                 timeranges['wet'] = time_range.drop(time_range[(time_range.month == 4) ## Dry season: Apr-Sep
#                         |(time_range.month == 5)
#                         |(time_range.month == 6)
#                         |(time_range.month == 7)
#                         |(time_range.month == 8)
#                         |(time_range.month == 9)
#                         ])
#             elif x == 'summer':
#                 timeranges['summer'] = time_range.drop(time_range[time_range.quarter != 1])
#             elif x == 'autumn':
#                 timeranges['autumn'] = time_range.drop(time_range[time_range.quarter != 2])
#             elif x == 'winter':
#                 timeranges['winter'] = time_range.drop(time_range[time_range.quarter != 3])
#             elif x == 'spring':
#                 timeranges['spring'] = time_range.drop(time_range[time_range.quarter != 4])
#             elif x == 'Jan':
#                 timeranges['Jan'] = time_range.drop(time_range[time_range.month != 1])
#             elif x == 'Feb':
#                 timeranges['Feb'] = time_range.drop(time_range[time_range.month != 2])
#             elif x == 'Mar':
#                 timeranges['Mar'] = time_range.drop(time_range[time_range.month != 3])
#             elif x == 'Apr':
#                 timeranges['Apr'] = time_range.drop(time_range[time_range.month != 4])
#             elif x == 'May':
#                 timeranges['May'] = time_range.drop(time_range[time_range.month != 5])
#             elif x == 'Jun':
#                 timeranges['Jun'] = time_range.drop(time_range[time_range.month != 6])
#             elif x == 'Jul':
#                 timeranges['Jul'] = time_range.drop(time_range[time_range.month != 7])
#             elif x == 'Aug':
#                 timeranges['Aug'] = time_range.drop(time_range[time_range.month != 8])
#             elif x == 'Sep':
#                 timeranges['Sep'] = time_range.drop(time_range[time_range.month != 9])
#             elif x == 'Oct':
#                 timeranges['Oct'] = time_range.drop(time_range[time_range.month != 10])
#             elif x == 'Nov':
#                 timeranges['Nov'] = time_range.drop(time_range[time_range.month != 11])
#             elif x == 'Dec':
#                 timeranges['Dec'] = time_range.drop(time_range[time_range.month != 12])
#             elif x == 'Daylight' or 'Night': 

#                 timezones = {'wa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/western_australia/cstwacd_r.shp',
#                              'nt':'../../gdata1/data/boundaries/GEODATA_COAST_100K/northern_territory/cstntcd_r.shp',
#                              'sa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/south_australia/cstsacd_r.shp',
#                              'qld':'../../gdata1/data/boundaries/GEODATA_COAST_100K/queensland/cstqldmd_r.shp',
#                              'nsw':'../../gdata1/data/boundaries/GEODATA_COAST_100K/new_south_wales/cstnswcd_r.shp',
#                              'vic':'../../gdata1/data/boundaries/GEODATA_COAST_100K/victoria/cstviccd_r.shp',
#                              'tas':'../../gdata1/data/boundaries/GEODATA_COAST_100K/tasmania/csttascd_r.shp'
#                              }

#                 ## Determine the timezone of the pixels
#                 ## Bring in the state polygons (note: native crs = epsg:4283)
#                 wa = gpd.read_file(timezones['wa'])
#                 nt = gpd.read_file(timezones['nt'])
#                 sa = gpd.read_file(timezones['sa'])
#                 qld = gpd.read_file(timezones['qld'])
#                 nsw = gpd.read_file(timezones['nsw'])
#                 vic = gpd.read_file(timezones['vic'])
#                 tas = gpd.read_file(timezones['tas'])

#                 # Merge the polygons to create single state/territory boundaries
#                 wa = gpd.GeoSeries(unary_union(wa.geometry))
#                 nt = gpd.GeoSeries(unary_union(nt.geometry))
#                 sa = gpd.GeoSeries(unary_union(sa.geometry))
#                 qld = gpd.GeoSeries(unary_union(qld.geometry))
#                 nsw = gpd.GeoSeries(unary_union(nsw.geometry))
#                 vic = gpd.GeoSeries(unary_union(vic.geometry))
#                 tas = gpd.GeoSeries(unary_union(tas.geometry))

#                 ## Note: day and night times will be calculated once per area-of-interest(ds)
#                 ## for the median latitude and longitude position of the aoi
#                 tidepost_lat_3577 = dem.x.median(dim='x').values
#                 tidepost_lon_3577 = dem.y.median(dim='y').values

#                 ## Set the Datacube native crs (GDA/Aus Albers (meters))
#                 crs_3577 = CRS.from_epsg(3577)

#                 ## Translate the crs of the tidepost to determine (1) local timezone
#                 ## and (2) the local sunrise and sunset times:
                
#                 ## (1) Create a transform to convert default epsg3577 coords to epsg4283 to compare 
#                 ## against state/territory boundary polygons and assign a timezone

#                 ## GDA94 CRS (degrees)
#                 crs_4283 = CRS.from_epsg(4283)
#                 ## Transfer coords from/to
#                 transformer_4283 = Transformer.from_crs(crs_3577, crs_4283) 
#                 ## Translate tidepost coords
#                 tidepost_lat_4283, tidepost_lon_4283 = transformer_4283.transform(tidepost_lat_3577,
#                                                                                   tidepost_lon_3577)
#                 ## Coordinate point to test for timezone   
#                 point_4283 = Point(tidepost_lon_4283, tidepost_lat_4283)

#                 ## (2) Create a transform to convert default epsg3577 coords to epsg4326 for use in 
#                 ## sunise/sunset library

#                 ## World WGS84 (degrees)
#                 crs_4326 = CRS.from_epsg(4326) 
#                 ## Transfer coords from/to
#                 transformer_4326 = Transformer.from_crs(crs_3577, crs_4326)
#                 ## Translate the tidepost coords
#                 tidepost_lat_4326, tidepost_lon_4326 = transformer_4326.transform(tidepost_lat_3577,
#                                                                                   tidepost_lon_3577)
#                 ## Coordinate point to locate the sunriset calculation
#                 point_4326 = Point(tidepost_lon_4326, tidepost_lat_4326)

#                 ## Set the local timezone for the analysis area of interest
#                 if wa.contains(point_4283)[0] == True:
#                     timezone = 'Australia/West'
#                     local_tz = 8

#                 elif nt.contains(point_4283)[0] == True:
#                     timezone = 'Australia/North'
#                     local_tz = 9.5

#                 elif sa.contains(point_4283)[0] == True:
#                     timezone = 'Australia/South'
#                     local_tz = 9.5

#                 elif qld.contains(point_4283)[0] == True:
#                     timezone = 'Australia/Queensland'
#                     local_tz = 10

#                 elif nsw.contains(point_4283)[0] == True:
#                     timezone = 'Australia/NSW'
#                     local_tz = 10

#                 elif vic.contains(point_4283)[0] == True:
#                     timezone = 'Australia/Victoria'
#                     local_tz = 10

#                 elif tas.contains(point_4283)[0] == True:
#                     timezone = 'Australia/Tasmania'
#                     local_tz = 10

#                 ## Calculate the local sunrise and sunset times
#                 # Place start and end dates in correct format
#                 start = time_range[0]
#                 end = time_range[-1]
#                 startdate = datetime.date(pd.to_datetime(start).year, 
#                                           pd.to_datetime(start).month, 
#                                           pd.to_datetime(start).day)

#                 # Make 'all_timerange' time-zone aware
#                 localtides = time_range.tz_localize(tz=pytz.UTC).tz_convert(timezone)

#                 # Replace the UTC datetimes from all_timerange with local times
#                 modelledtides = pd.DataFrame(index = localtides)

#                 # Return the difference in years for the time-period. 
#                 # Round up to ensure all modelledtide datetimes are captured in the solar model
#                 diff = pd.to_datetime(end) - pd.to_datetime(start)
#                 diff = int(ceil(diff.days/365))

#                 ## Model sunrise and sunset
#                 sun_df = sunriset.to_pandas(startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff)

#                 ## Set the index as a datetimeindex to match the modelledtide df
#                 sun_df = sun_df.set_index(pd.DatetimeIndex(sun_df.index))

#                 ## Append the date to each Sunrise and Sunset time
#                 sun_df['Sunrise dt'] = sun_df.index + sun_df['Sunrise']
#                 sun_df['Sunset dt'] = sun_df.index + (sun_df['Sunset'])

#                 ## Create new dataframes where daytime and nightime datetimes are recorded, then merged 
#                 ## on a new `Sunlight` column
#                 daytime=pd.DataFrame(data = 'Sunrise', index=sun_df['Sunrise dt'], columns=['Sunlight'])
#                 nighttime=pd.DataFrame(data = 'Sunset', index=sun_df['Sunset dt'], columns=['Sunlight'])
#                 DayNight = pd.concat([daytime, nighttime], join='outer')
#                 DayNight.sort_index(inplace=True)
#                 DayNight.index.rename('Datetime', inplace=True)

#                 ## Create an xarray object from the merged day/night dataframe
#                 day_night = xr.Dataset.from_dataframe(DayNight)

#                 ## Remove local timezone timestamp column in modelledtides dataframe. Xarray doesn't handle 
#                 ## timezone aware datetimeindexes 'from_dataframe' very well.
#                 modelledtides.index = modelledtides.index.tz_localize(tz=None)

#                 ## Create an xr Dataset from the modelledtides pd.dataframe
#                 mt = modelledtides.to_xarray()

#                 ## Filter the modelledtides (mt) by the daytime, nighttime datetimes from the sunriset module
#                 ## Modelled tides are designated as either day or night by propogation of the last valid index 
#                 ## value forward
#                 Solar=day_night.sel(Datetime=mt.index, method='ffill')

#                 ## Assign the day and night tideheight datasets
#                 SolarDayTides = mt.where(Solar.Sunlight=='Sunrise', drop=True)
#                 SolarNightTides = mt.where(Solar.Sunlight=='Sunset', drop=True)

#                 ## Extract DatetimeIndexes to use in exposure calculations
#                 all_timerange_day = pd.DatetimeIndex(SolarDayTides.index)
#                 all_timerange_night = pd.DatetimeIndex(SolarNightTides.index)

#                 if x == 'Daylight':
#                     timeranges['Daylight'] = all_timerange_day
#                 if x == 'Night':
#                     timeranges['Night'] = all_timerange_night

        elif x in sptl_filters:
            print(f'Calculating statial filter: {x}')
            
            timeranges, tide_cq_dict, exposure = spatial_filters(
                                                                modelled_freq,
                                                                x,
                                                                stacked_everything,
                                                                ModelledTides,
                                                                timeranges,
                                                                calculate_quantiles,
                                                                tide_cq_dict,
                                                                dem,
                                                                exposure
                                                                )
    
    ## Intersect the filters of interest to extract the common datetimes for calc of combined filters
    if filters_combined is not None:
        for x in filters_combined:
            y=x[0]
            z=x[1]
            timeranges[str(y+"_"+z)] = timeranges[y].intersection(timeranges[z])
    
    ## Generator expression to calculate exposure for each nominated filter in temp_filters
    # Don't calculate exposure for spatial filters. This has already been calculated.
    gen = (x for x in timeranges if x not in sptl_filters)
    
    for x in gen:
        # Run the pixel_tides function with the calculate_quantiles option.
        # For each pixel, an array of tideheights is returned, corresponding
        # to the percentiles from `calculate_quantiles` of the timerange-tide model that
        # each tideheight appears in the model.

        if (tide_model[0] == "ensemble") or (tide_model == "ensemble"):
            # Use ensemble model combining multiple input ocean tide models
            tide_cq, _ = pixel_tides_ensemble(
                dem,
                calculate_quantiles=calculate_quantiles,
                times=timeranges[str(x)],
                directory=tide_model_dir,
                ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
                top_n=3,
                reduce_method='mean',
                resolution=3000,
            )

        else:
            # Use single input ocean tide model
            tide_cq, _ = pixel_tides_ensemble(
                                    dem,
                                    resample=True,
                                    calculate_quantiles=calculate_quantiles,
                                    times=timeranges[str(x)],
                                    model=tide_model,
                                    directory=tide_model_dir,
            )

        # Add tide_cq to output dict
        tide_cq_dict[str(x)]=tide_cq

        # Calculate the tide-height difference between the elevation value and
        # each percentile value per pixel
        diff = abs(tide_cq - dem)

        # Take the percentile of the smallest tide-height difference as the
        # exposure % per pixel
        idxmin = diff.idxmin(dim="quantile")

        # Convert to percentage
        exposure[str(x)] = idxmin * 100


    return exposure, tide_cq_dict




