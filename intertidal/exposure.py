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


def exposure(
            dem,
  			times,
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
    temporal_filters = ['dry', 'wet', 'summer', 'autumn', 'winter', 'spring', 'Jan', 'Feb', 'Mar', 'Apr', 
                        'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Daylight', 'Night']
    spatial_filters = ['Spring_high', 'Spring_low', 'Neap_high', 'Neap_low', 'Hightide', 'Lowtide']
      
    ## Set the required range of tide-height percentiles for exposure calculation
    # calculate_quantiles = np.linspace(0, 1, 101)
    calculate_quantiles = np.linspace(0, 1, 1001)#Temporary to separate exposure values
        
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
                
    if any (x in spatial_filters for x in filters):
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
        stacked_everything = ModelledTides.stack(z=['y','x']).groupby('z')
    
    # Filter the input timerange to include only dates or tide ranges of interest
    # if filters is not None:
    for x in filters:
        if x in temporal_filters:
            print(f'Calculating temporal filter: {x}')

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
            elif x == 'Daylight' or 'Night': 

                timezones = {'wa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/western_australia/cstwacd_r.shp',
                             'nt':'../../gdata1/data/boundaries/GEODATA_COAST_100K/northern_territory/cstntcd_r.shp',
                             'sa':'../../gdata1/data/boundaries/GEODATA_COAST_100K/south_australia/cstsacd_r.shp',
                             'qld':'../../gdata1/data/boundaries/GEODATA_COAST_100K/queensland/cstqldmd_r.shp',
                             'nsw':'../../gdata1/data/boundaries/GEODATA_COAST_100K/new_south_wales/cstnswcd_r.shp',
                             'vic':'../../gdata1/data/boundaries/GEODATA_COAST_100K/victoria/cstviccd_r.shp',
                             'tas':'../../gdata1/data/boundaries/GEODATA_COAST_100K/tasmania/csttascd_r.shp'
                             }

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

                ## Set the Datacube native crs (GDA/Aus Albers (meters))
                crs_3577 = CRS.from_epsg(3577)

                ## Translate the crs of the tidepost to determine (1) local timezone
                ## and (2) the local sunrise and sunset times:
                
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
                start = time_range[0]
                end = time_range[-1]
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

                if x == 'Daylight':
                    timeranges['Daylight'] = all_timerange_day
                if x == 'Night':
                    timeranges['Night'] = all_timerange_night

        elif x in spatial_filters:
            print(f'Calculating statial filter: {x}')

            # # Extract the modelling freq units
            # Split the number and text characters in modelled_freq
            freq_time = int(re.findall(r'(\d+)(\w+)', modelled_freq)[0][0])
            freq_unit = str(re.findall(r'(\d+)(\w+)', modelled_freq)[0][-1])

            # Extract the number of modelled timesteps per 14 days (half lunar cycle) for neap/spring calcs
            mod_timesteps = pd.Timedelta(14,"d")/pd.Timedelta(freq_time, freq_unit)
            
            ## Identify kwargs for peak detection algorithm
            order=(int(mod_timesteps/2))

            ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
            if x == 'Spring_high':

                print ('Calculating Spring_high')

                ## apply the peak detection routine
                stacked_everything_high = stacked_everything.apply(
                    lambda x: xr.DataArray(argrelmax(x.values, order=order)[0])
                    )
                ## Unstack
                springhighs_all = stacked_everything_high.unstack('z')
                ##Reorder the y axis. Uncertain why it gets reversed during the stack/unstack.
                springhighs_all = springhighs_all.reindex(y=springhighs_all.y[::-1])
                ## Rename the time axis
                springhighs_all = springhighs_all.rename({'dim_0':'time'})
                ## Convert to dataset
                springhighs_all = springhighs_all.to_dataset(name = 'time')
                ## Reorder the dims
                springhighs_all = springhighs_all[['time','y','x']]

                # Select dates associated with detected peaks
                ## removed reference below to ModelledTides[0]. Possibly an artefact of new 
                ## pixel_tides_ensemble func. If using pixel_tides, may need to revert to ModelledTides[0].
                
                # springhighs_all = ModelledTides[0].to_dataset().isel(time=springhighs_all.time)
                springhighs_all = ModelledTides.to_dataset().isel(time=springhighs_all.time) 
                
                ## Save datetimes for calculation of combined filter exposure
                timeranges['Spring_high'] = pd.to_datetime(springhighs_all.isel(x=1,y=1).time)

                ## Extract the peak height dates
                tide_cq = springhighs_all.tide_m.quantile(q=calculate_quantiles,dim='time')
                
                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq

                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Spring_high'] = idxmin * 100


                ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
            if x == 'Spring_low':
                print ('Calculating Spring_low')

                ## apply the peak detection routine
                stacked_everything_low = stacked_everything.apply(lambda x: xr.DataArray(argrelmin(x.values, order=order)[0]))
                ## Unstack
                springlows_all = stacked_everything_low.unstack('z')
                ##Reorder the y axis. Uncertain why it gets reversed during the stack/unstack.
                springlows_all = springlows_all.reindex(y=springlows_all.y[::-1])
                ## Rename the time axis
                springlows_all = springlows_all.rename({'dim_0':'time'})
                ## Convert to dataset
                springlows_all = springlows_all.to_dataset(name = 'time')
                ## Reorder the dims
                springlows_all = springlows_all[['time','y','x']]
                ## Select dates associated with detected peaks
                # springlows_all = ModelledTides[0].to_dataset().isel(time=springlows_all.time)
                springlows_all = ModelledTides.to_dataset().isel(time=springlows_all.time)## removed reference to ModelledTides[0]. Possibly an artefact of new pixel_tides_ensemble func. If using pixel_tides, may need to revert to ModelledTides[0].

                ## Save datetimes for calculation of combined filter exposure
                timeranges['Spring_low'] = pd.to_datetime(springlows_all.isel(x=1,y=1).time)
                
                tide_cq = springlows_all.tide_m.quantile(q=calculate_quantiles,dim='time')
                
                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq

                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Spring_low'] = idxmin * 100

            if x == 'Neap_high':
                print ('Calculating Neap_high')
                ## Calculate the number of spring high tides to support calculation of neap highs
                ## apply the peak detection routine
                stacked_everything_high = stacked_everything.apply(lambda x: xr.DataArray(argrelmax(x.values, order=order)[0]))
                ## Unstack
                springhighs_all = stacked_everything_high.unstack('z')

                ## apply the peak detection routine to calculate all the high tide maxima
                Max_testarray = stacked_everything.apply(lambda x: xr.DataArray(argrelmax(x.values)[0]))
                ## extract the corresponding dates from the peaks
                Max_testarray = (Max_testarray.unstack('z'))
                Max_testarray = (Max_testarray.reindex(y=Max_testarray.y[::-1])
                                 .rename({'dim_0':'time'})
                                 .to_dataset(name = 'time')
                                 [['time','y','x']]
                                )
                ## extract all hightide peaks
                # Max_testarray = ModelledTides[0].to_dataset().isel(time=Max_testarray.time)
                Max_testarray = ModelledTides.to_dataset().isel(time=Max_testarray.time)## removed reference to ModelledTides[0]. Possibly an artefact of new pixel_tides_ensemble func. If using pixel_tides, may need to revert to ModelledTides[0].

                ## repeat the peak detection to identify neap high tides (minima in the high tide maxima)
                stacked_everything2 = Max_testarray.tide_m.stack(z=['y','x']).groupby('z')
                ## extract neap high tides based on 14 day half lunar cycle - determined as the fraction of all high tide points
                ## relative to the number of spring high tide values
                order_nh = int(ceil((len(Max_testarray.time)/(len(springhighs_all))/2)))
                ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
                neaphighs_all = stacked_everything2.apply(lambda x: xr.DataArray(argrelmin(x.values, order=order_nh)[0]))
                ## unstack and format as above                                    
                neaphighs_all = neaphighs_all.unstack('z')
                neaphighs_all = (
                                neaphighs_all
                                 .reindex(y=neaphighs_all.y[::-1])
                                 .rename({'dim_0':'time'})
                                 .to_dataset(name = 'time')
                                 [['time','y','x']]
                                )
                ## extract neap high tides
                neaphighs_all = Max_testarray.isel(time=neaphighs_all.time)

                ## Save datetimes for calculation of combined filter exposure
                timeranges['Neap_high'] = pd.to_datetime(neaphighs_all.isel(x=1,y=1).time)
                
                tide_cq = neaphighs_all.tide_m.quantile(q=calculate_quantiles,dim='time')

                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq
                
                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Neap_high'] = idxmin * 100

            if x == 'Neap_low':
                print ('Calculating Neap_low')
                ## Calculate the number of spring low tides to support calculation of neap low tides
                ## apply the peak detection routine
                stacked_everything_low = stacked_everything.apply(lambda x: xr.DataArray(argrelmin(x.values, order=order)[0]))
                
                ## Unstack
                springlows_all = stacked_everything_low.unstack('z')                    

                ## apply the peak detection routine to calculate all the low tide maxima
                Min_testarray = stacked_everything.apply(lambda x: xr.DataArray(argrelmin(x.values)[0]))
                
                ## extract the corresponding dates from the peaks
                Min_testarray = (Min_testarray.unstack('z'))
                Min_testarray = (Min_testarray.reindex(y=Min_testarray.y[::-1])
                                 .rename({'dim_0':'time'})
                                 .to_dataset(name = 'time')
                                 [['time','y','x']]
                                )
                
                ## extract all lowtide peaks
                # Min_testarray = ModelledTides[0].to_dataset().isel(time=Min_testarray.time)
                Max_testarray = ModelledTides.to_dataset().isel(time=Max_testarray.time)## removed reference to ModelledTides[0]. Possibly an artefact of new pixel_tides_ensemble func. If using pixel_tides, may need to revert to ModelledTides[0].

                ## repeat the peak detection to identify neap low tides (maxima in the low tide maxima)
                stacked_everything2 = Min_testarray.tide_m.stack(z=['y','x']).groupby('z')
                
                ## extract neap high tides based on 14 day half lunar cycle - determined as the fraction of all high tide points
                ## relative to the number of spring high tide values
                order_nl = int(ceil((len(Min_testarray.time)/(len(springlows_all))/2)))
                
                ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
                neaplows_all = stacked_everything2.apply(lambda x: xr.DataArray(argrelmax(x.values, order=order_nl)[0]))
                
                ## unstack and format as above                                    
                neaplows_all = neaplows_all.unstack('z')
                neaplows_all = (
                                neaplows_all
                                 .reindex(y=neaplows_all.y[::-1])
                                 .rename({'dim_0':'time'})
                                 .to_dataset(name = 'time')
                                 [['time','y','x']]
                                )
                
                ## extract neap high tides
                neaplows_all = Min_testarray.isel(time=neaplows_all.time)

                ## Save datetimes for calculation of combined filter exposure
                timeranges['Neap_low'] = pd.to_datetime(neaplows_all.isel(x=1,y=1).time)
                
                tide_cq = neaplows_all.tide_m.quantile(q=calculate_quantiles,dim='time')

                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq
                
                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Neap_low'] = idxmin * 100


            if x == 'Hightide':
                print ('Calculating Hightide')
                def lowesthightides(x):
                    '''
                    x is a grouping of x and y pixels from the peaks_array (labelled as 'z')
                    '''

                    ## apply the peak detection routine to calculate all the high tide maxima
                    high_peaks = np.array(argrelmax(x.values)[0])

                    ## extract all hightide peaks
                    Max_testarray = x.isel(time=high_peaks)

                    ## Identify all lower hightide peaks
                    lowhigh_peaks = np.array(argrelmin(Max_testarray.values)[0])

                    ## Interpolate the lower hightide curve
                    neap_high_linear = interp(
                                                ## Create an array to interpolate into
                                                np.arange(0,len(x.time)),
                                                ## low high peaks as a subset of all high tide peaks
                                                high_peaks[lowhigh_peaks],
                                                ## Corresponding tide heights
                                                Max_testarray.isel(time=lowhigh_peaks).squeeze(['z']).values,
                                                )

                    # # Extract hightides as all tides higher than/equal to the extrapolated lowest high tide line
                    hightide = x.squeeze(['z']).where(x.squeeze(['z']) >= neap_high_linear, drop=True)

                    return hightide

                ## Vectorise the hightide calculation
                lowhighs_all = stacked_everything.apply(lambda x: xr.DataArray(lowesthightides(x)))

                # ## Unstack and re-format the array
                lowhighs_all = lowhighs_all.unstack('z')
                lowhighs_all_unstacked = (
                                    lowhighs_all
                                     .reindex(y=lowhighs_all.y[::-1])
                                     .to_dataset()
                                     [['tide_m','time','y','x']]
                                    )

                ## Save datetimes for calculation of combined filter exposure
                timeranges['Hightide'] = pd.to_datetime(lowhighs_all_unstacked.isel(x=1,y=1).time)
                
                tide_cq = lowhighs_all_unstacked.tide_m.quantile(q=calculate_quantiles,dim='time')

                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq
                
                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Hightide'] = idxmin * 100

            if x == 'Lowtide':
                print ('Calculating Lowtide')
                def highestlowtides(x):
                    '''
                    x is a grouping of x and y pixels from the peaks_array (labelled as 'z')
                    '''

                    ## apply the peak detection routine to calculate all the high tide maxima
                    low_peaks = np.array(argrelmin(x.values)[0])

                    ## extract all hightide peaks
                    Min_testarray = x.isel(time=low_peaks)

                    ## Identify all lower hightide peaks
                    highlow_peaks = np.array(argrelmax(Min_testarray.values)[0])

                    ## Interpolate the lower hightide curve
                    neap_low_linear = interp(
                                            ## Create an array to interpolate into
                                            np.arange(0,len(x.time)),
                                            ## low high peaks as a subset of all high tide peaks
                                            low_peaks[highlow_peaks],
                                            ## Corresponding tide heights
                                            Min_testarray.isel(time=highlow_peaks).squeeze(['z']).values,
                                            )

                    # # Extract hightides as all tides higher than/equal to the extrapolated lowest high tide line
                    lowtide = x.squeeze(['z']).where(x.squeeze(['z']) <= neap_low_linear, drop=True)

                    return lowtide 

                ## Vectorise the lowtide calculation
                highlows_all = stacked_everything.apply(lambda x: xr.DataArray(highestlowtides(x)))

                # ## Unstack and re-format the array
                highlows_all = highlows_all.unstack('z')
                highlows_all_unstacked = (
                                    highlows_all
                                     .reindex(y=highlows_all.y[::-1])
                                     .to_dataset()
                                     [['tide_m','time','y','x']]
                                    )

                ## Save datetimes for calculation of combined filter exposure
                timeranges['Lowtide'] = pd.to_datetime(highlows_all.isel(x=1,y=1).time)
                
                tide_cq = highlows_all_unstacked.tide_m.quantile(q=calculate_quantiles,dim='time')
                
                # Add tide_cq to output dict
                tide_cq_dict[str(x)]=tide_cq

                # Calculate the tide-height difference between the elevation value and
                # each percentile value per pixel
                diff = abs(tide_cq - dem)

                # Take the percentile of the smallest tide-height difference as the
                # exposure % per pixel
                idxmin = diff.idxmin(dim="quantile")

                # Convert to percentage
                exposure['Lowtide'] = idxmin * 100
    
    ## Intersect the filters of interest to extract the common datetimes for calc of combined filters
    if filters_combined is not None:
        for x in filters_combined:
            y=x[0]
            z=x[1]
            timeranges[str(y+"_"+z)] = timeranges[y].intersection(timeranges[z])
    
    ## Generator expression to calculate exposure for each nominated filter in temporal_filters
    # Don't calculate exposure for spatial filters. This has already been calculated.
    gen = (x for x in timeranges if x not in spatial_filters)
    
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




