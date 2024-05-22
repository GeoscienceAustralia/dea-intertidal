import sunriset
import datetime
import re
import pytz
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

from shapely.geometry import Point
from shapely.ops import unary_union
from math import ceil
from datetime import timedelta
from pyproj import CRS, Transformer
from scipy.signal import argrelmax, argrelmin
from numpy import interp

from dea_tools.coastal import pixel_tides, model_tides
from intertidal.tide_modelling import pixel_tides_ensemble
from intertidal.utils import configure_logging, round_date_strings


def temporal_filters(x, timeranges, time_range, dem):
    """
    Identify and extract temporal-specific dates and times to feed
    into tidal modelling for custom exposure calculations.
    """

    if x == "Dry":
        timeranges["Dry"] = time_range.drop(
            time_range[
                (time_range.month == 10)  # Wet season: Oct-Mar
                | (time_range.month == 11)
                | (time_range.month == 12)
                | (time_range.month == 1)
                | (time_range.month == 2)
                | (time_range.month == 3)
            ]
        )
    elif x == "Wet":
        timeranges["Wet"] = time_range.drop(
            time_range[
                (time_range.month == 4)  # Dry season: Apr-Sep
                | (time_range.month == 5)
                | (time_range.month == 6)
                | (time_range.month == 7)
                | (time_range.month == 8)
                | (time_range.month == 9)
            ]
        )
    elif x == "Summer":
        timeranges["Summer"] = time_range.drop(time_range[time_range.quarter != 1])
    elif x == "Autumn":
        timeranges["Autumn"] = time_range.drop(time_range[time_range.quarter != 2])
    elif x == "Winter":
        timeranges["Winter"] = time_range.drop(time_range[time_range.quarter != 3])
    elif x == "Spring":
        timeranges["Spring"] = time_range.drop(time_range[time_range.quarter != 4])
    elif x == "Jan":
        timeranges["Jan"] = time_range.drop(time_range[time_range.month != 1])
    elif x == "Feb":
        timeranges["Feb"] = time_range.drop(time_range[time_range.month != 2])
    elif x == "Mar":
        timeranges["Mar"] = time_range.drop(time_range[time_range.month != 3])
    elif x == "Apr":
        timeranges["Apr"] = time_range.drop(time_range[time_range.month != 4])
    elif x == "May":
        timeranges["May"] = time_range.drop(time_range[time_range.month != 5])
    elif x == "Jun":
        timeranges["Jun"] = time_range.drop(time_range[time_range.month != 6])
    elif x == "Jul":
        timeranges["Jul"] = time_range.drop(time_range[time_range.month != 7])
    elif x == "Aug":
        timeranges["Aug"] = time_range.drop(time_range[time_range.month != 8])
    elif x == "Sep":
        timeranges["Sep"] = time_range.drop(time_range[time_range.month != 9])
    elif x == "Oct":
        timeranges["Oct"] = time_range.drop(time_range[time_range.month != 10])
    elif x == "Nov":
        timeranges["Nov"] = time_range.drop(time_range[time_range.month != 11])
    elif x == "Dec":
        timeranges["Dec"] = time_range.drop(time_range[time_range.month != 12])
    elif x in ["Daylight", "Night"]:

        # Identify the central coordinate directly from the dem GeoBox
        tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs(
            "EPSG:4326"
        ).coords[0]

        # Coordinate point to locate the sunriset calculation
        point_4326 = Point(tidepost_lon_4326, tidepost_lat_4326)

        # Calculate the local sunrise and sunset times
        # Place start and end dates in correct format
        start = time_range[0]
        end = time_range[-1]
        startdate = datetime.date(
            pd.to_datetime(start).year,
            pd.to_datetime(start).month,
            pd.to_datetime(start).day,
        )

        # Make 'all_timerange' time-zone aware
        localtides = time_range.tz_localize(tz=pytz.UTC)

        # Replace the UTC datetimes from all_timerange with local times
        ModTides = pd.DataFrame(index=localtides)

        # Return the difference in years for the time-period.
        # Round up to ensure all modelledtide datetimes are captured in
        # the solar model
        diff = pd.to_datetime(end) - pd.to_datetime(start)
        diff = int(ceil(diff.days / 365))

        local_tz = 0

        # Model sunrise and sunset
        sun_df = sunriset.to_pandas(
            startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff
        )

        # Set the index as a datetimeindex to match the modelledtide df
        sun_df = sun_df.set_index(pd.DatetimeIndex(sun_df.index))

        # Append the date to each Sunrise and Sunset time
        sun_df["Sunrise dt"] = sun_df.index + sun_df["Sunrise"]
        sun_df["Sunset dt"] = sun_df.index + sun_df["Sunset"]

        # Create new dataframes where daytime and nightime datetimes are
        # recorded, then merged on a new `Sunlight` column
        daytime = pd.DataFrame(
            data="Sunrise", index=sun_df["Sunrise dt"], columns=["Sunlight"]
        )
        nighttime = pd.DataFrame(
            data="Sunset", index=sun_df["Sunset dt"], columns=["Sunlight"]
        )
        DayNight = pd.concat([daytime, nighttime], join="outer")
        DayNight.sort_index(inplace=True)
        DayNight.index.rename("Datetime", inplace=True)

        # Create an xarray object from the merged day/night dataframe
        day_night = xr.Dataset.from_dataframe(DayNight)

        # Remove local timezone timestamp column in modelledtides
        # dataframe. Xarray doesn't handle timezone aware datetimeindexes
        # 'from_dataframe' very well.
        ModTides.index = ModTides.index.tz_localize(tz=None)

        # Create an xr Dataset from the modelledtides pd.dataframe
        mt = ModTides.to_xarray()

        # Filter the modelledtides (mt) by the daytime, nighttime
        # datetimes from the sunriset module.
        # Modelled tides are designated as either day or night by
        # propogation of the last valid index value forward
        Solar = day_night.sel(Datetime=mt.index, method="ffill")

        # Assign the day and night tideheight datasets
        SolarDayTides = mt.where(Solar.Sunlight == "Sunrise", drop=True)
        SolarNightTides = mt.where(Solar.Sunlight == "Sunset", drop=True)

        # Extract DatetimeIndexes to use in exposure calculations
        all_timerange_day = pd.DatetimeIndex(SolarDayTides.index)
        all_timerange_night = pd.DatetimeIndex(SolarNightTides.index)

        if x == "Daylight":
            timeranges["Daylight"] = all_timerange_day
        if x == "Night":
            timeranges["Night"] = all_timerange_night

    return timeranges


def spatial_filters(
    modelled_freq,
    x,
    modelledtides_flat,
    ModelledTides,
    timeranges,
    calculate_quantiles,
    modelledtides_dict,
    dem,
):
    """
    Identify and extract spatial-specific dates and times to feed
    into tidal modelling for custom exposure calculations.
    """

    # Extract the modelling freq units
    # Split the number and text characters in modelled_freq
    freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
    freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])

    # Extract the number of modelled timesteps per 14 days (half lunar
    # cycle) for neap/spring calcs
    mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)

    # Identify kwargs for peak detection algorithm
    order = int(mod_timesteps / 2)

    # Calculate the spring highest and spring lowest tides per 14 day
    # half lunar cycle
    if x in ["Spring_high", "Spring_low", "Neap_high", "Neap_low"]:

        # 1D tide modelling workflow
        # apply the peak detection routine
        if x in ["Spring_high", "Neap_high"]:
            modelledtides_flat_peaks = argrelmax(
                modelledtides_flat.values, order=order
            )[0]

        if x in ["Spring_low", "Neap_low"]:
            modelledtides_flat_peaks = argrelmin(
                modelledtides_flat.values, order=order
            )[0]

        if x == "Neap_high":
            # Apply the peak detection routine to calculate all high tide
            # maxima
            tide_maxima = argrelmax(modelledtides_flat.values)[0]
            tide_maxima = modelledtides_flat.isel(time=tide_maxima).to_dataset()

            # Extract neap high tides based on a half lunar cycle -
            # determined as the fraction of all high tide points
            # relative to the number of spring high tide values
            order_nh = int(
                ceil((len(tide_maxima.time) / (len(modelledtides_flat_peaks)) / 2))
            )

            # Apply the peak detection routine to calculate all the neap
            # high tide minima within the high tide peaks
            neap_peaks = argrelmin(tide_maxima.tide_m.values, order=order_nh)[0]

        if x == "Neap_low":
            # Apply the peak detection routine to calculate all low tide
            # maxima
            tide_maxima = argrelmin(modelledtides_flat.values)[0]
            tide_maxima = modelledtides_flat.isel(time=tide_maxima).to_dataset()

            # extract neap low tides based on 14 day half lunar cycle -
            # determined as the fraction of all high tide points relative
            # to the number of spring high tide values
            order_nl = int(
                ceil((len(tide_maxima.time) / (len(modelledtides_flat_peaks)) / 2))
            )

            # Apply the peak detection routine to calculate all the neap
            # low tide maxima within the low tide peaks
            neap_peaks = argrelmax(tide_maxima.tide_m.values, order=order_nl)[0]

        if x in ["Neap_high", "Neap_low"]:
            # Extract neap high tides
            neappeaks = tide_maxima.isel(time=neap_peaks)
            timeranges[str(x)] = pd.to_datetime(neappeaks.time)

            # Extract the peak height dates
            modelledtides = neappeaks.quantile(q=calculate_quantiles, dim="time")

        if x in ["Spring_high", "Spring_low"]:
            # Select for indices associated with peaks
            springpeaks = modelledtides_flat.isel(
                time=modelledtides_flat_peaks
            ).to_dataset()

            # Save datetimes for calculation of combined filter exposure
            timeranges[str(x)] = pd.to_datetime(springpeaks.time)

            # Extract the peak height dates
            modelledtides = springpeaks.quantile(q=calculate_quantiles, dim="time")

    if x == "Hightide":
        # Calculate all the high tide maxima
        high_peaks = argrelmax(modelledtides_flat.values)[0]

        # Extract all hightide peaks
        high_peaks2 = modelledtides_flat.isel(time=high_peaks)

        # Identify all lower hightide peaks
        lowhigh_peaks = argrelmin(high_peaks2.values)[0]

        # Extract all lower hightide peaks
        lowhigh_peaks2 = high_peaks2.isel(time=lowhigh_peaks)

        # Test for diurnal tidal regimes on the assumption that
        # semi-diurnal and mixed tidal settings should have approximately
        # equal proportions of daytime and nighttime hightide peaks
        if len(lowhigh_peaks) / len(high_peaks) < 0.2:
            timeranges[str(x)] = pd.to_datetime(high_peaks2.time)
            modelledtides = high_peaks2.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()
        else:
            # Interpolate the lower hightide curve
            low_high_linear = interp(
                np.arange(0, len(modelledtides_flat)),
                high_peaks[lowhigh_peaks],
                lowhigh_peaks2.values,
            )

            # Extract all tides higher than/equal to the extrapolated
            # lowest high tide line
            hightide = modelledtides_flat.where(
                modelledtides_flat >= low_high_linear, drop=True
            )

            # Save datetimes for calculation of combined filter exposure
            timeranges[str(x)] = pd.to_datetime(hightide.time)
            modelledtides = hightide.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()

    if x == "Lowtide":
        # Calculate all the low tide maxima
        low_peaks = argrelmin(modelledtides_flat.values)[0]

        # Extract all lowtide peaks
        low_peaks2 = modelledtides_flat.isel(time=low_peaks)

        # Identify all higher lowtide peaks
        highlow_peaks = argrelmax(low_peaks2.values)[0]

        # Extract all higher lowtide peaks
        highlow_peaks2 = low_peaks2.isel(time=highlow_peaks)

        # Test for diurnal tidal regimes on the assumption that
        # semi-diurnal and mixed tidal settings should have
        # approximately equal proportions of daytime and nighttime
        # lowtide peaks
        if len(highlow_peaks) / len(low_peaks) < 0.2:
            timeranges[str(x)] = pd.to_datetime(low_peaks2.time)
            modelledtides = low_peaks2.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()
        else:
            # Interpolate the higher lowtide curve
            high_low_linear = interp(
                np.arange(0, len(modelledtides_flat)),
                low_peaks[highlow_peaks],
                highlow_peaks2.values,
            )

            # Extract all tides lower than/equal to the extrapolated
            # higher lowtide line
            lowtide = modelledtides_flat.where(
                modelledtides_flat <= high_low_linear, drop=True
            )

            # Save datetimes for calculation of combined filter exposure
            timeranges[str(x)] = pd.to_datetime(lowtide.time)
            modelledtides = lowtide.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()

    # Add modelledtides to output dict
    modelledtides_dict[str(x)] = modelledtides.tide_m

    return timeranges, modelledtides_dict  # , exposure


def exposure(
    dem,
    start_date,
    end_date,
    modelled_freq="30min",
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
    filters=None,
    filters_combined=None,
    run_id=None,
    log=None,
):
    """
    Calculate intertidal exposure, indicating the proportion of time
    that each pixel was "exposed" from tidal inundation during the time
    period of interest.

    The exposure calculation is based on tide-height differences between
    the elevation value and modelled tide height percentiles.

    For an 'unfiltered', all of epoch-time, analysis, exposure is
    calculated per pixel. All other filter options calculate exposure 
    from a high temporal resolution tide model that is generated for the
    center of the nominated area of interest only.

    This function firstly calculates a high temporal resolution tidal
    model for area (or pixels) of interest. Filtered datetimes and
    associated tide heights are then isolated from the tidal model.
    Exposure is calculated by comparing the quantiled distribution curve
    of modelled tide heights from the filtered datetime dataset with dem
    pixel elevations to identify exposure %.

    Parameters
    ----------
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data
        and coordinates and attributes metadata.
    start_date  : str
        A string containing the start year of the desired analysis period
        as "YYYY". Note: analysis will start from "YYYY-01-01".
    end_date  :  str
        A string containing the end year of the desired analysis period
        as "YYYY". Note: analysis will end at "YYYY-12-31".
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
        An optional list of customisation options to input into the tidal
        modelling to calculate exposure. Filters include the following:
        - 'unfiltered' calculates exposure for the full input time period,
        - 'Dry' season, defined as April to September,
        - 'Wet' season, defined as October to March,
        - 'Summer',
        - 'Autumn',
        - 'Winter',
        - 'Spring',
        - 'Jan',
        - 'Feb',
        - 'Mar',
        - 'Apr',
        - 'May',
        - 'Jun',
        - 'Jul',
        - 'Aug',
        - 'Sep',
        - 'Oct',
        - 'Nov',
        - 'Dec',
        - 'Daylight', all tide heights occurring between sunrise and
          sunset in daily UTC time,
        - 'Night', all tide heights occurring between sunset and sunrise
          in daily UTC time,
        - 'Spring_high', high tide exposure during the fortnightly
          spring tide cycle,
        - 'Spring_low', low tide exposure during the fortnightly spring
          tide cycle,
        - 'Neap_high', high tide exposure during the fortnightly neap 
          tide cycle,
        - 'Neap_low', low tide exposure during the fortnightly neap tide
          cycle,
        - 'Hightide', all tide heights greater than or equal to the
          local lowest high tide heights in high temporal resolution 
          tidal modelling,
        - 'Lowtide' all tide heights lower than or equal to the local
          highest low tide heights in high temporal resolution tidal
          modelling,
        Defaults to ['unfiltered'] if none supplied.
    filters_combined  :  list of two-object tuples, optional
        An optional list of paired customisation options from which to
        calculate exposure. Filters must be sourced from the list under 
        'filters' and include one temporal and one spatial filter - 
        defined in the `Notes` below. Example to calculate exposure
        during daylight hours (temporal) in the wet season (spatial) is
        [('Wet', 'Daylight')]. Multiple tuple pairs are supported.
        Defaults to None.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    exposure : dict
        A dictionary of xarray.Datasets containing a named exposure dataset for each
        nominated filter, representing the percentage time exposurs of each pixel from seawater
        for the duration of the associated filtered time period between `start` and `end`.
    modelledtides : dict
        A dictionary of xarray.Datasets containing a named dataset of the quantiled high temporal
        resolution tide modelling for each filter. Dimesions should be
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
    - temporal filters include any of: 'Dry', 'Wet', 'Summer', 'Autumn',
    'Winter', 'Spring', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Daylight', 'Night'
    - spatial filters include any of: 'Spring_high', 'Spring_low',
    'Neap_high', 'Neap_low', 'Hightide', 'Lowtide'

    """
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id

    # Create the tide-height percentiles from which to calculate
    # exposure statistics
    calculate_quantiles = np.linspace(0, 1, 101)

    # Generate range of times covering entire period of satellite record
    # for exposure and bias/offset calculation
    time_range = pd.date_range(
        start=round_date_strings(start_date, round_type="start"),
        end=round_date_strings(end_date, round_type="end"),
        freq=modelled_freq,
    )

    # Separate 'filters' into spatial and temporal categories to define
    # which exposure workflow to use
    temp_filters = [
        "Dry",
        "Wet",
        "Summer",
        "Autumn",
        "Winter",
        "Spring",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Daylight",
        "Night",
    ]
    sptl_filters = [
        "Spring_high",
        "Spring_low",
        "Neap_high",
        "Neap_low",
        "Hightide",
        "Lowtide",
    ]

    # Create empty datasets to store outputs into
    exposure = xr.Dataset(coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values)))
    modelledtides_dict = xr.Dataset(
        coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values))
    )

    # Create an empty dict to store temporal `time_range` variables into
    timeranges = {}

    # Set filters variable if none supplied
    if filters is None:
        filters = ["unfiltered"]

    # If filter combinations are desired, make sure each filter is
    # calculated individually for later combination
    if filters_combined is not None:
        for x in filters_combined:
            filters.append(str(x[0])) if str(x[0]) not in filters else next
            filters.append(str(x[1])) if x[1] not in filters else next

    # Return error for incorrect filter-names
    all_filters = temp_filters + sptl_filters + ["unfiltered"]
    for x in filters:
        assert (
            x in all_filters
        ), f'Nominated filter "{x}" is not in {all_filters}. Check spelling and retry'

    # Calculate a tidal model. Run at pixel resolution if any filter is
    # 'unfiltered' else run at low res
    if "unfiltered" in filters:
        mod_tides, _ = pixel_tides_ensemble(
            dem,
            model=tide_model,
            calculate_quantiles=calculate_quantiles,
            times=time_range,
            directory=tide_model_dir,
            ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
        )
        # Add modelledtides to output dict
        modelledtides_dict["unfiltered"] = mod_tides

    # For all other filter types, calculate a low spatial res tidal model
    if (len(filters) >= 1) & (filters != ["unfiltered"]):
        modelledtides = pixel_tides_ensemble(
            dem,
            model=tide_model,
            times=time_range,
            directory=tide_model_dir,
            ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
            resample=False,
        )

        # Flatten low res tidal model. To reduce compute, average across
        # the y and x dimensions
        modelledtides_flat = modelledtides.mean(dim=["x", "y"])

    # Filter the input timerange to include only dates or tide ranges of
    # interest if filters is not None:
    for x in filters:
        if x in temp_filters:
            print(f"-----\nCalculating {x} timerange")

            timeranges = temporal_filters(x, timeranges, time_range, dem)

        elif x in sptl_filters:
            print(f"-----\nCalculating {x} timerange")

            timeranges, modelledtides_dict = spatial_filters(
                modelled_freq,
                x,
                modelledtides_flat,
                modelledtides,
                timeranges,
                calculate_quantiles,
                modelledtides_dict,
                dem,
            )

    # Intersect the filters of interest to extract the common datetimes for
    # calculation of combined filters
    if filters_combined is not None:
        for x in filters_combined:
            y = x[0]
            z = x[1]
            timeranges[str(y + "_" + z)] = timeranges[y].intersection(timeranges[z])

    # Intersect datetimes of interest with the low-res tidal model
    # Don't calculate exposure for spatial filters. This has already
    # been calculated.
    gen = (x for x in timeranges if x not in sptl_filters)
    for x in gen:
        # Extract filtered datetimes from the full tidal model
        modelledtides_x = modelledtides_flat.sel(time=timeranges[str(x)])

        # Calculate quantile values on remaining tide heights
        modelledtides_x = (
            modelledtides_x.quantile(q=calculate_quantiles, dim="time")
            .to_dataset()
            .tide_m
        )

        # Add modelledtides to output dict
        modelledtides_dict[str(x)] = modelledtides_x

    # Calculate exposure per filter
    for x in modelledtides_dict:
        print(f"-----\nCalculating {x} exposure")

        # Calculate the tide-height difference between the elevation
        # value and each percentile value per pixel
        diff = abs(modelledtides_dict[str(x)] - dem)

        # Take the percentile of the smallest tide-height difference as
        # the exposure % per pixel
        idxmin = diff.idxmin(dim="quantile")
        
        # Convert to percentage
        exposure[str(x)] = idxmin * 100

    return exposure, modelledtides_dict
