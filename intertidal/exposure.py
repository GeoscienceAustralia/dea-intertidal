import sunriset
import datetime
import re
import pytz
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

from math import ceil
from datetime import timedelta
from scipy.signal import argrelmax, argrelmin
from numpy import interp

from dea_tools.coastal import _pixel_tides_resample
from intertidal.tide_modelling import pixel_tides_ensemble
from intertidal.utils import configure_logging, round_date_strings


def temporal_filters(x, time_range, dem):
    """
    Identify and extract temporal-specific dates and times to feed into
    tidal modelling for custom exposure calculations.

    Parameters
    -------
    x : str
        A string identifier to nominate the temporal filter to
        calculate in this workflow. Must be one of: 'dry', 'wet',
        'summer', 'autumn', 'winter', 'spring', 'jan', 'feb',
        'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
        'nov', 'dec', 'daylight', 'night'.
    time_range : pd.DatetimeIndex
        A fixed frequency pd.DataTimeIndex matching the datetimes used
        to model tide heights
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data
        and coordinates and attributes metadata. Used to model sunrise
        and sunset times for the matching coordinates in dem.

    Returns
    -------
    filtered_time_range : pd.DataTimeIndex
        An updated pd.DataTimeIndex containing a filtered set of
        timesteps.
    """

    if x == "dry":
        return time_range.drop(
            time_range[
                (time_range.month == 10)  # Wet season: Oct-Mar
                | (time_range.month == 11)
                | (time_range.month == 12)
                | (time_range.month == 1)
                | (time_range.month == 2)
                | (time_range.month == 3)
            ]
        )
    elif x == "wet":
        return time_range.drop(
            time_range[
                (time_range.month == 4)  # Dry season: Apr-Sep
                | (time_range.month == 5)
                | (time_range.month == 6)
                | (time_range.month == 7)
                | (time_range.month == 8)
                | (time_range.month == 9)
            ]
        )
    elif x == "summer":
        return time_range.drop(time_range[time_range.quarter != 1])
    elif x == "autumn":
        return time_range.drop(time_range[time_range.quarter != 2])
    elif x == "winter":
        return time_range.drop(time_range[time_range.quarter != 3])
    elif x == "spring":
        return time_range.drop(time_range[time_range.quarter != 4])
    elif x == "jan":
        return time_range.drop(time_range[time_range.month != 1])
    elif x == "feb":
        return time_range.drop(time_range[time_range.month != 2])
    elif x == "mar":
        return time_range.drop(time_range[time_range.month != 3])
    elif x == "apr":
        return time_range.drop(time_range[time_range.month != 4])
    elif x == "may":
        return time_range.drop(time_range[time_range.month != 5])
    elif x == "jun":
        return time_range.drop(time_range[time_range.month != 6])
    elif x == "jul":
        return time_range.drop(time_range[time_range.month != 7])
    elif x == "aug":
        return time_range.drop(time_range[time_range.month != 8])
    elif x == "sep":
        return time_range.drop(time_range[time_range.month != 9])
    elif x == "oct":
        return time_range.drop(time_range[time_range.month != 10])
    elif x == "nov":
        return time_range.drop(time_range[time_range.month != 11])
    elif x == "dec":
        return time_range.drop(time_range[time_range.month != 12])
    elif x in ["daylight", "night"]:

        # Identify the central coordinate directly from the dem GeoBox
        tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs(
            "EPSG:4326"
        ).coords[0]

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

        if x == "daylight":
            return all_timerange_day
        if x == "night":
            return all_timerange_night


def spatial_filters(
    x,
    modelled_freq,
    modelledtides_1d,
    calculate_quantiles,
):
    """
    This function supports custom calculation of intertidal exposure by
    identifying and extracting datetimes and tide heights that are
    specific to unique spatial characteristics of the tidal frame from
    high temporal resolution tidal modelling.

    The function operates inside the exposure function and from a list
    of input filter types, runs this function for each filter that
    requires interrogation of the spatial range of modelled tide heights.

    Parameters
    -------
    x : str
        A string identifier to nominate the spatial filter to
        calculate in this workflow. Must be one of: 'spring_high',
        'spring_low', 'neap_high', 'neap_low', 'high_tide', 'low_tide'.
    modelled_freq : str
        A pandas time offset alias for the frequency with which to
        calculate the tide model during exposure calculations. Examples
        include '30min' for 30 minute cadence or '1h' for a one-hourly
        cadence. Defaults to '30min'.
    modelledtides_1d : xr.DataArray
        A 1-dimensional modelled tide timeseries providing the mean tide
        height per timestep for entire area of interest.
    calculate_quantiles : ndarray
        An array of equally spaced values representing the desired
        percentile values to map the distribution of tide heights
        onto.

    Returns
    -------
    modelledtides.tide_m : xarray.DataArray
        Tide height percentile distributions for each filter.
    filtered_time_range : pd.DataTimeIndex
        An pd.DataTimeIndex containing a filtered set of timesteps.
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
    if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

        # 1D tide modelling workflow
        # apply the peak detection routine
        if x in ["spring_high", "neap_high"]:
            modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]

        if x in ["spring_low", "neap_low"]:
            modelledtides_1d_peaks = argrelmin(modelledtides_1d.values, order=order)[0]

        if x == "neap_high":
            # Apply the peak detection routine to calculate all high tide
            # maxima
            tide_maxima = argrelmax(modelledtides_1d.values)[0]
            tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()

            # Extract neap high tides based on a half lunar cycle -
            # determined as the fraction of all high tide points
            # relative to the number of spring high tide values
            order_nh = int(
                ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
            )

            # Apply the peak detection routine to calculate all the neap
            # high tide minima within the high tide peaks
            neap_peaks = argrelmin(tide_maxima.tide_m.values, order=order_nh)[0]

        if x == "neap_low":
            # Apply the peak detection routine to calculate all low tide
            # maxima
            tide_maxima = argrelmin(modelledtides_1d.values)[0]
            tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()

            # extract neap low tides based on 14 day half lunar cycle -
            # determined as the fraction of all high tide points relative
            # to the number of spring high tide values
            order_nl = int(
                ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
            )

            # Apply the peak detection routine to calculate all the neap
            # low tide maxima within the low tide peaks
            neap_peaks = argrelmax(tide_maxima.tide_m.values, order=order_nl)[0]

        if x in ["neap_high", "neap_low"]:
            # Extract neap high tides
            neappeaks = tide_maxima.isel(time=neap_peaks)
            filtered_time_range = pd.to_datetime(neappeaks.time)

            # Extract the peak height dates
            modelledtides = neappeaks.quantile(q=calculate_quantiles, dim="time")

        if x in ["spring_high", "spring_low"]:
            # Select for indices associated with peaks
            springpeaks = modelledtides_1d.isel(
                time=modelledtides_1d_peaks
            ).to_dataset()

            # Save datetimes for calculation of combined filter exposure
            filtered_time_range = pd.to_datetime(springpeaks.time)

            # Extract the peak height dates
            modelledtides = springpeaks.quantile(q=calculate_quantiles, dim="time")

    if x == "high_tide":
        # Calculate all the high tide maxima
        high_peaks = argrelmax(modelledtides_1d.values)[0]

        # Extract all hightide peaks
        high_peaks2 = modelledtides_1d.isel(time=high_peaks)

        # Identify all lower hightide peaks
        lowhigh_peaks = argrelmin(high_peaks2.values)[0]

        # Extract all lower hightide peaks
        lowhigh_peaks2 = high_peaks2.isel(time=lowhigh_peaks)

        # Test for diurnal tidal regimes on the assumption that
        # semi-diurnal and mixed tidal settings should have approximately
        # equal proportions of daytime and nighttime hightide peaks
        if len(lowhigh_peaks) / len(high_peaks) < 0.2:
            filtered_time_range = pd.to_datetime(high_peaks2.time)
            modelledtides = high_peaks2.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()
        else:
            # Interpolate the lower hightide curve
            low_high_linear = interp(
                np.arange(0, len(modelledtides_1d)),
                high_peaks[lowhigh_peaks],
                lowhigh_peaks2.values,
            )

            # Extract all tides higher than/equal to the extrapolated
            # lowest high tide line
            hightide = modelledtides_1d.where(
                modelledtides_1d >= low_high_linear, drop=True
            )

            # Save datetimes for calculation of combined filter exposure
            filtered_time_range = pd.to_datetime(hightide.time)
            modelledtides = hightide.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()

    if x == "low_tide":
        # Calculate all the low tide maxima
        low_peaks = argrelmin(modelledtides_1d.values)[0]

        # Extract all lowtide peaks
        low_peaks2 = modelledtides_1d.isel(time=low_peaks)

        # Identify all higher lowtide peaks
        highlow_peaks = argrelmax(low_peaks2.values)[0]

        # Extract all higher lowtide peaks
        highlow_peaks2 = low_peaks2.isel(time=highlow_peaks)

        # Test for diurnal tidal regimes on the assumption that
        # semi-diurnal and mixed tidal settings should have
        # approximately equal proportions of daytime and nighttime
        # lowtide peaks
        if len(highlow_peaks) / len(low_peaks) < 0.2:
            filtered_time_range = pd.to_datetime(low_peaks2.time)
            modelledtides = low_peaks2.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()
        else:
            # Interpolate the higher lowtide curve
            high_low_linear = interp(
                np.arange(0, len(modelledtides_1d)),
                low_peaks[highlow_peaks],
                highlow_peaks2.values,
            )

            # Extract all tides lower than/equal to the extrapolated
            # higher lowtide line
            lowtide = modelledtides_1d.where(
                modelledtides_1d <= high_low_linear, drop=True
            )

            # Save datetimes for calculation of combined filter exposure
            filtered_time_range = pd.to_datetime(lowtide.time)
            modelledtides = lowtide.quantile(
                q=calculate_quantiles, dim="time"
            ).to_dataset()

    # Return modelledtides
    return modelledtides.tide_m, filtered_time_range


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
    that each pixel was 'exposed' from tidal inundation during the time
    period of interest.

    The exposure calculation is based on tide-height differences between
    the elevation value and modelled tide height percentiles.

    For an 'unfiltered', all of epoch-time, analysis, exposure is
    calculated per pixel. All other filter options calculate exposure
    from high temporal resolution modelled tides that are averaged 
    across the nominated area of interest only.

    This function firstly models high temporal resolution tides across
    the area of interest. Filtered datetimes and associated tide heights
    are then extracted from the modelled tides. Exposure is calculated
    by comparing the quantiled distribution curve of modelled tide
    heights from the filtered datetime dataset with DEM pixel elevations,
    returning an exposure percent.

    Parameters
    ----------
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data
        and coordinates and attributes metadata.
    start_date : str
        A string containing the start year of the desired analysis period
        as "YYYY". Note: analysis will start from "YYYY-01-01".
    end_date  :  str
        A string containing the end year of the desired analysis period
        as "YYYY". Note: analysis will end at "YYYY-12-31".
    modelled_freq : str
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
    filters : list of strings, optional
        An optional list of customisation options to input into the tidal
        modelling to calculate exposure. Filters include the following:
        - 'unfiltered': calculates exposure for the full input time period,
        - 'dry': Southern Hemisphere dry season, defined as April to
          September
        - 'wet': Southern Hemisphere wet season, defined as October to
          March
        - 'summer', 'autumn', 'winter', 'spring': exposure during
          specific seasons
        - 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
          'oct', 'nov', 'dec': expsure during specific months
        - 'daylight': all tide heights occurring between sunrise and
          sunset local time
        - 'night': all tide heights occurring between sunset and sunrise
          local time
        - 'spring_high': high tide exposure during the fortnightly
          spring tide cycle
        - 'spring_low': low tide exposure during the fortnightly spring
          tide cycle
        - 'neap_high': high tide exposure during the fortnightly neap
          tide cycle
        - 'neap_low': low tide exposure during the fortnightly neap tide
          cycle
        - 'high_tide': all tide heights greater than or equal to the
          local lowest high tide heights in high temporal resolution
          tidal modelling
        - 'low_tide': all tide heights lower than or equal to the local
          highest low tide heights in high temporal resolution tidal
          modelling
        Defaults to ['unfiltered'] if none supplied.
    filters_combined : list of two-object tuples, optional
        An optional list of paired customisation options from which to
        calculate exposure. Filters must be sourced from the list under
        'filters' and include one temporal and one spatial filter -
        defined in the `Notes` below. Example to calculate exposure
        during daylight hours (temporal) in the wet season (spatial) is
        [('wet', 'daylight')]. Multiple tuple pairs are supported.
        Defaults to None.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    exposure_ds : xarray.Dataset
        An xarray.Dataset containing a named exposure variable for each
        nominated filter, representing the percentage time exposure of
        each pixel from tidal inundation for the duration of the
        associated filtered time period between `start` and `end`.
    modelledtides_ds : dict
        An xarray.Dataset containing quantiled high temporal resolution
        tide modelling for each filter. Outputs will have dimensions of
        either ['quantile', 'x', 'y'] for "unfiltered", or ['quantile']
        for all other filters.

    Notes
    -----
    - The tide-height percentiles range from 0 to 100, divided into 101
    equally spaced values.
    - The 'diff' variable is calculated as the absolute difference
    between tide model percentile value and the DEM value at each pixel.
    - The 'idxmin' variable is the index of the smallest tide-height
    difference (i.e., maximum similarity) per pixel and is equivalent
    to the exposure percent.
    - temporal filters include any of: 'dry', 'wet', 'summer', 'autumn',
    'winter', 'spring', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
    'aug', 'sep', 'oct', 'nov', 'dec', 'daylight', 'night'
    - spatial filters include any of: 'spring_high', 'spring_low',
    'neap_high', 'neap_low', 'high_tide', 'low_tide'

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
        "dry",
        "wet",
        "summer",
        "autumn",
        "winter",
        "spring",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "daylight",
        "night",
    ]
    sptl_filters = [
        "spring_high",
        "spring_low",
        "neap_high",
        "neap_low",
        "high_tide",
        "low_tide",
    ]

    # Create empty xarray.Datasets to store outputs into
    exposure_ds = xr.Dataset(
        coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values))
    )
    modelledtides_ds = xr.Dataset(
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
            if str(x[0]) not in filters:
                filters.append(str(x[0]))
            if str(x[1]) not in filters:
                filters.append(str(x[1]))

    # Return error for incorrect filter-names
    all_filters = temp_filters + sptl_filters + ["unfiltered"]
    for x in filters:
        assert (
            x in all_filters
        ), f'Nominated filter "{x}" is not in {all_filters}. Check spelling and retry'

    # Run tide model at low resolution
    modelledtides_lowres = pixel_tides_ensemble(
        dem,
        model=tide_model,
        times=time_range,
        directory=tide_model_dir,
        ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
        resample=False,
    )

    # If custom filters are requested, calculate a 1D tide height time
    # series
    if (len(filters) >= 1) & (filters != ["unfiltered"]):

        # Calculate tide height time series. To reduce compute, average
        # across the y and x dimensions
        modelledtides_1d = modelledtides_lowres.mean(dim=["x", "y"])

    # Calculate quantiles and reproject low resolution tide data to
    # pixel resolution if any filter is "unfiltered"
    if "unfiltered" in filters:

        # Convert to quantiles
        modelledtides_lowres = modelledtides_lowres.quantile(
            q=calculate_quantiles, dim="time"
        ).astype(modelledtides_lowres.dtype)

        # Reproject into pixel resolution, after making sure CRS is present
        modelledtides_highres, _ = _pixel_tides_resample(
            tides_lowres=modelledtides_lowres.odc.assign_crs(dem.odc.geobox.crs),
            ds=dem,
        )

        # Add pixel resolution tides into to output dataset
        modelledtides_ds["unfiltered"] = modelledtides_highres

    # Filter the input timerange to include only dates or tide ranges of
    # interest if filters is not None:
    for x in filters:
        if x in temp_filters:
            print(f"Filtering timesteps for {x}")
            timeranges[x] = temporal_filters(x, time_range, dem)

        elif x in sptl_filters:
            print(f"Filtering timesteps for {x}")
            modelledtides_ds[x], timeranges[x] = spatial_filters(
                x=x,
                modelled_freq=modelled_freq,
                modelledtides_1d=modelledtides_1d,
                calculate_quantiles=calculate_quantiles,
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
        modelledtides_x = modelledtides_1d.sel(time=timeranges[str(x)])

        # Calculate quantile values on remaining tide heights
        modelledtides_x = (
            modelledtides_x.quantile(q=calculate_quantiles, dim="time")
            .to_dataset()
            .tide_m
        )

        # Add modelledtides_x to output dict
        modelledtides_ds[str(x)] = modelledtides_x

    # Calculate exposure per filter
    for x in modelledtides_ds:
        print(f"Calculating {x} exposure")

        # Calculate the tide-height difference between the elevation
        # value and each percentile value per pixel
        diff = abs(modelledtides_ds[str(x)] - dem)

        # Take the percentile of the smallest tide-height difference as
        # the exposure % per pixel
        idxmin = diff.idxmin(dim="quantile")

        # Convert to percentage and add as variable in exposure dataset
        exposure_ds[str(x)] = idxmin * 100

    return exposure_ds, modelledtides_ds
