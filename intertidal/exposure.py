import sunriset
import datetime
import re
import pytz

import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

from math import ceil
from scipy.signal import argrelmax, argrelmin
from numpy import interp
from eo_tides.eo import _pixel_tides_resample, pixel_tides
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
        return time_range.drop(
            time_range[
                (time_range.month == 3)
                | (time_range.month == 4)
                | (time_range.month == 5)
                | (time_range.month == 6)
                | (time_range.month == 7)
                | (time_range.month == 8)
                | (time_range.month == 9)
                | (time_range.month == 10)
                | (time_range.month == 11)
            ]
        )
    elif x == "autumn":
        return time_range.drop(
            time_range[
                (time_range.month == 1)
                | (time_range.month == 2)
                | (time_range.month == 6)
                | (time_range.month == 7)
                | (time_range.month == 8)
                | (time_range.month == 9)
                | (time_range.month == 10)
                | (time_range.month == 11)
                | (time_range.month == 12)
            ]
        )
    elif x == "winter":
        return time_range.drop(
            time_range[
                (time_range.month == 1)
                | (time_range.month == 2)
                | (time_range.month == 3)
                | (time_range.month == 4)
                | (time_range.month == 5)
                | (time_range.month == 9)
                | (time_range.month == 10)
                | (time_range.month == 11)
                | (time_range.month == 12)
            ]
        )
    elif x == "spring":
        return time_range.drop(
            time_range[
                (time_range.month == 1)
                | (time_range.month == 2)
                | (time_range.month == 3)
                | (time_range.month == 4)
                | (time_range.month == 5)
                | (time_range.month == 6)
                | (time_range.month == 7)
                | (time_range.month == 8)
                | (time_range.month == 12)
            ]
        )
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

        # Make 'timerange' time-zone aware
        localtides = time_range.tz_localize(tz=pytz.UTC)

        # Replace the UTC datetimes from timerange with local times
        ModTides = pd.DataFrame(index=localtides)

        # Return the difference in years for the time-period.
        # Round up to ensure all modelledtide datetimes are captured in
        # the solar model
        diff = pd.to_datetime(end) - pd.to_datetime(start)
        diff = int(ceil(diff.days / 365))

        # Set to UTC time
        local_tz = 0

        # Model sunrise and sunset
        sun_df = sunriset.to_pandas(
            startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff
        )

        # Set the index as a datetimeindex to match the ModTides ds
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

        # Remove local timezone timestamp column in ModTides
        # dataframe. Xarray doesn't handle timezone aware datetimeindexes
        # 'from_dataframe' very well.
        ModTides.index = ModTides.index.tz_localize(tz=None)

        # Create an xr Dataset from the ModTides pd.dataframe
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

def build_expected_grid(time_range_peaks, expected_gap_days=14.75):
    """
    Fit a regular grid to detected peaks using the median observed gap
    and the first detected peak as the anchor point.
    """
    # Guard against empty or single-element arrays
    if len(time_range_peaks) < 2:
        return pd.DatetimeIndex([])
        
    median_gap = pd.to_timedelta(
        np.median((time_range_peaks[1:] - time_range_peaks[:-1]).days), "d"
    )
    # Build expected peak times from first detected peak
    n_cycles = int((time_range_peaks[-1] - time_range_peaks[0]) / median_gap) + 1
    expected = pd.DatetimeIndex([
        time_range_peaks[0] + i * median_gap for i in range(n_cycles)
    ])
    return expected


def fill_peak_gaps(
    time_range_peaks,
    modelledtides_1d,
    search_da=None,       
    reference_peaks=None,
    expected_gap_days=14.75,
    gap_tolerance=1.5,
    search_width=0.6,
    use_min=False,
):
    # Use search_da if provided, otherwise fall back to modelledtides_1d
    search_series = search_da if search_da is not None else modelledtides_1d

        # Guard against empty or single-element arrays
    if len(time_range_peaks) < 2:
        # print(f"Warning: fill_peak_gaps received only {len(time_range_peaks)} peaks — skipping gap fill")
        return time_range_peaks
        
    threshold = pd.Timedelta(expected_gap_days * gap_tolerance, "d")
    gaps = time_range_peaks[1:] - time_range_peaks[:-1]
    filled_peaks = list(time_range_peaks)

    # Build an expected grid from the detected peaks themselves
    expected_grid = build_expected_grid(time_range_peaks, expected_gap_days)

    for i, gap in enumerate(gaps):
        if gap > threshold:
            gap_start = time_range_peaks[i]
            gap_end = time_range_peaks[i + 1]

            # Find expected peaks that fall within this gap
            expected_in_gap = expected_grid[
                (expected_grid > gap_start) & (expected_grid < gap_end)
            ]

            if len(expected_in_gap) == 0:
                # print(f"Warning: no expected peaks found in gap {gap_start.date()} → {gap_end.date()}, skipping")
                continue

            # Search around each expected peak position
            half = pd.Timedelta(expected_gap_days * search_width, "d")

            for expected_peak in expected_in_gap:
                search_start = expected_peak - half
                search_end = expected_peak + half

                # Clamp search window to within the actual gap boundaries
                search_start = max(search_start, gap_start)
                search_end = min(search_end, gap_end)
                
                # Skip if window is invalid after clamping
                if search_end <= search_start:
                    # print(f"Warning: invalid search window after clamping "
                    #       f"{search_start.date()} → {search_end.date()}, skipping")
                    continue

                # Optionally narrow the search window using reference peaks
                if reference_peaks is not None:
                    ref_in_window = reference_peaks[
                        (reference_peaks > gap_start) & (reference_peaks < gap_end)
                    ]
                    if len(ref_in_window) >= 2:
                        # Narrow to between the bracketing reference peaks
                        search_start = max(search_start, ref_in_window[0])
                        search_end = min(search_end, ref_in_window[1])
                    elif len(ref_in_window) == 1:
                        # Use reference peak as a tighter anchor
                        search_start = max(search_start, ref_in_window[0] - half / 2)
                        search_end = min(search_end, ref_in_window[0] + half / 2)

                if search_end <= search_start:
                    # print(f"Warning: invalid search window after reference narrowing "
                    #       f"{search_start.date()} → {search_end.date()}, skipping")
                    continue

                window_tides = search_series.sel(time=slice(search_start, search_end))

                if len(window_tides.time) == 0:
                    print(f"Warning: no data in search window {search_start.date()} → {search_end.date()}")
                    continue
            
                best_time = (
                    window_tides.idxmin(dim="time").values
                    if use_min
                    else window_tides.idxmax(dim="time").values
                )
                filled_peaks.append(pd.Timestamp(best_time))
                # print(
                #     f"Gap of {gap.days}d between {gap_start.date()} and {gap_end.date()} "
                #     f"— inserted peak at {pd.Timestamp(best_time).date()} "
                #     f"(expected ~{expected_peak.date()})"
                # )

    return pd.DatetimeIndex(sorted(filled_peaks))

def exposure(
    dem,
    start_date,
    end_date,
    modelled_freq="30min",
    tide_model="EOT20",
    tide_model_dir="/var/share/tide_models",
    filters=None,
    filters_combined=None,
    run_id=None,
    log=None,
    return_tide_modelling=False,
    phases =4
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
    into a 1D timeseries across the nominated area of interest.

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
        supported by the `eo-tides` Python package. Options include:
        - "EOT20" (default)
        - "TPXO10-atlas-v2-nc"
        - "FES2022"
        - "FES2022_extrapolated"
        - "FES2014"
        - "FES2014_extrapolated"
        - "GOT5.6"
        - "ensemble" (experimental: combine all above into single ensemble)
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `eo-tides.utils.list_models`.
    filters : list of strings, optional
        An optional list of customisation options to input into the tidal
        modelling to calculate exposure. Filters include:
        - 'unfiltered': calculates exposure for the full input time period,
        - 'dry': Southern Hemisphere dry season, defined as April to
          September
        - 'wet': Southern Hemisphere wet season, defined as October to
          March
        - 'summer', 'autumn', 'winter', 'spring': exposure during
          specific seasons
        - 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
          'oct', 'nov', 'dec': exposure during specific months
        - 'daylight': all tide heights occurring between sunrise and
          sunset local time
        - 'night': all tide heights occurring between sunset and sunrise
          local time
        - 'spring_high', high tide exposure during the fortnightly spring tide cycle,
        - 'spring_low', low tide exposure during the fortnightly spring tide cycle,
        - 'neap_high', high tide exposure during the fortnightly neap tide cycle,
        - 'neap_low', low tide exposure during the fortnightly neap tide cycle,
        - 'hightide', all tide heights greater than or equal to the local lowest high
        tide heights in high temporal resolution tidal modelling,
        - 'lowtide' all tide heights lower than or equal to the local highest low tide
        heights in high temporal resolution tidal modelling,
        Defaults to ['unfiltered'] if none supplied.
    filters_combined : list of two-object tuples, optional
        An optional list of paired customisation options from which to
        calculate exposure. Filters must be sourced from the list under
        'filters'. Example: to calculate exposure
        during daylight hours in the wet season is
        [('wet', 'daylight')]. Multiple tuple pairs are supported.
        Defaults to None.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.
    return_tide_modelling  :  Boolean
        When `True`, returns the full epoch tide modelling, as well
        as filtered tide model datetimes and heights for all filter
        options. If true, ensure the function call is set to return
        exposure_ds, modelledtides_ds, modelledtides_1d,timeranges.
        If false, set the function call to return exposure_ds and
        modelledtides_ds only. Default = False.

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
    modelledtides_1d  :  xarray.DataArray
        The 'mean' 1D high temporal resolution tide model for the area of
        interest. Returned when return_tide_modelling = True.
    timeranges  :  dict
        A dictionary of filtered DatetimeIndex's, corresponding to the
        filtered dates of interest from modelledtides_1d. Returned
        when return_tide_modelling = True.

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
    'neap_high', 'neap_low', 'hightide', 'lowtide'

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

    # Define the temporal filters
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
    # Define the spatial filters
    sptl_filters = [
        "neaptide",
        "springtide",
        "spring_high",
        "spring_low",
        "neap_high",
        "neap_low",
        "hightide",
        "lowtide",
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
    modelledtides_lowres = pixel_tides(
        data=dem,
        time=time_range,
        model=tide_model,
        directory=tide_model_dir,
        resample=False,
    )

    # Calculate a 1D tide height time series to use with filtered exposure calc's
    modelledtides_1d = modelledtides_lowres.mean(dim=["x", "y"])

    # Calculate quantiles and reproject low resolution tide data to
    # pixel resolution if any filter is "unfiltered"
    if "unfiltered" in filters:

        # Convert to quantiles, and make sure CRS is present
        modelledtides_lowres_quantiles = (
            modelledtides_lowres.quantile(q=calculate_quantiles, dim="time")
            .astype(modelledtides_lowres.dtype)
            .odc.assign_crs(dem.odc.geobox.crs)
        )

        # Reproject into pixel resolution
        modelledtides_highres = _pixel_tides_resample(
            tides_lowres=modelledtides_lowres_quantiles,
            dask_chunks=dem.shape,
            gbox=dem.odc.geobox,
        )

        # Add pixel resolution tides into to output dataset
        modelledtides_ds["unfiltered"] = modelledtides_highres

    # Filter the input timerange to include only dates or tide ranges of
    # interest if filters is not None:
    for x in filters:
        if x in temp_filters:
            print(f"Filtering timesteps for {x}")
            timeranges[x] = temporal_filters(x, time_range, dem)
        #I think this is where a spatial filter calculation goes, returning timeranges[x] only...
        if x in sptl_filters:
            print(f"Filtering timesteps for {x}")
            timeranges[x] = spatial_filters(modelled_freq, 
                                            x,
                                            modelledtides_1d,
                                            modelledtides_lowres,
                                            phases
                                           )
    
    # Intersect the filters of interest to extract the common datetimes for
    # calculation of combined filters
    if filters_combined is not None:
        for x in filters_combined:
            y = x[0]
            z = x[1]
            timeranges[str(y + "_" + z)] = timeranges[y].intersection(timeranges[z])

    # Intersect datetimes of interest with the 1D tidal model

    for x in timeranges:
        # Extract filtered datetimes from the full tidal model
        modelledtides_x = modelledtides_1d.sel(time=timeranges[str(x)])

        # Calculate quantile values on remaining tide heights
        modelledtides_x = (
            modelledtides_x.quantile(q=calculate_quantiles, dim="time")
            .to_dataset()
            .tide_height
        )

        # Add modelledtides_x to output dataset
        modelledtides_ds[str(x)] = modelledtides_x

    

    # Calculate exposure per filter
    for x in modelledtides_ds:
        print(f"Calculating {x} exposure")

        exposure_ds[str(x)] = exposure_percentiles(modelledtides_ds[str(x)], dem)

    if return_tide_modelling:
        return exposure_ds, modelledtides_ds, modelledtides_1d, timeranges
    else:
        return exposure_ds, modelledtides_ds

def exposure_percentiles(modelledtides_ds, dem):
    # Calculate the tide-height difference between the elevation
    # value and each percentile value per pixel
    diff = abs(modelledtides_ds - dem)

    # Take the percentile of the smallest tide-height difference as
    # the exposure % per pixel
    idxmin = diff.idxmin(dim="quantile")

    # Reorder dimensions
    if "time" in list(idxmin.dims):
        idxmin = idxmin.transpose("time", "y", "x")
    else:
        idxmin = idxmin.transpose("y", "x")

    # Convert to percentage and add as variable in exposure dataset
    exposure = idxmin * 100

    return exposure

def detect_neaps(time_range_springtides, 
                 springpeaks,
                 calc_low = False):
    """
    Detect neap high tide peaks as the minimum tide_height value in the
    high tide envelope (tide_maxima) between each successive pair of
    spring high tide peaks.

    Parameters
    ----------
    time_range_springtides : pd.DatetimeIndex
        Filled spring high or low tide peak datetimes, one per half lunar cycle.
    spring_peaks : xr.Dataset
        Dataset of all local high or low tide maxima from the full timeseries,
        with a 'tide_height' variable and 'time' dimension.
    calc_low  :  bool
        Calculate neap high tides by default. Set to True to calculate
        neap low tides.

    Returns
    -------
    time_range_neaphigh : pd.DatetimeIndex
        Detected neap high tide peak datetimes, one per successive pair
        of spring high peaks.
    """

    # # Drop duplicate timestamps which would produce zero-width windows
    # spring_peaks = time_range_springtides.drop_duplicates()

    neap_times = []

    for i in range(len(time_range_springtides) - 1):
        peak_start = time_range_springtides[i]
        peak_end = time_range_springtides[i + 1]

        # Skip zero-width or negative windows after deduplication
        if peak_end <= peak_start:
            print(f"Warning: skipping invalid window {peak_start} → {peak_end}")
            continue

        # Slice the high tide envelope between this pair of spring highs
        window = springpeaks.sel(time=slice(peak_start, peak_end))

        # Skip if no high tide maxima fall within this window
        if len(window.time) == 0:
            print(f"Warning: no tide_maxima found between "
                  f"{peak_start.date()} → {peak_end.date()}, skipping")
            continue

        if calc_low is False:
            # The neap high is the lowest high tide peak in the window
            best_time = window.tide_height.idxmin(dim="time").values
        else:
            # The neap low is the highest low tide peak in the window
            best_time = window.tide_height.idxmax(dim="time").values
            
        neap_times.append(pd.Timestamp(best_time))

    time_range_neap = pd.DatetimeIndex(neap_times)

    return time_range_neap

def spatial_filters(
    modelled_freq,
    x,
    modelledtides_1d,
    modelledtides_lowres,
    phases=4
):
    """
    Identify and extract spatial-specific dates and times to feed
    into tidal modelling for custom exposure calculations.

    phases  |  int
        The number of phases to model each lunar month. Defaults to
        4, evenly distributing tides across 2 neap and 2 spring tide
        cycles each month. Alternative: 8, distributing tides across
        2 spring, neap, crescent and gibbous moons per lunar month. 8
        phases narrows the window of possible dates upon which a
        spring or neap tide can be modelled. 4 phases will model all 
        tides into either a spring or neap phase.
    """

    # Extract the modelling freq units
    freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
    freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
    # Extract the number of modelled timesteps per half lunar cycle (where lunar cycle = 29.5 days)
    mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
    # Calculate the 'order' window for spring tide calculation
    order = int(mod_timesteps / 2)

    # Spring highs: largest maxima in the high tide envelope
    modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
    springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
    time_range_springhigh = pd.to_datetime(springpeaks.time)

    # Spring lows: smallest minima in the low tide envelope
    modelledtides_1d_peakslow = argrelmin(modelledtides_1d.values, order=order)[0]
    springpeakslow = modelledtides_1d.isel(time=modelledtides_1d_peakslow).to_dataset()
    time_range_springlow = pd.to_datetime(springpeakslow.time)

    time_range_springhigh = fill_peak_gaps(
            time_range_springhigh,
            modelledtides_1d,
            reference_peaks=None,
            expected_gap_days=14.75,
            search_width=0.4,
            use_min=False,
            )
    time_range_springlow = fill_peak_gaps(
            time_range_springlow,
            modelledtides_1d,
            reference_peaks=None,
            expected_gap_days=14.75,
            search_width=0.4,
            use_min=True,
            )
    # Find all high tide maxima from full timeseries
    tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
    tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()

    # Find all low tide minima from full timeseries
    tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
    tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

    # Calculate neap peaks
    time_range_neaphigh = detect_neaps(time_range_springhigh, tide_maxima)
    time_range_neaplow = detect_neaps(time_range_springlow, tide_minima, calc_low=True)
    
    if x == "spring_high":
        return time_range_springhigh.drop_duplicates()
        
    if x == "neap_high":
        return time_range_neaphigh
        
    if x == "spring_low":
        return time_range_springlow.drop_duplicates()
        
    if x == "neap_low":
        return time_range_neaplow
    
    if x in ["neaptide", "springtide"]:
    
        expected_cycle = 14.75
        tolerance = 0.4
        lower = pd.Timedelta(expected_cycle * (1 - tolerance), "d")
        upper = pd.Timedelta(expected_cycle * (1 + tolerance), "d")
        half_window = pd.Timedelta(expected_cycle / 4, "d")
    
        spring_list = []
        neap_list = []
    
        for i in range(len(time_range_neaphigh) - 1):
            gap = time_range_neaphigh[i + 1] - time_range_neaphigh[i]
    
            if lower <= gap <= upper:
                # Neap period: window around this neap peak
                neap_window = modelledtides_1d.sel(
                    time=slice(
                        time_range_neaphigh[i] - half_window,
                        time_range_neaphigh[i] + half_window
                    )
                )
                if len(neap_window.time) > 0:
                    neap_list.append(neap_window)
    
                # Spring period: window around midpoint to next neap peak
                midpoint = time_range_neaphigh[i] + gap / 2
                spring_window = modelledtides_1d.sel(
                    time=slice(midpoint - half_window, midpoint + half_window)
                )
                if len(spring_window.time) > 0:
                    spring_list.append(spring_window)
            else:
                print(f"Skipping pair at {time_range_neaphigh[i].date()} → "
                      f"{time_range_neaphigh[i+1].date()} — gap of {gap.days}d "
                      f"outside expected range")
    
        # Handle the last neap peak if it has a valid predecessor
        if len(time_range_neaphigh) > 1:
            last_gap = time_range_neaphigh[-1] - time_range_neaphigh[-2]
            if lower <= last_gap <= upper:
                neap_window = modelledtides_1d.sel(
                    time=slice(
                        time_range_neaphigh[-1] - half_window,
                        time_range_neaphigh[-1] + half_window
                    )
                )
                if len(neap_window.time) > 0:
                    neap_list.append(neap_window)
    
        if x in ['springtide']:
            springtide = xr.concat(spring_list, dim="time")
            return pd.to_datetime(springtide.time)
        if x in ['neaptide']:
            neaptide = xr.concat(neap_list, dim="time")
            return pd.to_datetime(neaptide.time)

    # ## Block 1: neaptide / springtide
    # if x in ["neaptide", "springtide"]:

    #     # Fill gaps using observed median gap as expected spacing
    #     actual_spring_gap = np.median(
    #         (time_range_spring[1:] - time_range_spring[:-1]).days
    #     )
    #     actual_neap_gap = np.median(
    #         (time_range_neap[1:] - time_range_neap[:-1]).days
    #     )
    #     time_range_spring = fill_peak_gaps(
    #         time_range_spring,
    #         modelledtides_1d,
    #         reference_peaks=time_range_neap,
    #         expected_gap_days=actual_spring_gap,
    #         use_min=False,
    #     )
    #     time_range_neap = fill_peak_gaps(
    #         time_range_neap,
    #         modelledtides_1d,
    #         reference_peaks=time_range_spring,
    #         expected_gap_days=actual_neap_gap,
    #         use_min=False,
    #     )

    #     # Fixed window of +/- half lunar cycle around each detected peak
    #     half_cycle = pd.Timedelta(14.75 / 2, "d")

    #     spring_list = []
    #     for peak in time_range_spring:
    #         window = modelledtides_1d.sel(
    #             time=slice(peak - half_cycle, peak + half_cycle)
    #         )
    #         if len(window.time) > 0:
    #             spring_list.append(window)

    #     neap_list = []
    #     for peak in time_range_neap:
    #         window = modelledtides_1d.sel(
    #             time=slice(peak - half_cycle, peak + half_cycle)
    #         )
    #         if len(window.time) > 0:
    #             neap_list.append(window)

    #     if x in ['springtide']:
    #         springtide = xr.concat(spring_list, dim="time")
    #         return pd.to_datetime(springtide.time)
    #     if x in ['neaptide']:
    #         neaptide = xr.concat(neap_list, dim="time")
    #         return pd.to_datetime(neaptide.time)

    # ## Block 2: spring_high / spring_low / neap_high / neap_low
    # if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

    #     # For low tide filters, recompute peaks on troughs
    #     if x in ["spring_low", "neap_low"]:
    #         tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
    #         tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

    #         spring_low_idx = argrelmin(tide_minima.tide_height.values, order=order)[0]
    #         modelledtides_1d_peaks_low = tide_minima_idx[spring_low_idx]
    #         springpeaks_low = modelledtides_1d.isel(
    #             time=modelledtides_1d_peaks_low
    #         ).to_dataset()
    #         time_range_spring = pd.to_datetime(springpeaks_low.time)

    #         neap_low_idx = argrelmax(tide_minima.tide_height.values, order=order_nh)[0]
    #         neappeaks_low = tide_minima.isel(time=neap_low_idx)
    #         time_range_neap = pd.to_datetime(neappeaks_low.time)

    #     if x in ["spring_high", "spring_low"]:
    #         return time_range_spring
    #     if x in ["neap_high", "neap_low"]:
    #         return time_range_neap

# def spatial_filters(
#     modelled_freq,
#     x,
#     modelledtides_1d,
#     modelledtides_lowres,
#     phases=4
# ):
#     """
#     Identify and extract spatial-specific dates and times to feed
#     into tidal modelling for custom exposure calculations.

#     phases  |  int
#         The number of phases to model each lunar month. Defaults to
#         4, evenly distributing tides across 2 neap and 2 spring tide
#         cycles each month. Alternative: 8, distributing tides across
#         2 spring, neap, crescent and gibbous moons per lunar month. 8
#         phases narrows the window of possible dates upon which a
#         spring or neap tide can be modelled. 4 phases will model all 
#         tides into either a spring or neap phase.
#     """

#     # Extract the modelling freq units
#     freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
#     freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
#     # Extract the number of modelled timesteps per half lunar cycle (29.5 days)
#     mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
#     order = int(mod_timesteps / 2)

#     # ---- SHARED PEAK DETECTION ----
#    # Find all high tide maxima
#     tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
#     tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()
    
#     # Spring highs: largest maxima in the high tide envelope
#     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
#     springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
#     time_range_spring = pd.to_datetime(springpeaks.time)
    
#     # Neap highs: smallest maxima in the high tide envelope
#     order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
#     neap_peak_idx = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
#     neappeaks = tide_maxima.isel(time=neap_peak_idx)
#     time_range_neap = pd.to_datetime(neappeaks.time)
    
#     print(f"order: {order}, order_nh: {order_nh}")
#     print(f"tide_maxima length: {len(tide_maxima.time)}, "
#           f"spring peaks: {len(modelledtides_1d_peaks)}")
#     print(f"Spring peaks detected: {len(time_range_spring)}, "
#           f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
#           f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
#     print(f"Neap peaks detected: {len(time_range_neap)}, "
#           f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
#           f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
#     # ---- END SHARED DETECTION ----

#     ## Block 1: neaptide / springtide
#     if x in ["neaptide", "springtide"]:

#         # Use observed median gap for fill_peak_gaps
#         actual_spring_gap = np.median(
#             (time_range_spring[1:] - time_range_spring[:-1]).days
#         )
#         actual_neap_gap = np.median(
#             (time_range_neap[1:] - time_range_neap[:-1]).days
#         )

#         # Fill gaps in both lists using expected grid as primary reference
#         # and each other as loose secondary constraint
#         time_range_spring = fill_peak_gaps(
#             time_range_spring,
#             modelledtides_1d,
#             reference_peaks=time_range_neap,
#             expected_gap_days=actual_spring_gap,
#             use_min=False,
#         )
#         time_range_neap = fill_peak_gaps(
#             time_range_neap,
#             modelledtides_1d,
#             reference_peaks=time_range_spring,
#             expected_gap_days=actual_neap_gap,
#             use_min=False,
#         )

#         # Identifying quartile ranges between neap and spring highs
#         idx1 = time_range_spring
#         idx2 = time_range_neap

#         if idx2[0] < idx1[0]:
#             idx2 = time_range_neap[1:]

#         if len(idx1) != len(idx2):
#             min_len = min(len(idx1), len(idx2))
#             idx1 = idx1[:min_len]
#             idx2 = idx2[:min_len]

#         # Interleave idx1 and idx2
#         interleaved = pd.DatetimeIndex(np.ravel(np.column_stack([idx1, idx2])))

#         # Calculate quartile boundaries between consecutive elements
#         delta = interleaved[1:] - interleaved[:-1]
#         q1 = interleaved[:-1] + delta * 0.25
#         q2 = interleaved[:-1] + delta * 0.50
#         q3 = interleaved[:-1] + delta * 0.75

#         spring_list = []
#         neap_list = []

#         if phases == 4:
#             for i in range(0, len(q2) - 1, 2):
#                 neap_list.append(
#                     modelledtides_1d.sel(time=slice(q2[i], q2[i + 1]))
#                 )
#                 if i + 2 < len(q2):
#                     spring_list.append(
#                         modelledtides_1d.sel(time=slice(q2[i + 1], q2[i + 2]))
#                     )

#         if phases == 8:
#             for i in range(0, len(q3) - 1, 2):
#                 neap_list.append(
#                     modelledtides_1d.sel(time=slice(q3[i], q1[i + 1]))
#                 )
#                 if i + 2 < len(q3):
#                     spring_list.append(
#                         modelledtides_1d.sel(time=slice(q3[i + 1], q1[i + 2]))
#                     )

#         if x in ['springtide']:
#             springtide = xr.concat(spring_list, dim="time")
#             return pd.to_datetime(springtide.time)
#         if x in ['neaptide']:
#             neaptide = xr.concat(neap_list, dim="time")
#             return pd.to_datetime(neaptide.time)

#     ## Block 2: spring_high / spring_low / neap_high / neap_low
#     if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

#         # For low tide filters, recompute peaks on troughs
#         if x in ["spring_low", "neap_low"]:
#             tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
#             tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

#             spring_low_idx = argrelmin(tide_minima.tide_height.values, order=order_envelope)[0]
#             modelledtides_1d_peaks_low = tide_minima_idx[spring_low_idx]
#             springpeaks_low = modelledtides_1d.isel(time=modelledtides_1d_peaks_low).to_dataset()
#             time_range_spring = pd.to_datetime(springpeaks_low.time)

#             neap_low_idx = argrelmax(tide_minima.tide_height.values, order=order_envelope)[0]
#             neappeaks_low = tide_minima.isel(time=neap_low_idx)
#             time_range_neap = pd.to_datetime(neappeaks_low.time)

#         actual_spring_gap = np.median(
#             (time_range_spring[1:] - time_range_spring[:-1]).days
#         )
#         actual_neap_gap = np.median(
#             (time_range_neap[1:] - time_range_neap[:-1]).days
#         )

#         # Fill both lists using expected grid as primary, each other as secondary
#         time_range_spring_filled = fill_peak_gaps(
#             time_range_spring,
#             modelledtides_1d,
#             reference_peaks=time_range_neap,
#             expected_gap_days=actual_spring_gap,
#             use_min=(x == "spring_low"),
#         )
#         time_range_neap_filled = fill_peak_gaps(
#             time_range_neap,
#             modelledtides_1d,
#             reference_peaks=time_range_spring_filled,
#             expected_gap_days=actual_neap_gap,
#             use_min=(x == "neap_low"),
#         )

#         if x in ["spring_high", "spring_low"]:
#             return time_range_spring_filled
#         if x in ["neap_high", "neap_low"]:
#             return time_range_neap_filled

# # def fill_peak_gaps(time_range_peaks, 
# #                    modelledtides_1d, 
# #                    expected_gap_days=14.75, 
# #                    gap_tolerance=1.3,
# #                    use_min=False
# #                   ):
# #     """
# #     Identify gaps between consecutive peaks that are significantly larger
# #     than expected (~14.75 days for spring/neap cycles), then search for
# #     the best candidate peak within each gap.

# #     Parameters
# #     ----------
# #     time_range_peaks : pd.DatetimeIndex
# #         Detected spring_high or neap_high peak datetimes.
# #     modelledtides_1d : xr.DataArray
# #         1D tide height timeseries.
# #     expected_gap_days : float
# #         Expected gap between consecutive peaks in days. Default 14.75
# #         (half lunar cycle).
# #     gap_tolerance : float
# #         Multiplier of expected_gap_days above which a gap is considered
# #         anomalous and worth searching. Default 1.5 (i.e. gaps > ~22 days).

# #     Returns
# #     -------
# #     pd.DatetimeIndex
# #         Filled peak datetimes, sorted.
# #     """
# #     threshold = pd.Timedelta(expected_gap_days * gap_tolerance, "d")
# #     gaps = time_range_peaks[1:] - time_range_peaks[:-1]

# #     filled_peaks = list(time_range_peaks)

# #     for i, gap in enumerate(gaps):
# #         if gap > threshold:
# #             # Define search window: centre on the expected midpoint of the gap,
# #             # +/- half the expected cycle length
# #             gap_start = time_range_peaks[i]
# #             gap_end   = time_range_peaks[i + 1]
            
# #             window_centre = gap_start + (gap_end - gap_start) / 2
# #             half_window   = pd.Timedelta(expected_gap_days / 2, "d")
            
# #             search_start = window_centre - half_window
# #             search_end   = window_centre + half_window

# #             # Extract tide heights within the search window
# #             window_tides = modelledtides_1d.sel(
# #                 time=slice(search_start, search_end)
# #             )

# #             if len(window_tides.time) == 0:
# #                 print(f"Warning: no data found in gap window {search_start} → {search_end}")
# #                 continue

# #             # Find the peak (max for spring_high/neap_high; swap to .idxmin for lows)
# #             best_time = window_tides.idxmin(dim="time").values if use_min else window_tides.idxmax(dim="time").values

# #             filled_peaks.append(pd.Timestamp(best_time))
# #             print(f"Gap of {gap.days}d detected between {gap_start.date()} and "
# #                   f"{gap_end.date()} — inserting peak at {pd.Timestamp(best_time).date()}")

# #     return pd.DatetimeIndex(sorted(filled_peaks))

# # def spatial_filters(
# #     modelled_freq,
# #     x,
# #     modelledtides_1d,
# #     modelledtides_lowres,
# #     phases=4
# #     # timeranges,
# #     # calculate_quantiles,
# #     # modelledtides_ds,
# #     # dem,
# #     # exposure,
# # ):
# #     """
# #     Identify and extract spatial-specific dates and times to feed
# #     into tidal modelling for custom exposure calculations.

# #     phases  |  int
# #         The number of phases to model each lunar month. Defaults to
# #         4, evenly distributing tides across 2 neap and 2 spring tide
# #         cycles each month. Alternative: 8, distributing tides across
# #         2 spring, neap, crescent and gibbous moons per lunar month. 8
# #         phases narrows the window of possible dates upon which a
# #         spring or neap tide can be modelled. 4 phases will model all 
# #         tides into either a spring or neap phase.
# #     """

# #     # Extract the modelling freq units
# #     # Split the number and text characters in modelled_freq
# #     freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
# #     freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
# #     # Extract the number of modelled timesteps per half lunar cycle (29.5 days) for neap/spring calcs
# #     mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
# #     ## Identify kwargs for peak detection algorithm
# #     order = int(mod_timesteps / 2)
# #     print (f'order: {order}')

# #     # Tidal regime detection
# #     high_peaks_all = argrelmax(modelledtides_1d.values)[0]
# #     high_peaks_envelope = modelledtides_1d.isel(time=high_peaks_all)
# #     lower_highs = argrelmin(high_peaks_envelope.values)[0]
# #     is_diurnal = len(lower_highs) / len(high_peaks_all) < 0.2
# #     print(f"Tidal regime: {'diurnal' if is_diurnal else 'semi-diurnal'} "
# #           f"(lower_highs ratio: {len(lower_highs) / len(high_peaks_all):.2f})")

# #     # Find all high tide maxima from full timeseries
# #     tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
# #     tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()
    
# #     # Recalculate order for the envelope specifically
# #     # Envelope has ~2 points per day (semi-diurnal), so half lunar cycle in envelope points:
# #     envelope_timesteps = pd.Timedelta(29.5 / 2, "d") / pd.Timedelta(freq_time, freq_unit)
# #     # Adjust order for the fact that envelope has ~2x fewer points than full timeseries
# #     order_envelope = max(1, int(envelope_timesteps / 2 / 2))
    
# #     print(f"order (full timeseries): {order}")
# #     print(f"order_envelope: {order_envelope}")
# #     print(f"tide_maxima length: {len(tide_maxima.time)}")
    
# #     # Spring highs: largest maxima in envelope
# #     spring_peak_idx = argrelmax(tide_maxima.tide_height.values, order=order_envelope)[0]
# #     modelledtides_1d_peaks = tide_maxima_idx[spring_peak_idx]
# #     springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# #     time_range_spring = pd.to_datetime(springpeaks.time)
    
# #     # Neap highs: smallest maxima in envelope, same order
# #     order_nh = order_envelope
# #     neap_peak_idx = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
# #     neappeaks = tide_maxima.isel(time=neap_peak_idx)
# #     time_range_neap = pd.to_datetime(neappeaks.time)
    
# #     print(f"Spring peaks detected: {len(time_range_spring)}, "
# #           f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
# #           f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
# #     print(f"Neap peaks detected: {len(time_range_neap)}, "
# #           f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
# #           f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
# #     # # spring peak detection
# #     # if is_diurnal:
# #     #     print("Diurnal regime — using broad argrelmax for spring peaks")
# #     #     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
# #     # else:
# #     #     print("Semi-diurnal regime — using argrelmax on high tide envelope for spring peaks")
# #     #     envelope_peak_idx = argrelmax(high_peaks_envelope.values, order=order)[0]
# #     #     modelledtides_1d_peaks = high_peaks_all[envelope_peak_idx]

# #     # springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# #     # time_range_spring = pd.to_datetime(springpeaks.time)

# #     # # neap peak detection
# #     # tide_maxima = argrelmax(modelledtides_1d.values)[0]
# #     # tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #     # order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# #     # neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# #     # # if is_diurnal:
# #     # #     print("Diurnal regime — using argrelmin with broader order for neap peaks")
# #     # #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=max(1, order_nh * 2))[0]
# #     # # else:
# #     # #     print("Semi-diurnal regime — excluding spring peaks from high tide envelope for neap peaks")
# #     # #     all_hightide_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nh)[0]
# #     # #     spring_times = pd.to_datetime(springpeaks.time)
# #     # #     all_peak_times = pd.to_datetime(tide_maxima.isel(time=all_hightide_peaks).time)
# #     # #     half_cycle = pd.Timedelta(14.75 / 2, "d")
# #     # #     neap_mask = np.array([
# #     # #         not any(abs(t - s) < half_cycle for s in spring_times)
# #     # #         for t in all_peak_times
# #     # #     ])
# #     # #     neap_peaks = all_hightide_peaks[neap_mask]

# #     # neappeaks = tide_maxima.isel(time=neap_peaks)
# #     # time_range_neap = pd.to_datetime(neappeaks.time)


# #     ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
# #     if x in ["neaptide", "springtide"]:#"spring_high", "spring_low", "neap_high", "neap_low"]:

# #         # # 1D tide modelling workflow
# #         # # apply the peak detection routine
# #         # # if x in ["spring_high", "neap_high"]:
# #         # modelledtides_1d_peaks = argrelmax(
# #         #     modelledtides_1d.values, order=order
# #         # )[0]
# #         # # if x in ["spring_low", "neap_low"]:
# #         # #     modelledtides_1d_peaks = argrelmin(
# #         # #         modelledtides_1d.values, order=order
# #         # #     )[0]
# #         # # if x == "neap_high":
# #         # ## apply the peak detection routine to calculate all the high tide maxima
# #         # tide_maxima = argrelmax(modelledtides_1d.values)[0]
# #         # tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         # ## extract neap high tides based on a half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# #         # order_nh = int(
# #         #     ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# #         # )
# #         # ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
# #         # neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# #         # # if x == "neap_low":
# #         # #     ## apply the peak detection routine to calculate all the low tide maxima
# #         # #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# #         # #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         # #     ## extract neap low tides based on 14 day half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# #         # #     order_nl = int(
# #         # #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# #         # #     )
# #         # #     ## apply the peak detection routine to calculate all the neap low tide maxima within the low tide peaks
# #         # #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
# #         # #     # neap_peaks = argrelmax(tide_maxima.values, order=order_nl)[0]

        
# #         # # if x in ["neap_high", "neap_low"]:
# #         # ## extract neap high tides
# #         # neappeaks = tide_maxima.isel(time=neap_peaks)
# #         # time_range_neap = pd.to_datetime(neappeaks.time)
# #         # # time_range_neap = fill_peak_gaps(time_range_neap, modelledtides_1d, use_min=False)
# #         #     # return time_range
# #         #     # Extract the peak height dates
# #         #     # tide_cq = neappeaks.quantile(q=calculate_quantiles, dim="time")

# #         # # if x in ["spring_high", "spring_low"]:
# #         # # select for indices associated with peaks
# #         # springpeaks = modelledtides_1d.isel(
# #         #     time=modelledtides_1d_peaks
# #         # ).to_dataset()
# #         # # Save datetimes for calculation of combined filter exposure
# #         # time_range_spring = pd.to_datetime(springpeaks.time)
# #         # # time_range_spring = fill_peak_gaps(time_range_spring, modelledtides_1d, use_min=False)
# #         #     # return time_range
# #         #     # Extract the peak height dates
# #         #     # tide_cq = springpeaks.quantile(q=calculate_quantiles, dim="time")

# #         # # Fill gaps in both lists using expected grid as primary reference
# #         # # and each other as secondary reference only after both are initially filled

# #         print(f"Spring peaks detected: {len(time_range_spring)}, "
# #               f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
# #               f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
# #         print(f"Neap peaks detected: {len(time_range_neap)}, "
# #               f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
# #               f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
        
# #         time_range_spring = fill_peak_gaps(
# #             time_range_spring,
# #             modelledtides_1d,
# #             reference_peaks=time_range_neap,  # raw neap as loose secondary constraint only
# #             use_min=False,
# #         )
# #         time_range_neap = fill_peak_gaps(
# #             time_range_neap,
# #             modelledtides_1d,
# #             reference_peaks=time_range_spring,  # filled spring as secondary constraint
# #             use_min=False,
# #         )

        
# #         # Identifying quartile ranges between neap and spring highs
# #         idx1 = time_range_spring
# #         idx2 = time_range_neap
        
# #         if idx2[0] < idx1[0]:
# #             idx2=time_range_neap[1:]
        
# #         # Ensure peak lists are the same length
# #         if len(idx1) != len(idx2):
# #             min_len = min(len(idx1), len(idx2))
# #             idx1 = idx1[:min_len]
# #             idx2 = idx2[:min_len]
        
# #         # Interleave idx1 and idx2
# #         interleaved = pd.DatetimeIndex(np.ravel(np.column_stack([idx1, idx2])))
        
# #         # Calculate quartile boundaries between consecutive elements
# #         delta = interleaved[1:] - interleaved[:-1]
# #         q1 = interleaved[:-1] + delta * 0.25
# #         q2 = interleaved[:-1] + delta * 0.50  # midpoints (same as before)
# #         q3 = interleaved[:-1] + delta * 0.75

# #         spring_list = []
# #         neap_list = []

# #         if phases == 4:
# #             # Split modelledtides_1d collecting midpoint (q2) tide heights for neap/spring high tide periods           
# #             for i in range(0, len(q2) - 1, 2):
# #                 neap_list.append(
# #                     modelledtides_1d.sel(time=slice(q2[i], q2[i + 1]))
# #                 )
# #                 if i + 2 < len(q2):
# #                     spring_list.append(
# #                         modelledtides_1d.sel(time=slice(q2[i + 1], q2[i + 2]))
# #                     )

# #         if phases == 8:
# #             # Split modelledtides_1d collecting q3 to q1 tide heights for neap/spring high tide periods            
# #             for i in range(0, len(q3) - 1, 2):
# #                 neap_list.append(
# #                     modelledtides_1d.sel(time=slice(q3[i], q1[i + 1]))
# #                 )
# #                 if i + 2 < len(q3):
# #                     spring_list.append(
# #                         modelledtides_1d.sel(time=slice(q3[i + 1], q1[i + 2]))
# #                     )

# #         if x in ['springtide']:
# #             springtide = xr.concat(spring_list, dim="time")
# #             springtide = pd.to_datetime(springtide.time)
# #             return springtide
# #         if x in ['neaptide']:
# #             neaptide = xr.concat(neap_list, dim="time")
# #             neaptide = pd.to_datetime(neaptide.time)
# #             return neaptide


# #     ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
# #     if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

# #         # For low tide filters, recompute peaks on troughs
# #         if x in ["spring_low", "neap_low"]:
# #             modelledtides_1d_peaks_low = argrelmin(modelledtides_1d.values, order=order)[0]
# #             springpeaks_low = modelledtides_1d.isel(time=modelledtides_1d_peaks_low).to_dataset()
# #             time_range_spring = pd.to_datetime(springpeaks_low.time)

# #             tide_minima = argrelmin(modelledtides_1d.values)[0]
# #             tide_minima = modelledtides_1d.isel(time=tide_minima).to_dataset()
# #             order_nl = int(ceil((len(tide_minima.time) / (len(modelledtides_1d_peaks_low)) / 2)))
# #             neap_peaks_low = argrelmax(tide_minima.tide_height.values, order=order_nl)[0]
# #             neappeaks_low = tide_minima.isel(time=neap_peaks_low)
# #             time_range_neap = pd.to_datetime(neappeaks_low.time)

# #         # Fill both lists
# #         time_range_spring_filled = fill_peak_gaps(
# #             time_range_spring,
# #             modelledtides_1d,
# #             reference_peaks=time_range_neap,
# #             use_min=(x == "spring_low"),
# #         )
# #         time_range_neap_filled = fill_peak_gaps(
# #             time_range_neap,
# #             modelledtides_1d,
# #             reference_peaks=time_range_spring_filled,
# #             use_min=(x == "neap_low"),
# #         )

# #         if x in ["spring_high", "spring_low"]:
# #             return time_range_spring_filled
# #         if x in ["neap_high", "neap_low"]:
# #             return time_range_neap_filled


# #         #         # Compute both raw peak lists before filling either
# #         # if x in ["spring_high", "neap_high"]:
# #         #     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
# #         # if x in ["spring_low", "neap_low"]:
# #         #     modelledtides_1d_peaks = argrelmin(modelledtides_1d.values, order=order)[0]
        
# #         # # Always compute both spring and neap raw lists regardless of x
# #         # springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# #         # time_range_spring = pd.to_datetime(springpeaks.time)
        
# #         # if x in ["neap_high", "spring_high"]:
# #         #     tide_maxima = argrelmax(modelledtides_1d.values)[0]
# #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         #     order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# #         #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
# #         # if x in ["neap_low", "spring_low"]:
# #         #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         #     order_nl = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# #         #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
        
# #         # neappeaks = tide_maxima.isel(time=neap_peaks)
# #         # time_range_neap = pd.to_datetime(neappeaks.time)
        
# #         # # Fill both lists using expected grid as primary, each other as secondary
# #         # time_range_spring_filled = fill_peak_gaps(
# #         #     time_range_spring,
# #         #     modelledtides_1d,
# #         #     reference_peaks=time_range_neap,  # raw neap as loose secondary constraint
# #         #     use_min=(x == "spring_low"),
# #         # )
# #         # time_range_neap_filled = fill_peak_gaps(
# #         #     time_range_neap,
# #         #     modelledtides_1d,
# #         #     reference_peaks=time_range_spring_filled,  # filled spring as secondary constraint
# #         #     use_min=(x == "neap_low"),
# #         # )
        
# #         # if x in ["spring_high", "spring_low"]:
# #         #     return time_range_spring_filled
# #         # if x in ["neap_high", "neap_low"]:
# #         #     return time_range_neap_filled

# #         # # 1D tide modelling workflow
# #         # # apply the peak detection routine
# #         # if x in ["spring_high", "neap_high"]:
# #         #     modelledtides_1d_peaks = argrelmax(
# #         #         modelledtides_1d.values, order=order
# #         #     )[0]
# #         # if x in ["spring_low", "neap_low"]:
# #         #     modelledtides_1d_peaks = argrelmin(
# #         #         modelledtides_1d.values, order=order
# #         #     )[0]
# #         # if x == "neap_high":
# #         #     ## apply the peak detection routine to calculate all the high tide maxima
# #         #     tide_maxima = argrelmax(modelledtides_1d.values)[0]
# #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         #     ## extract neap high tides based on a half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# #         #     order_nh = int(
# #         #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# #         #     )
# #         #     ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
# #         #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# #         # if x == "neap_low":
# #         #     ## apply the peak detection routine to calculate all the low tide maxima
# #         #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# #         #     ## extract neap low tides based on 14 day half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# #         #     order_nl = int(
# #         #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# #         #     )
# #         #     ## apply the peak detection routine to calculate all the neap low tide maxima within the low tide peaks
# #         #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
# #         #     # neap_peaks = argrelmax(tide_maxima.values, order=order_nl)[0]

        
# #         # if x in ["neap_high", "neap_low"]:
# #         #     ## extract neap high tides
# #         #     neappeaks = tide_maxima.isel(time=neap_peaks)
# #         #     time_range = pd.to_datetime(neappeaks.time)
# #         #     time_range = fill_peak_gaps(time_range, modelledtides_1d, use_min=(x == "neap_low"))
# #         #     return time_range
# #         #     # Extract the peak height dates
# #         #     # tide_cq = neappeaks.quantile(q=calculate_quantiles, dim="time")

# #         # if x in ["spring_high", "spring_low"]:
# #         #     # select for indices associated with peaks
# #         #     springpeaks = modelledtides_1d.isel(
# #         #         time=modelledtides_1d_peaks
# #         #     ).to_dataset()
# #         #     # Save datetimes for calculation of combined filter exposure
# #         #     time_range = pd.to_datetime(springpeaks.time)
# #         #     time_range = fill_peak_gaps(time_range, modelledtides_1d, use_min=(x == "spring_low"))
# #         #     return time_range
# #         #     # Extract the peak height dates
# #         #     # tide_cq = springpeaks.quantile(q=calculate_quantiles, dim="time")






# #     # if x == "hightide":
# #     #     # calculate all the high tide maxima
# #     #     high_peaks = argrelmax(modelledtides_1d.values)[0]
# #     #     # extract all hightide peaks
# #     #     high_peaks2 = modelledtides_1d.isel(time=high_peaks)
# #     #     # identify all lower hightide peaks
# #     #     lowhigh_peaks = argrelmin(high_peaks2.values)[0]
# #     #     # extract all lower hightide peaks
# #     #     lowhigh_peaks2 = high_peaks2.isel(time=lowhigh_peaks)

# #     #     # Test for diurnal tidal regimes on the assumption that semi-diurnal and mixed tidal settings
# #     #     # should have approximately equal proportions of daytime and nighttime hightide peaks
# #     #     if len(lowhigh_peaks) / len(high_peaks) < 0.2:
# #     #         # timeranges[str(x)] = pd.to_datetime(high_peaks2.time)
# #     #         return pd.to_datetime(high_peaks2.time)
# #     #         # tide_cq = high_peaks2.quantile(
# #     #         #     q=calculate_quantiles, dim="time"
# #     #         # ).to_dataset()
# #     #     else:
# #     #         # interpolate the lower hightide curve
# #     #         low_high_linear = interp(
# #     #             np.arange(0, len(modelledtides_1d)),
# #     #             high_peaks[lowhigh_peaks],
# #     #             lowhigh_peaks2.values,
# #     #         )
# #     #         # Extract all tides higher than/equal to the extrapolated lowest high tide line
# #     #         hightide = modelledtides_1d.where(
# #     #             modelledtides_1d >= low_high_linear, drop=True
# #     #         )
# #     #         ## Save datetimes for calculation of combined filter exposure
# #     #         time_range = pd.to_datetime(hightide.time)
# #     #         return time_range
# #     #         # tide_cq = hightide.quantile(q=calculate_quantiles, dim="time").to_dataset()

# #     # if x == "lowtide":
# #     #     # calculate all the low tide maxima
# #     #     low_peaks = argrelmin(modelledtides_1d.values)[0]
# #     #     # extract all lowtide peaks
# #     #     low_peaks2 = modelledtides_1d.isel(time=low_peaks)
# #     #     # identify all higher lowtide peaks
# #     #     highlow_peaks = argrelmax(low_peaks2.values)[0]
# #     #     # extract all higher lowtide peaks
# #     #     highlow_peaks2 = low_peaks2.isel(time=highlow_peaks)

# #     #     # Test for diurnal tidal regimes on the assumption that semi-diurnal and mixed tidal settings
# #     #     # should have approximately equal proportions of daytime and nighttime lowtide peaks
# #     #     if len(highlow_peaks) / len(low_peaks) < 0.2:
# #     #         # timeranges[str(x)] = pd.to_datetime(low_peaks2.time)
# #     #         return pd.to_datetime(low_peaks2.time)
# #     #         # tide_cq = low_peaks2.quantile(
# #     #         #     q=calculate_quantiles, dim="time"
# #     #         # ).to_dataset()
# #     #     else:
# #     #         # interpolate the higher lowtide curve
# #     #         high_low_linear = interp(
# #     #             np.arange(0, len(modelledtides_1d)),
# #     #             low_peaks[highlow_peaks],
# #     #             highlow_peaks2.values,
# #     #         )
# #     #         # Extract all tides lower than/equal to the extrapolated higher lowtide line
# #     #         lowtide = modelledtides_1d.where(
# #     #             modelledtides_1d <= high_low_linear, drop=True
# #     #         )
# #     #         ## Save datetimes for calculation of combined filter exposure
# #     #         time_range = pd.to_datetime(lowtide.time)
# #     #         return time_range
# #     #         # tide_cq = lowtide.quantile(q=calculate_quantiles, dim="time").to_dataset()

















































# import sunriset
# import datetime
# import re
# import pytz

# import xarray as xr
# import numpy as np
# import geopandas as gpd
# import pandas as pd

# from math import ceil
# from scipy.signal import argrelmax, argrelmin
# from numpy import interp
# from eo_tides.eo import _pixel_tides_resample, pixel_tides
# from intertidal.utils import configure_logging, round_date_strings


# def temporal_filters(x, time_range, dem):
#     """
#     Identify and extract temporal-specific dates and times to feed into
#     tidal modelling for custom exposure calculations.

#     Parameters
#     -------
#     x : str
#         A string identifier to nominate the temporal filter to
#         calculate in this workflow. Must be one of: 'dry', 'wet',
#         'summer', 'autumn', 'winter', 'spring', 'jan', 'feb',
#         'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
#         'nov', 'dec', 'daylight', 'night'.
#     time_range : pd.DatetimeIndex
#         A fixed frequency pd.DataTimeIndex matching the datetimes used
#         to model tide heights
#     dem : xarray.DataArray
#         xarray.DataArray containing Digital Elevation Model (DEM) data
#         and coordinates and attributes metadata. Used to model sunrise
#         and sunset times for the matching coordinates in dem.

#     Returns
#     -------
#     filtered_time_range : pd.DataTimeIndex
#         An updated pd.DataTimeIndex containing a filtered set of
#         timesteps.
#     """

#     if x == "dry":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 10)  # Wet season: Oct-Mar
#                 | (time_range.month == 11)
#                 | (time_range.month == 12)
#                 | (time_range.month == 1)
#                 | (time_range.month == 2)
#                 | (time_range.month == 3)
#             ]
#         )
#     elif x == "wet":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 4)  # Dry season: Apr-Sep
#                 | (time_range.month == 5)
#                 | (time_range.month == 6)
#                 | (time_range.month == 7)
#                 | (time_range.month == 8)
#                 | (time_range.month == 9)
#             ]
#         )
#     elif x == "summer":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 3)
#                 | (time_range.month == 4)
#                 | (time_range.month == 5)
#                 | (time_range.month == 6)
#                 | (time_range.month == 7)
#                 | (time_range.month == 8)
#                 | (time_range.month == 9)
#                 | (time_range.month == 10)
#                 | (time_range.month == 11)
#             ]
#         )
#     elif x == "autumn":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 1)
#                 | (time_range.month == 2)
#                 | (time_range.month == 6)
#                 | (time_range.month == 7)
#                 | (time_range.month == 8)
#                 | (time_range.month == 9)
#                 | (time_range.month == 10)
#                 | (time_range.month == 11)
#                 | (time_range.month == 12)
#             ]
#         )
#     elif x == "winter":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 1)
#                 | (time_range.month == 2)
#                 | (time_range.month == 3)
#                 | (time_range.month == 4)
#                 | (time_range.month == 5)
#                 | (time_range.month == 9)
#                 | (time_range.month == 10)
#                 | (time_range.month == 11)
#                 | (time_range.month == 12)
#             ]
#         )
#     elif x == "spring":
#         return time_range.drop(
#             time_range[
#                 (time_range.month == 1)
#                 | (time_range.month == 2)
#                 | (time_range.month == 3)
#                 | (time_range.month == 4)
#                 | (time_range.month == 5)
#                 | (time_range.month == 6)
#                 | (time_range.month == 7)
#                 | (time_range.month == 8)
#                 | (time_range.month == 12)
#             ]
#         )
#     elif x == "jan":
#         return time_range.drop(time_range[time_range.month != 1])
#     elif x == "feb":
#         return time_range.drop(time_range[time_range.month != 2])
#     elif x == "mar":
#         return time_range.drop(time_range[time_range.month != 3])
#     elif x == "apr":
#         return time_range.drop(time_range[time_range.month != 4])
#     elif x == "may":
#         return time_range.drop(time_range[time_range.month != 5])
#     elif x == "jun":
#         return time_range.drop(time_range[time_range.month != 6])
#     elif x == "jul":
#         return time_range.drop(time_range[time_range.month != 7])
#     elif x == "aug":
#         return time_range.drop(time_range[time_range.month != 8])
#     elif x == "sep":
#         return time_range.drop(time_range[time_range.month != 9])
#     elif x == "oct":
#         return time_range.drop(time_range[time_range.month != 10])
#     elif x == "nov":
#         return time_range.drop(time_range[time_range.month != 11])
#     elif x == "dec":
#         return time_range.drop(time_range[time_range.month != 12])
#     elif x in ["daylight", "night"]:

#         # Identify the central coordinate directly from the dem GeoBox
#         tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs(
#             "EPSG:4326"
#         ).coords[0]

#         # Calculate the local sunrise and sunset times
#         # Place start and end dates in correct format
#         start = time_range[0]
#         end = time_range[-1]
#         startdate = datetime.date(
#             pd.to_datetime(start).year,
#             pd.to_datetime(start).month,
#             pd.to_datetime(start).day,
#         )

#         # Make 'timerange' time-zone aware
#         localtides = time_range.tz_localize(tz=pytz.UTC)

#         # Replace the UTC datetimes from timerange with local times
#         ModTides = pd.DataFrame(index=localtides)

#         # Return the difference in years for the time-period.
#         # Round up to ensure all modelledtide datetimes are captured in
#         # the solar model
#         diff = pd.to_datetime(end) - pd.to_datetime(start)
#         diff = int(ceil(diff.days / 365))

#         # Set to UTC time
#         local_tz = 0

#         # Model sunrise and sunset
#         sun_df = sunriset.to_pandas(
#             startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff
#         )

#         # Set the index as a datetimeindex to match the ModTides ds
#         sun_df = sun_df.set_index(pd.DatetimeIndex(sun_df.index))

#         # Append the date to each Sunrise and Sunset time
#         sun_df["Sunrise dt"] = sun_df.index + sun_df["Sunrise"]
#         sun_df["Sunset dt"] = sun_df.index + sun_df["Sunset"]

#         # Create new dataframes where daytime and nightime datetimes are
#         # recorded, then merged on a new `Sunlight` column
#         daytime = pd.DataFrame(
#             data="Sunrise", index=sun_df["Sunrise dt"], columns=["Sunlight"]
#         )
#         nighttime = pd.DataFrame(
#             data="Sunset", index=sun_df["Sunset dt"], columns=["Sunlight"]
#         )
#         DayNight = pd.concat([daytime, nighttime], join="outer")
#         DayNight.sort_index(inplace=True)
#         DayNight.index.rename("Datetime", inplace=True)

#         # Create an xarray object from the merged day/night dataframe
#         day_night = xr.Dataset.from_dataframe(DayNight)

#         # Remove local timezone timestamp column in ModTides
#         # dataframe. Xarray doesn't handle timezone aware datetimeindexes
#         # 'from_dataframe' very well.
#         ModTides.index = ModTides.index.tz_localize(tz=None)

#         # Create an xr Dataset from the ModTides pd.dataframe
#         mt = ModTides.to_xarray()

#         # Filter the modelledtides (mt) by the daytime, nighttime
#         # datetimes from the sunriset module.
#         # Modelled tides are designated as either day or night by
#         # propogation of the last valid index value forward
#         Solar = day_night.sel(Datetime=mt.index, method="ffill")

#         # Assign the day and night tideheight datasets
#         SolarDayTides = mt.where(Solar.Sunlight == "Sunrise", drop=True)
#         SolarNightTides = mt.where(Solar.Sunlight == "Sunset", drop=True)

#         # Extract DatetimeIndexes to use in exposure calculations
#         all_timerange_day = pd.DatetimeIndex(SolarDayTides.index)
#         all_timerange_night = pd.DatetimeIndex(SolarNightTides.index)

#         if x == "daylight":
#             return all_timerange_day
#         if x == "night":
#             return all_timerange_night

# def build_expected_grid(time_range_peaks, expected_gap_days=14.75):
#     """
#     Fit a regular grid to detected peaks using the median observed gap
#     and the first detected peak as the anchor point.
#     """
#     # Guard against empty or single-element arrays
#     if len(time_range_peaks) < 2:
#         return pd.DatetimeIndex([])
        
#     median_gap = pd.to_timedelta(
#         np.median((time_range_peaks[1:] - time_range_peaks[:-1]).days), "d"
#     )
#     # Build expected peak times from first detected peak
#     n_cycles = int((time_range_peaks[-1] - time_range_peaks[0]) / median_gap) + 1
#     expected = pd.DatetimeIndex([
#         time_range_peaks[0] + i * median_gap for i in range(n_cycles)
#     ])
#     return expected


# def fill_peak_gaps(
#     time_range_peaks,
#     modelledtides_1d,
#     reference_peaks=None,
#     expected_gap_days=14.75,
#     gap_tolerance=1.5,
#     use_min=False,
# ):

#         # Guard against empty or single-element arrays
#     if len(time_range_peaks) < 2:
#         print(f"Warning: fill_peak_gaps received only {len(time_range_peaks)} peaks — skipping gap fill")
#         return time_range_peaks
        
#     threshold = pd.Timedelta(expected_gap_days * gap_tolerance, "d")
#     gaps = time_range_peaks[1:] - time_range_peaks[:-1]
#     filled_peaks = list(time_range_peaks)

#     # Build an expected grid from the detected peaks themselves
#     expected_grid = build_expected_grid(time_range_peaks, expected_gap_days)

#     for i, gap in enumerate(gaps):
#         if gap > threshold:
#             gap_start = time_range_peaks[i]
#             gap_end = time_range_peaks[i + 1]

#             # Find expected peaks that fall within this gap
#             expected_in_gap = expected_grid[
#                 (expected_grid > gap_start) & (expected_grid < gap_end)
#             ]

#             if len(expected_in_gap) == 0:
#                 print(f"Warning: no expected peaks found in gap {gap_start.date()} → {gap_end.date()}, skipping")
#                 continue

#             # Search around each expected peak position
#             half = pd.Timedelta(expected_gap_days * 0.4, "d")

#             for expected_peak in expected_in_gap:
#                 search_start = expected_peak - half
#                 search_end = expected_peak + half

#                 # Clamp search window to within the actual gap boundaries
#                 search_start = max(search_start, gap_start)
#                 search_end = min(search_end, gap_end)
                
#                 # Skip if window is invalid after clamping
#                 if search_end <= search_start:
#                     print(f"Warning: invalid search window after clamping "
#                           f"{search_start.date()} → {search_end.date()}, skipping")
#                     continue

#                 # Optionally narrow the search window using reference peaks
#                 if reference_peaks is not None:
#                     ref_in_window = reference_peaks[
#                         (reference_peaks > gap_start) & (reference_peaks < gap_end)
#                     ]
#                     if len(ref_in_window) >= 2:
#                         # Narrow to between the bracketing reference peaks
#                         search_start = max(search_start, ref_in_window[0])
#                         search_end = min(search_end, ref_in_window[1])
#                     elif len(ref_in_window) == 1:
#                         # Use reference peak as a tighter anchor
#                         search_start = max(search_start, ref_in_window[0] - half / 2)
#                         search_end = min(search_end, ref_in_window[0] + half / 2)

#                 if search_end <= search_start:
#                     print(f"Warning: invalid search window after reference narrowing "
#                           f"{search_start.date()} → {search_end.date()}, skipping")
#                     continue

#                 window_tides = modelledtides_1d.sel(time=slice(search_start, search_end))

#                 if len(window_tides.time) == 0:
#                     print(f"Warning: no data in search window {search_start.date()} → {search_end.date()}")
#                     continue

#                 best_time = (
#                     window_tides.idxmin(dim="time").values
#                     if use_min
#                     else window_tides.idxmax(dim="time").values
#                 )
#                 filled_peaks.append(pd.Timestamp(best_time))
#                 print(
#                     f"Gap of {gap.days}d between {gap_start.date()} and {gap_end.date()} "
#                     f"— inserted peak at {pd.Timestamp(best_time).date()} "
#                     f"(expected ~{expected_peak.date()})"
#                 )

#     return pd.DatetimeIndex(sorted(filled_peaks))

# def exposure(
#     dem,
#     start_date,
#     end_date,
#     modelled_freq="30min",
#     tide_model="EOT20",
#     tide_model_dir="/var/share/tide_models",
#     filters=None,
#     filters_combined=None,
#     run_id=None,
#     log=None,
#     return_tide_modelling=False,
#     phases =4
# ):
#     """
#     Calculate intertidal exposure, indicating the proportion of time
#     that each pixel was 'exposed' from tidal inundation during the time
#     period of interest.

#     The exposure calculation is based on tide-height differences between
#     the elevation value and modelled tide height percentiles.

#     For an 'unfiltered', all of epoch-time, analysis, exposure is
#     calculated per pixel. All other filter options calculate exposure
#     from high temporal resolution modelled tides that are averaged
#     into a 1D timeseries across the nominated area of interest.

#     This function firstly models high temporal resolution tides across
#     the area of interest. Filtered datetimes and associated tide heights
#     are then extracted from the modelled tides. Exposure is calculated
#     by comparing the quantiled distribution curve of modelled tide
#     heights from the filtered datetime dataset with DEM pixel elevations,
#     returning an exposure percent.

#     Parameters
#     ----------
#     dem : xarray.DataArray
#         xarray.DataArray containing Digital Elevation Model (DEM) data
#         and coordinates and attributes metadata.
#     start_date : str
#         A string containing the start year of the desired analysis period
#         as "YYYY". Note: analysis will start from "YYYY-01-01".
#     end_date  :  str
#         A string containing the end year of the desired analysis period
#         as "YYYY". Note: analysis will end at "YYYY-12-31".
#     modelled_freq : str
#         A pandas time offset alias for the frequency with which to
#         calculate the tide model during exposure calculations. Examples
#         include '30min' for 30 minute cadence or '1h' for a one-hourly
#         cadence. Defaults to '30min'.
#     tide_model : str, optional
#         The tide model or a list of models used to model tides, as
#         supported by the `eo-tides` Python package. Options include:
#         - "EOT20" (default)
#         - "TPXO10-atlas-v2-nc"
#         - "FES2022"
#         - "FES2022_extrapolated"
#         - "FES2014"
#         - "FES2014_extrapolated"
#         - "GOT5.6"
#         - "ensemble" (experimental: combine all above into single ensemble)
#     tide_model_dir : str, optional
#         The directory containing tide model data files. Defaults to
#         "/var/share/tide_models"; for more information about the
#         directory structure, refer to `eo-tides.utils.list_models`.
#     filters : list of strings, optional
#         An optional list of customisation options to input into the tidal
#         modelling to calculate exposure. Filters include:
#         - 'unfiltered': calculates exposure for the full input time period,
#         - 'dry': Southern Hemisphere dry season, defined as April to
#           September
#         - 'wet': Southern Hemisphere wet season, defined as October to
#           March
#         - 'summer', 'autumn', 'winter', 'spring': exposure during
#           specific seasons
#         - 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
#           'oct', 'nov', 'dec': exposure during specific months
#         - 'daylight': all tide heights occurring between sunrise and
#           sunset local time
#         - 'night': all tide heights occurring between sunset and sunrise
#           local time
#         - 'spring_high', high tide exposure during the fortnightly spring tide cycle,
#         - 'spring_low', low tide exposure during the fortnightly spring tide cycle,
#         - 'neap_high', high tide exposure during the fortnightly neap tide cycle,
#         - 'neap_low', low tide exposure during the fortnightly neap tide cycle,
#         - 'hightide', all tide heights greater than or equal to the local lowest high
#         tide heights in high temporal resolution tidal modelling,
#         - 'lowtide' all tide heights lower than or equal to the local highest low tide
#         heights in high temporal resolution tidal modelling,
#         Defaults to ['unfiltered'] if none supplied.
#     filters_combined : list of two-object tuples, optional
#         An optional list of paired customisation options from which to
#         calculate exposure. Filters must be sourced from the list under
#         'filters'. Example: to calculate exposure
#         during daylight hours in the wet season is
#         [('wet', 'daylight')]. Multiple tuple pairs are supported.
#         Defaults to None.
#     run_id : string, optional
#         An optional string giving the name of the analysis; used to
#         prefix log entries.
#     log : logging.Logger, optional
#         Logger object, by default None.
#     return_tide_modelling  :  Boolean
#         When `True`, returns the full epoch tide modelling, as well
#         as filtered tide model datetimes and heights for all filter
#         options. If true, ensure the function call is set to return
#         exposure_ds, modelledtides_ds, modelledtides_1d,timeranges.
#         If false, set the function call to return exposure_ds and
#         modelledtides_ds only. Default = False.

#     Returns
#     -------
#     exposure_ds : xarray.Dataset
#         An xarray.Dataset containing a named exposure variable for each
#         nominated filter, representing the percentage time exposure of
#         each pixel from tidal inundation for the duration of the
#         associated filtered time period between `start` and `end`.
#     modelledtides_ds : dict
#         An xarray.Dataset containing quantiled high temporal resolution
#         tide modelling for each filter. Outputs will have dimensions of
#         either ['quantile', 'x', 'y'] for "unfiltered", or ['quantile']
#         for all other filters.
#     modelledtides_1d  :  xarray.DataArray
#         The 'mean' 1D high temporal resolution tide model for the area of
#         interest. Returned when return_tide_modelling = True.
#     timeranges  :  dict
#         A dictionary of filtered DatetimeIndex's, corresponding to the
#         filtered dates of interest from modelledtides_1d. Returned
#         when return_tide_modelling = True.

#     Notes
#     -----
#     - The tide-height percentiles range from 0 to 100, divided into 101
#     equally spaced values.
#     - The 'diff' variable is calculated as the absolute difference
#     between tide model percentile value and the DEM value at each pixel.
#     - The 'idxmin' variable is the index of the smallest tide-height
#     difference (i.e., maximum similarity) per pixel and is equivalent
#     to the exposure percent.
#     - temporal filters include any of: 'dry', 'wet', 'summer', 'autumn',
#     'winter', 'spring', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
#     'aug', 'sep', 'oct', 'nov', 'dec', 'daylight', 'night'
#     - spatial filters include any of: 'spring_high', 'spring_low',
#     'neap_high', 'neap_low', 'hightide', 'lowtide'

#     """
#     # Set up logs if no log is passed in
#     if log is None:
#         log = configure_logging()

#     # Use run ID name for logs if it exists
#     run_id = "Processing" if run_id is None else run_id

#     # Create the tide-height percentiles from which to calculate
#     # exposure statistics
#     calculate_quantiles = np.linspace(0, 1, 101)

#     # Generate range of times covering entire period of satellite record
#     # for exposure and bias/offset calculation
#     time_range = pd.date_range(
#         start=round_date_strings(start_date, round_type="start"),
#         end=round_date_strings(end_date, round_type="end"),
#         freq=modelled_freq,
#     )

#     # Define the temporal filters
#     temp_filters = [
#         "dry",
#         "wet",
#         "summer",
#         "autumn",
#         "winter",
#         "spring",
#         "jan",
#         "feb",
#         "mar",
#         "apr",
#         "may",
#         "jun",
#         "jul",
#         "aug",
#         "sep",
#         "oct",
#         "nov",
#         "dec",
#         "daylight",
#         "night",
#     ]
#     # Define the spatial filters
#     sptl_filters = [
#         "neaptide",
#         "springtide",
#         "spring_high",
#         "spring_low",
#         "neap_high",
#         "neap_low",
#         "hightide",
#         "lowtide",
#     ]

#     # Create empty xarray.Datasets to store outputs into
#     exposure_ds = xr.Dataset(
#         coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values))
#     )
#     modelledtides_ds = xr.Dataset(
#         coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values))
#     )

#     # Create an empty dict to store temporal `time_range` variables into
#     timeranges = {}

#     # Set filters variable if none supplied
#     if filters is None:
#         filters = ["unfiltered"]

#     # If filter combinations are desired, make sure each filter is
#     # calculated individually for later combination
#     if filters_combined is not None:
#         for x in filters_combined:
#             if str(x[0]) not in filters:
#                 filters.append(str(x[0]))
#             if str(x[1]) not in filters:
#                 filters.append(str(x[1]))
    
#     # Return error for incorrect filter-names
#     all_filters = temp_filters + sptl_filters + ["unfiltered"]

#     for x in filters:
#         assert (
#             x in all_filters
#         ), f'Nominated filter "{x}" is not in {all_filters}. Check spelling and retry'

#     # Run tide model at low resolution
#     modelledtides_lowres = pixel_tides(
#         data=dem,
#         time=time_range,
#         model=tide_model,
#         directory=tide_model_dir,
#         resample=False,
#     )

#     # Calculate a 1D tide height time series to use with filtered exposure calc's
#     modelledtides_1d = modelledtides_lowres.mean(dim=["x", "y"])

#     # Calculate quantiles and reproject low resolution tide data to
#     # pixel resolution if any filter is "unfiltered"
#     if "unfiltered" in filters:

#         # Convert to quantiles, and make sure CRS is present
#         modelledtides_lowres_quantiles = (
#             modelledtides_lowres.quantile(q=calculate_quantiles, dim="time")
#             .astype(modelledtides_lowres.dtype)
#             .odc.assign_crs(dem.odc.geobox.crs)
#         )

#         # Reproject into pixel resolution
#         modelledtides_highres = _pixel_tides_resample(
#             tides_lowres=modelledtides_lowres_quantiles,
#             dask_chunks=dem.shape,
#             gbox=dem.odc.geobox,
#         )

#         # Add pixel resolution tides into to output dataset
#         modelledtides_ds["unfiltered"] = modelledtides_highres

#     # Filter the input timerange to include only dates or tide ranges of
#     # interest if filters is not None:
#     for x in filters:
#         if x in temp_filters:
#             print(f"Filtering timesteps for {x}")
#             timeranges[x] = temporal_filters(x, time_range, dem)
#         #I think this is where a spatial filter calculation goes, returning timeranges[x] only...
#         if x in sptl_filters:
#             print(f"Filtering timesteps for {x}")
#             timeranges[x] = spatial_filters(modelled_freq, 
#                                             x,
#                                             modelledtides_1d,
#                                             modelledtides_lowres,
#                                             phases
#                                            )
    
#     # Intersect the filters of interest to extract the common datetimes for
#     # calculation of combined filters
#     if filters_combined is not None:
#         for x in filters_combined:
#             y = x[0]
#             z = x[1]
#             timeranges[str(y + "_" + z)] = timeranges[y].intersection(timeranges[z])

#     # Intersect datetimes of interest with the 1D tidal model

#     for x in timeranges:
#         # Extract filtered datetimes from the full tidal model
#         modelledtides_x = modelledtides_1d.sel(time=timeranges[str(x)])

#         # Calculate quantile values on remaining tide heights
#         modelledtides_x = (
#             modelledtides_x.quantile(q=calculate_quantiles, dim="time")
#             .to_dataset()
#             .tide_height
#         )

#         # Add modelledtides_x to output dataset
#         modelledtides_ds[str(x)] = modelledtides_x

    

#     # Calculate exposure per filter
#     for x in modelledtides_ds:
#         print(f"Calculating {x} exposure")

#         exposure_ds[str(x)] = exposure_percentiles(modelledtides_ds[str(x)], dem)

#     if return_tide_modelling:
#         return exposure_ds, modelledtides_ds, modelledtides_1d, timeranges
#     else:
#         return exposure_ds, modelledtides_ds

# def exposure_percentiles(modelledtides_ds, dem):
#     # Calculate the tide-height difference between the elevation
#     # value and each percentile value per pixel
#     diff = abs(modelledtides_ds - dem)

#     # Take the percentile of the smallest tide-height difference as
#     # the exposure % per pixel
#     idxmin = diff.idxmin(dim="quantile")

#     # Reorder dimensions
#     if "time" in list(idxmin.dims):
#         idxmin = idxmin.transpose("time", "y", "x")
#     else:
#         idxmin = idxmin.transpose("y", "x")

#     # Convert to percentage and add as variable in exposure dataset
#     exposure = idxmin * 100

#     return exposure

# def spatial_filters(
#     modelled_freq,
#     x,
#     modelledtides_1d,
#     modelledtides_lowres,
#     phases=4
# ):
#     """
#     Identify and extract spatial-specific dates and times to feed
#     into tidal modelling for custom exposure calculations.

#     phases  |  int
#         The number of phases to model each lunar month. Defaults to
#         4, evenly distributing tides across 2 neap and 2 spring tide
#         cycles each month. Alternative: 8, distributing tides across
#         2 spring, neap, crescent and gibbous moons per lunar month. 8
#         phases narrows the window of possible dates upon which a
#         spring or neap tide can be modelled. 4 phases will model all 
#         tides into either a spring or neap phase.
#     """

#     # Extract the modelling freq units
#     freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
#     freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
#     # Extract the number of modelled timesteps per half lunar cycle (where lunar cycle = 29.5 days)
#     mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
#     order = int(mod_timesteps / 2)

#     # ---- SHARED PEAK DETECTION ----
#     # Find all high tide maxima from full timeseries
#     tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
#     tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()

#     # Spring highs: largest maxima in the high tide envelope
#     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
#     springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
#     time_range_springhigh = pd.to_datetime(springpeaks.time)

#     # Neap highs: smallest maxima in the high tide envelope
#     order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
#     neap_peak_idx = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
#     neappeaks = tide_maxima.isel(time=neap_peak_idx)
#     time_range_neaphigh = pd.to_datetime(neappeaks.time)
#     #-------------
#     # Final all low tide minima from full timeseries
#     tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
#     tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

#     # # Spring lows: smallest minima in the low tide envelope
#     # spring_low_idx = argrelmin(tide_minima.tide_height.values, order=order)[0]
#     # modelledtides_1d_peaks_low = tide_minima_idx[spring_low_idx]
#     # springpeaks_low = modelledtides_1d.isel(time=modelledtides_1d_peaks_low).to_dataset()
#     # time_range_springlow = pd.to_datetime(springpeaks_low.time) # rename var

#     # neap_low_idx = argrelmax(tide_minima.tide_height.values, order=order_nh)[0]
#     # neappeaks_low = tide_minima.isel(time=neap_low_idx)
#     # time_range_neap = pd.to_datetime(neappeaks_low.time)

#     # Spring lows: smallest minima in the low tide envelope
#     modelledtides_1d_peakslow = argrelmin(modelledtides_1d.values, order=order)[0]
#     springpeakslow = modelledtides_1d.isel(time=modelledtides_1d_peakslow).to_dataset()
#     time_range_springlow = pd.to_datetime(springpeakslow.time)

#     # Neap lows: largest minima in the low tide envelope
#     order_nl = int(ceil((len(tide_minima.time) / (len(modelledtides_1d_peakslow)) / 2)))
#     neap_peak_idxlow = argrelmax(tide_minima.tide_height.values, order=order_nl)[0]
#     neappeakslow = tide_minima.isel(time=neap_peak_idxlow)
#     time_range_neaplow = pd.to_datetime(neappeakslow.time)

#     print(f"order: {order}, order_nl: {order_nl}")
#     print(f"Spring lows detected: {len(time_range_springlow)}, "
#           f"first: {time_range_springlow[0].date()}, "
#           f"last: {time_range_springlow[-1].date()}")
#     print(f"Gaps between spring lows (days): "
#           f"{(time_range_springlow[1:] - time_range_springlow[:-1]).days.tolist()}")
#     print(f"Neap lows detected: {len(time_range_neaplow)}, "
#           f"first: {time_range_neaplow[0].date()}, "
#           f"last: {time_range_neaplow[-1].date()}")
#     print(f"Gaps between neap lows (days): "
#           f"{(time_range_neaplow[1:] - time_range_neaplow[:-1]).days.tolist()}")
    
#     if x in ["spring_high"]:
#         return time_range_springhigh
#     if x in ["neap_high"]:
#         return time_range_neaphigh
#     if x in ["spring_low"]:
#         return time_range_springlow
#     if x in ["neap_low"]:
#         return time_range_neaplow    

    

#     # ---- END SHARED DETECTION ----
    
#     if x in ["neaptide", "springtide"]:
    
#         expected_cycle = 14.75
#         tolerance = 0.4
#         lower = pd.Timedelta(expected_cycle * (1 - tolerance), "d")
#         upper = pd.Timedelta(expected_cycle * (1 + tolerance), "d")
#         half_window = pd.Timedelta(expected_cycle / 4, "d")
    
#         spring_list = []
#         neap_list = []
    
#         for i in range(len(time_range_neaphigh) - 1):
#             gap = time_range_neaphigh[i + 1] - time_range_neaphigh[i]
    
#             if lower <= gap <= upper:
#                 # Neap period: window around this neap peak
#                 neap_window = modelledtides_1d.sel(
#                     time=slice(
#                         time_range_neaphigh[i] - half_window,
#                         time_range_neaphigh[i] + half_window
#                     )
#                 )
#                 if len(neap_window.time) > 0:
#                     neap_list.append(neap_window)
    
#                 # Spring period: window around midpoint to next neap peak
#                 midpoint = time_range_neaphigh[i] + gap / 2
#                 spring_window = modelledtides_1d.sel(
#                     time=slice(midpoint - half_window, midpoint + half_window)
#                 )
#                 if len(spring_window.time) > 0:
#                     spring_list.append(spring_window)
#             else:
#                 print(f"Skipping pair at {time_range_neaphigh[i].date()} → "
#                       f"{time_range_neaphigh[i+1].date()} — gap of {gap.days}d "
#                       f"outside expected range")
    
#         # Handle the last neap peak if it has a valid predecessor
#         if len(time_range_neaphigh) > 1:
#             last_gap = time_range_neaphigh[-1] - time_range_neaphigh[-2]
#             if lower <= last_gap <= upper:
#                 neap_window = modelledtides_1d.sel(
#                     time=slice(
#                         time_range_neaphigh[-1] - half_window,
#                         time_range_neaphigh[-1] + half_window
#                     )
#                 )
#                 if len(neap_window.time) > 0:
#                     neap_list.append(neap_window)
    
#         if x in ['springtide']:
#             springtide = xr.concat(spring_list, dim="time")
#             return pd.to_datetime(springtide.time)
#         if x in ['neaptide']:
#             neaptide = xr.concat(neap_list, dim="time")
#             return pd.to_datetime(neaptide.time)

#     # ## Block 1: neaptide / springtide
#     # if x in ["neaptide", "springtide"]:

#     #     # Fill gaps using observed median gap as expected spacing
#     #     actual_spring_gap = np.median(
#     #         (time_range_spring[1:] - time_range_spring[:-1]).days
#     #     )
#     #     actual_neap_gap = np.median(
#     #         (time_range_neap[1:] - time_range_neap[:-1]).days
#     #     )
#     #     time_range_spring = fill_peak_gaps(
#     #         time_range_spring,
#     #         modelledtides_1d,
#     #         reference_peaks=time_range_neap,
#     #         expected_gap_days=actual_spring_gap,
#     #         use_min=False,
#     #     )
#     #     time_range_neap = fill_peak_gaps(
#     #         time_range_neap,
#     #         modelledtides_1d,
#     #         reference_peaks=time_range_spring,
#     #         expected_gap_days=actual_neap_gap,
#     #         use_min=False,
#     #     )

#     #     # Fixed window of +/- half lunar cycle around each detected peak
#     #     half_cycle = pd.Timedelta(14.75 / 2, "d")

#     #     spring_list = []
#     #     for peak in time_range_spring:
#     #         window = modelledtides_1d.sel(
#     #             time=slice(peak - half_cycle, peak + half_cycle)
#     #         )
#     #         if len(window.time) > 0:
#     #             spring_list.append(window)

#     #     neap_list = []
#     #     for peak in time_range_neap:
#     #         window = modelledtides_1d.sel(
#     #             time=slice(peak - half_cycle, peak + half_cycle)
#     #         )
#     #         if len(window.time) > 0:
#     #             neap_list.append(window)

#     #     if x in ['springtide']:
#     #         springtide = xr.concat(spring_list, dim="time")
#     #         return pd.to_datetime(springtide.time)
#     #     if x in ['neaptide']:
#     #         neaptide = xr.concat(neap_list, dim="time")
#     #         return pd.to_datetime(neaptide.time)

#     # ## Block 2: spring_high / spring_low / neap_high / neap_low
#     # if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

#     #     # For low tide filters, recompute peaks on troughs
#     #     if x in ["spring_low", "neap_low"]:
#     #         tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
#     #         tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

#     #         spring_low_idx = argrelmin(tide_minima.tide_height.values, order=order)[0]
#     #         modelledtides_1d_peaks_low = tide_minima_idx[spring_low_idx]
#     #         springpeaks_low = modelledtides_1d.isel(
#     #             time=modelledtides_1d_peaks_low
#     #         ).to_dataset()
#     #         time_range_spring = pd.to_datetime(springpeaks_low.time)

#     #         neap_low_idx = argrelmax(tide_minima.tide_height.values, order=order_nh)[0]
#     #         neappeaks_low = tide_minima.isel(time=neap_low_idx)
#     #         time_range_neap = pd.to_datetime(neappeaks_low.time)

#     #     if x in ["spring_high", "spring_low"]:
#     #         return time_range_spring
#     #     if x in ["neap_high", "neap_low"]:
#     #         return time_range_neap

# # def spatial_filters(
# #     modelled_freq,
# #     x,
# #     modelledtides_1d,
# #     modelledtides_lowres,
# #     phases=4
# # ):
# #     """
# #     Identify and extract spatial-specific dates and times to feed
# #     into tidal modelling for custom exposure calculations.

# #     phases  |  int
# #         The number of phases to model each lunar month. Defaults to
# #         4, evenly distributing tides across 2 neap and 2 spring tide
# #         cycles each month. Alternative: 8, distributing tides across
# #         2 spring, neap, crescent and gibbous moons per lunar month. 8
# #         phases narrows the window of possible dates upon which a
# #         spring or neap tide can be modelled. 4 phases will model all 
# #         tides into either a spring or neap phase.
# #     """

# #     # Extract the modelling freq units
# #     freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
# #     freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
# #     # Extract the number of modelled timesteps per half lunar cycle (29.5 days)
# #     mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
# #     order = int(mod_timesteps / 2)

# #     # ---- SHARED PEAK DETECTION ----
# #    # Find all high tide maxima
# #     tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
# #     tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()
    
# #     # Spring highs: largest maxima in the high tide envelope
# #     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
# #     springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# #     time_range_spring = pd.to_datetime(springpeaks.time)
    
# #     # Neap highs: smallest maxima in the high tide envelope
# #     order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# #     neap_peak_idx = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
# #     neappeaks = tide_maxima.isel(time=neap_peak_idx)
# #     time_range_neap = pd.to_datetime(neappeaks.time)
    
# #     print(f"order: {order}, order_nh: {order_nh}")
# #     print(f"tide_maxima length: {len(tide_maxima.time)}, "
# #           f"spring peaks: {len(modelledtides_1d_peaks)}")
# #     print(f"Spring peaks detected: {len(time_range_spring)}, "
# #           f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
# #           f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
# #     print(f"Neap peaks detected: {len(time_range_neap)}, "
# #           f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
# #           f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
# #     # ---- END SHARED DETECTION ----

# #     ## Block 1: neaptide / springtide
# #     if x in ["neaptide", "springtide"]:

# #         # Use observed median gap for fill_peak_gaps
# #         actual_spring_gap = np.median(
# #             (time_range_spring[1:] - time_range_spring[:-1]).days
# #         )
# #         actual_neap_gap = np.median(
# #             (time_range_neap[1:] - time_range_neap[:-1]).days
# #         )

# #         # Fill gaps in both lists using expected grid as primary reference
# #         # and each other as loose secondary constraint
# #         time_range_spring = fill_peak_gaps(
# #             time_range_spring,
# #             modelledtides_1d,
# #             reference_peaks=time_range_neap,
# #             expected_gap_days=actual_spring_gap,
# #             use_min=False,
# #         )
# #         time_range_neap = fill_peak_gaps(
# #             time_range_neap,
# #             modelledtides_1d,
# #             reference_peaks=time_range_spring,
# #             expected_gap_days=actual_neap_gap,
# #             use_min=False,
# #         )

# #         # Identifying quartile ranges between neap and spring highs
# #         idx1 = time_range_spring
# #         idx2 = time_range_neap

# #         if idx2[0] < idx1[0]:
# #             idx2 = time_range_neap[1:]

# #         if len(idx1) != len(idx2):
# #             min_len = min(len(idx1), len(idx2))
# #             idx1 = idx1[:min_len]
# #             idx2 = idx2[:min_len]

# #         # Interleave idx1 and idx2
# #         interleaved = pd.DatetimeIndex(np.ravel(np.column_stack([idx1, idx2])))

# #         # Calculate quartile boundaries between consecutive elements
# #         delta = interleaved[1:] - interleaved[:-1]
# #         q1 = interleaved[:-1] + delta * 0.25
# #         q2 = interleaved[:-1] + delta * 0.50
# #         q3 = interleaved[:-1] + delta * 0.75

# #         spring_list = []
# #         neap_list = []

# #         if phases == 4:
# #             for i in range(0, len(q2) - 1, 2):
# #                 neap_list.append(
# #                     modelledtides_1d.sel(time=slice(q2[i], q2[i + 1]))
# #                 )
# #                 if i + 2 < len(q2):
# #                     spring_list.append(
# #                         modelledtides_1d.sel(time=slice(q2[i + 1], q2[i + 2]))
# #                     )

# #         if phases == 8:
# #             for i in range(0, len(q3) - 1, 2):
# #                 neap_list.append(
# #                     modelledtides_1d.sel(time=slice(q3[i], q1[i + 1]))
# #                 )
# #                 if i + 2 < len(q3):
# #                     spring_list.append(
# #                         modelledtides_1d.sel(time=slice(q3[i + 1], q1[i + 2]))
# #                     )

# #         if x in ['springtide']:
# #             springtide = xr.concat(spring_list, dim="time")
# #             return pd.to_datetime(springtide.time)
# #         if x in ['neaptide']:
# #             neaptide = xr.concat(neap_list, dim="time")
# #             return pd.to_datetime(neaptide.time)

# #     ## Block 2: spring_high / spring_low / neap_high / neap_low
# #     if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

# #         # For low tide filters, recompute peaks on troughs
# #         if x in ["spring_low", "neap_low"]:
# #             tide_minima_idx = argrelmin(modelledtides_1d.values)[0]
# #             tide_minima = modelledtides_1d.isel(time=tide_minima_idx).to_dataset()

# #             spring_low_idx = argrelmin(tide_minima.tide_height.values, order=order_envelope)[0]
# #             modelledtides_1d_peaks_low = tide_minima_idx[spring_low_idx]
# #             springpeaks_low = modelledtides_1d.isel(time=modelledtides_1d_peaks_low).to_dataset()
# #             time_range_spring = pd.to_datetime(springpeaks_low.time)

# #             neap_low_idx = argrelmax(tide_minima.tide_height.values, order=order_envelope)[0]
# #             neappeaks_low = tide_minima.isel(time=neap_low_idx)
# #             time_range_neap = pd.to_datetime(neappeaks_low.time)

# #         actual_spring_gap = np.median(
# #             (time_range_spring[1:] - time_range_spring[:-1]).days
# #         )
# #         actual_neap_gap = np.median(
# #             (time_range_neap[1:] - time_range_neap[:-1]).days
# #         )

# #         # Fill both lists using expected grid as primary, each other as secondary
# #         time_range_spring_filled = fill_peak_gaps(
# #             time_range_spring,
# #             modelledtides_1d,
# #             reference_peaks=time_range_neap,
# #             expected_gap_days=actual_spring_gap,
# #             use_min=(x == "spring_low"),
# #         )
# #         time_range_neap_filled = fill_peak_gaps(
# #             time_range_neap,
# #             modelledtides_1d,
# #             reference_peaks=time_range_spring_filled,
# #             expected_gap_days=actual_neap_gap,
# #             use_min=(x == "neap_low"),
# #         )

# #         if x in ["spring_high", "spring_low"]:
# #             return time_range_spring_filled
# #         if x in ["neap_high", "neap_low"]:
# #             return time_range_neap_filled

# # # def fill_peak_gaps(time_range_peaks, 
# # #                    modelledtides_1d, 
# # #                    expected_gap_days=14.75, 
# # #                    gap_tolerance=1.3,
# # #                    use_min=False
# # #                   ):
# # #     """
# # #     Identify gaps between consecutive peaks that are significantly larger
# # #     than expected (~14.75 days for spring/neap cycles), then search for
# # #     the best candidate peak within each gap.

# # #     Parameters
# # #     ----------
# # #     time_range_peaks : pd.DatetimeIndex
# # #         Detected spring_high or neap_high peak datetimes.
# # #     modelledtides_1d : xr.DataArray
# # #         1D tide height timeseries.
# # #     expected_gap_days : float
# # #         Expected gap between consecutive peaks in days. Default 14.75
# # #         (half lunar cycle).
# # #     gap_tolerance : float
# # #         Multiplier of expected_gap_days above which a gap is considered
# # #         anomalous and worth searching. Default 1.5 (i.e. gaps > ~22 days).

# # #     Returns
# # #     -------
# # #     pd.DatetimeIndex
# # #         Filled peak datetimes, sorted.
# # #     """
# # #     threshold = pd.Timedelta(expected_gap_days * gap_tolerance, "d")
# # #     gaps = time_range_peaks[1:] - time_range_peaks[:-1]

# # #     filled_peaks = list(time_range_peaks)

# # #     for i, gap in enumerate(gaps):
# # #         if gap > threshold:
# # #             # Define search window: centre on the expected midpoint of the gap,
# # #             # +/- half the expected cycle length
# # #             gap_start = time_range_peaks[i]
# # #             gap_end   = time_range_peaks[i + 1]
            
# # #             window_centre = gap_start + (gap_end - gap_start) / 2
# # #             half_window   = pd.Timedelta(expected_gap_days / 2, "d")
            
# # #             search_start = window_centre - half_window
# # #             search_end   = window_centre + half_window

# # #             # Extract tide heights within the search window
# # #             window_tides = modelledtides_1d.sel(
# # #                 time=slice(search_start, search_end)
# # #             )

# # #             if len(window_tides.time) == 0:
# # #                 print(f"Warning: no data found in gap window {search_start} → {search_end}")
# # #                 continue

# # #             # Find the peak (max for spring_high/neap_high; swap to .idxmin for lows)
# # #             best_time = window_tides.idxmin(dim="time").values if use_min else window_tides.idxmax(dim="time").values

# # #             filled_peaks.append(pd.Timestamp(best_time))
# # #             print(f"Gap of {gap.days}d detected between {gap_start.date()} and "
# # #                   f"{gap_end.date()} — inserting peak at {pd.Timestamp(best_time).date()}")

# # #     return pd.DatetimeIndex(sorted(filled_peaks))

# # # def spatial_filters(
# # #     modelled_freq,
# # #     x,
# # #     modelledtides_1d,
# # #     modelledtides_lowres,
# # #     phases=4
# # #     # timeranges,
# # #     # calculate_quantiles,
# # #     # modelledtides_ds,
# # #     # dem,
# # #     # exposure,
# # # ):
# # #     """
# # #     Identify and extract spatial-specific dates and times to feed
# # #     into tidal modelling for custom exposure calculations.

# # #     phases  |  int
# # #         The number of phases to model each lunar month. Defaults to
# # #         4, evenly distributing tides across 2 neap and 2 spring tide
# # #         cycles each month. Alternative: 8, distributing tides across
# # #         2 spring, neap, crescent and gibbous moons per lunar month. 8
# # #         phases narrows the window of possible dates upon which a
# # #         spring or neap tide can be modelled. 4 phases will model all 
# # #         tides into either a spring or neap phase.
# # #     """

# # #     # Extract the modelling freq units
# # #     # Split the number and text characters in modelled_freq
# # #     freq_time = int(re.findall(r"(\d+)(\w+)", modelled_freq)[0][0])
# # #     freq_unit = str(re.findall(r"(\d+)(\w+)", modelled_freq)[0][-1])
# # #     # Extract the number of modelled timesteps per half lunar cycle (29.5 days) for neap/spring calcs
# # #     mod_timesteps = pd.Timedelta((29.5 / 2), "d") / pd.Timedelta(freq_time, freq_unit)
# # #     ## Identify kwargs for peak detection algorithm
# # #     order = int(mod_timesteps / 2)
# # #     print (f'order: {order}')

# # #     # Tidal regime detection
# # #     high_peaks_all = argrelmax(modelledtides_1d.values)[0]
# # #     high_peaks_envelope = modelledtides_1d.isel(time=high_peaks_all)
# # #     lower_highs = argrelmin(high_peaks_envelope.values)[0]
# # #     is_diurnal = len(lower_highs) / len(high_peaks_all) < 0.2
# # #     print(f"Tidal regime: {'diurnal' if is_diurnal else 'semi-diurnal'} "
# # #           f"(lower_highs ratio: {len(lower_highs) / len(high_peaks_all):.2f})")

# # #     # Find all high tide maxima from full timeseries
# # #     tide_maxima_idx = argrelmax(modelledtides_1d.values)[0]
# # #     tide_maxima = modelledtides_1d.isel(time=tide_maxima_idx).to_dataset()
    
# # #     # Recalculate order for the envelope specifically
# # #     # Envelope has ~2 points per day (semi-diurnal), so half lunar cycle in envelope points:
# # #     envelope_timesteps = pd.Timedelta(29.5 / 2, "d") / pd.Timedelta(freq_time, freq_unit)
# # #     # Adjust order for the fact that envelope has ~2x fewer points than full timeseries
# # #     order_envelope = max(1, int(envelope_timesteps / 2 / 2))
    
# # #     print(f"order (full timeseries): {order}")
# # #     print(f"order_envelope: {order_envelope}")
# # #     print(f"tide_maxima length: {len(tide_maxima.time)}")
    
# # #     # Spring highs: largest maxima in envelope
# # #     spring_peak_idx = argrelmax(tide_maxima.tide_height.values, order=order_envelope)[0]
# # #     modelledtides_1d_peaks = tide_maxima_idx[spring_peak_idx]
# # #     springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# # #     time_range_spring = pd.to_datetime(springpeaks.time)
    
# # #     # Neap highs: smallest maxima in envelope, same order
# # #     order_nh = order_envelope
# # #     neap_peak_idx = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
# # #     neappeaks = tide_maxima.isel(time=neap_peak_idx)
# # #     time_range_neap = pd.to_datetime(neappeaks.time)
    
# # #     print(f"Spring peaks detected: {len(time_range_spring)}, "
# # #           f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
# # #           f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
# # #     print(f"Neap peaks detected: {len(time_range_neap)}, "
# # #           f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
# # #           f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
# # #     # # spring peak detection
# # #     # if is_diurnal:
# # #     #     print("Diurnal regime — using broad argrelmax for spring peaks")
# # #     #     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
# # #     # else:
# # #     #     print("Semi-diurnal regime — using argrelmax on high tide envelope for spring peaks")
# # #     #     envelope_peak_idx = argrelmax(high_peaks_envelope.values, order=order)[0]
# # #     #     modelledtides_1d_peaks = high_peaks_all[envelope_peak_idx]

# # #     # springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# # #     # time_range_spring = pd.to_datetime(springpeaks.time)

# # #     # # neap peak detection
# # #     # tide_maxima = argrelmax(modelledtides_1d.values)[0]
# # #     # tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #     # order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# # #     # neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# # #     # # if is_diurnal:
# # #     # #     print("Diurnal regime — using argrelmin with broader order for neap peaks")
# # #     # #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=max(1, order_nh * 2))[0]
# # #     # # else:
# # #     # #     print("Semi-diurnal regime — excluding spring peaks from high tide envelope for neap peaks")
# # #     # #     all_hightide_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nh)[0]
# # #     # #     spring_times = pd.to_datetime(springpeaks.time)
# # #     # #     all_peak_times = pd.to_datetime(tide_maxima.isel(time=all_hightide_peaks).time)
# # #     # #     half_cycle = pd.Timedelta(14.75 / 2, "d")
# # #     # #     neap_mask = np.array([
# # #     # #         not any(abs(t - s) < half_cycle for s in spring_times)
# # #     # #         for t in all_peak_times
# # #     # #     ])
# # #     # #     neap_peaks = all_hightide_peaks[neap_mask]

# # #     # neappeaks = tide_maxima.isel(time=neap_peaks)
# # #     # time_range_neap = pd.to_datetime(neappeaks.time)


# # #     ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
# # #     if x in ["neaptide", "springtide"]:#"spring_high", "spring_low", "neap_high", "neap_low"]:

# # #         # # 1D tide modelling workflow
# # #         # # apply the peak detection routine
# # #         # # if x in ["spring_high", "neap_high"]:
# # #         # modelledtides_1d_peaks = argrelmax(
# # #         #     modelledtides_1d.values, order=order
# # #         # )[0]
# # #         # # if x in ["spring_low", "neap_low"]:
# # #         # #     modelledtides_1d_peaks = argrelmin(
# # #         # #         modelledtides_1d.values, order=order
# # #         # #     )[0]
# # #         # # if x == "neap_high":
# # #         # ## apply the peak detection routine to calculate all the high tide maxima
# # #         # tide_maxima = argrelmax(modelledtides_1d.values)[0]
# # #         # tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         # ## extract neap high tides based on a half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# # #         # order_nh = int(
# # #         #     ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# # #         # )
# # #         # ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
# # #         # neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# # #         # # if x == "neap_low":
# # #         # #     ## apply the peak detection routine to calculate all the low tide maxima
# # #         # #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# # #         # #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         # #     ## extract neap low tides based on 14 day half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# # #         # #     order_nl = int(
# # #         # #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# # #         # #     )
# # #         # #     ## apply the peak detection routine to calculate all the neap low tide maxima within the low tide peaks
# # #         # #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
# # #         # #     # neap_peaks = argrelmax(tide_maxima.values, order=order_nl)[0]

        
# # #         # # if x in ["neap_high", "neap_low"]:
# # #         # ## extract neap high tides
# # #         # neappeaks = tide_maxima.isel(time=neap_peaks)
# # #         # time_range_neap = pd.to_datetime(neappeaks.time)
# # #         # # time_range_neap = fill_peak_gaps(time_range_neap, modelledtides_1d, use_min=False)
# # #         #     # return time_range
# # #         #     # Extract the peak height dates
# # #         #     # tide_cq = neappeaks.quantile(q=calculate_quantiles, dim="time")

# # #         # # if x in ["spring_high", "spring_low"]:
# # #         # # select for indices associated with peaks
# # #         # springpeaks = modelledtides_1d.isel(
# # #         #     time=modelledtides_1d_peaks
# # #         # ).to_dataset()
# # #         # # Save datetimes for calculation of combined filter exposure
# # #         # time_range_spring = pd.to_datetime(springpeaks.time)
# # #         # # time_range_spring = fill_peak_gaps(time_range_spring, modelledtides_1d, use_min=False)
# # #         #     # return time_range
# # #         #     # Extract the peak height dates
# # #         #     # tide_cq = springpeaks.quantile(q=calculate_quantiles, dim="time")

# # #         # # Fill gaps in both lists using expected grid as primary reference
# # #         # # and each other as secondary reference only after both are initially filled

# # #         print(f"Spring peaks detected: {len(time_range_spring)}, "
# # #               f"first: {time_range_spring[0].date() if len(time_range_spring) > 0 else 'none'}, "
# # #               f"last: {time_range_spring[-1].date() if len(time_range_spring) > 0 else 'none'}")
# # #         print(f"Neap peaks detected: {len(time_range_neap)}, "
# # #               f"first: {time_range_neap[0].date() if len(time_range_neap) > 0 else 'none'}, "
# # #               f"last: {time_range_neap[-1].date() if len(time_range_neap) > 0 else 'none'}")
        
# # #         time_range_spring = fill_peak_gaps(
# # #             time_range_spring,
# # #             modelledtides_1d,
# # #             reference_peaks=time_range_neap,  # raw neap as loose secondary constraint only
# # #             use_min=False,
# # #         )
# # #         time_range_neap = fill_peak_gaps(
# # #             time_range_neap,
# # #             modelledtides_1d,
# # #             reference_peaks=time_range_spring,  # filled spring as secondary constraint
# # #             use_min=False,
# # #         )

        
# # #         # Identifying quartile ranges between neap and spring highs
# # #         idx1 = time_range_spring
# # #         idx2 = time_range_neap
        
# # #         if idx2[0] < idx1[0]:
# # #             idx2=time_range_neap[1:]
        
# # #         # Ensure peak lists are the same length
# # #         if len(idx1) != len(idx2):
# # #             min_len = min(len(idx1), len(idx2))
# # #             idx1 = idx1[:min_len]
# # #             idx2 = idx2[:min_len]
        
# # #         # Interleave idx1 and idx2
# # #         interleaved = pd.DatetimeIndex(np.ravel(np.column_stack([idx1, idx2])))
        
# # #         # Calculate quartile boundaries between consecutive elements
# # #         delta = interleaved[1:] - interleaved[:-1]
# # #         q1 = interleaved[:-1] + delta * 0.25
# # #         q2 = interleaved[:-1] + delta * 0.50  # midpoints (same as before)
# # #         q3 = interleaved[:-1] + delta * 0.75

# # #         spring_list = []
# # #         neap_list = []

# # #         if phases == 4:
# # #             # Split modelledtides_1d collecting midpoint (q2) tide heights for neap/spring high tide periods           
# # #             for i in range(0, len(q2) - 1, 2):
# # #                 neap_list.append(
# # #                     modelledtides_1d.sel(time=slice(q2[i], q2[i + 1]))
# # #                 )
# # #                 if i + 2 < len(q2):
# # #                     spring_list.append(
# # #                         modelledtides_1d.sel(time=slice(q2[i + 1], q2[i + 2]))
# # #                     )

# # #         if phases == 8:
# # #             # Split modelledtides_1d collecting q3 to q1 tide heights for neap/spring high tide periods            
# # #             for i in range(0, len(q3) - 1, 2):
# # #                 neap_list.append(
# # #                     modelledtides_1d.sel(time=slice(q3[i], q1[i + 1]))
# # #                 )
# # #                 if i + 2 < len(q3):
# # #                     spring_list.append(
# # #                         modelledtides_1d.sel(time=slice(q3[i + 1], q1[i + 2]))
# # #                     )

# # #         if x in ['springtide']:
# # #             springtide = xr.concat(spring_list, dim="time")
# # #             springtide = pd.to_datetime(springtide.time)
# # #             return springtide
# # #         if x in ['neaptide']:
# # #             neaptide = xr.concat(neap_list, dim="time")
# # #             neaptide = pd.to_datetime(neaptide.time)
# # #             return neaptide


# # #     ## Calculate the spring highest and spring lowest tides per 14 day half lunar cycle
# # #     if x in ["spring_high", "spring_low", "neap_high", "neap_low"]:

# # #         # For low tide filters, recompute peaks on troughs
# # #         if x in ["spring_low", "neap_low"]:
# # #             modelledtides_1d_peaks_low = argrelmin(modelledtides_1d.values, order=order)[0]
# # #             springpeaks_low = modelledtides_1d.isel(time=modelledtides_1d_peaks_low).to_dataset()
# # #             time_range_spring = pd.to_datetime(springpeaks_low.time)

# # #             tide_minima = argrelmin(modelledtides_1d.values)[0]
# # #             tide_minima = modelledtides_1d.isel(time=tide_minima).to_dataset()
# # #             order_nl = int(ceil((len(tide_minima.time) / (len(modelledtides_1d_peaks_low)) / 2)))
# # #             neap_peaks_low = argrelmax(tide_minima.tide_height.values, order=order_nl)[0]
# # #             neappeaks_low = tide_minima.isel(time=neap_peaks_low)
# # #             time_range_neap = pd.to_datetime(neappeaks_low.time)

# # #         # Fill both lists
# # #         time_range_spring_filled = fill_peak_gaps(
# # #             time_range_spring,
# # #             modelledtides_1d,
# # #             reference_peaks=time_range_neap,
# # #             use_min=(x == "spring_low"),
# # #         )
# # #         time_range_neap_filled = fill_peak_gaps(
# # #             time_range_neap,
# # #             modelledtides_1d,
# # #             reference_peaks=time_range_spring_filled,
# # #             use_min=(x == "neap_low"),
# # #         )

# # #         if x in ["spring_high", "spring_low"]:
# # #             return time_range_spring_filled
# # #         if x in ["neap_high", "neap_low"]:
# # #             return time_range_neap_filled


# # #         #         # Compute both raw peak lists before filling either
# # #         # if x in ["spring_high", "neap_high"]:
# # #         #     modelledtides_1d_peaks = argrelmax(modelledtides_1d.values, order=order)[0]
# # #         # if x in ["spring_low", "neap_low"]:
# # #         #     modelledtides_1d_peaks = argrelmin(modelledtides_1d.values, order=order)[0]
        
# # #         # # Always compute both spring and neap raw lists regardless of x
# # #         # springpeaks = modelledtides_1d.isel(time=modelledtides_1d_peaks).to_dataset()
# # #         # time_range_spring = pd.to_datetime(springpeaks.time)
        
# # #         # if x in ["neap_high", "spring_high"]:
# # #         #     tide_maxima = argrelmax(modelledtides_1d.values)[0]
# # #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         #     order_nh = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# # #         #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]
# # #         # if x in ["neap_low", "spring_low"]:
# # #         #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# # #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         #     order_nl = int(ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2)))
# # #         #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
        
# # #         # neappeaks = tide_maxima.isel(time=neap_peaks)
# # #         # time_range_neap = pd.to_datetime(neappeaks.time)
        
# # #         # # Fill both lists using expected grid as primary, each other as secondary
# # #         # time_range_spring_filled = fill_peak_gaps(
# # #         #     time_range_spring,
# # #         #     modelledtides_1d,
# # #         #     reference_peaks=time_range_neap,  # raw neap as loose secondary constraint
# # #         #     use_min=(x == "spring_low"),
# # #         # )
# # #         # time_range_neap_filled = fill_peak_gaps(
# # #         #     time_range_neap,
# # #         #     modelledtides_1d,
# # #         #     reference_peaks=time_range_spring_filled,  # filled spring as secondary constraint
# # #         #     use_min=(x == "neap_low"),
# # #         # )
        
# # #         # if x in ["spring_high", "spring_low"]:
# # #         #     return time_range_spring_filled
# # #         # if x in ["neap_high", "neap_low"]:
# # #         #     return time_range_neap_filled

# # #         # # 1D tide modelling workflow
# # #         # # apply the peak detection routine
# # #         # if x in ["spring_high", "neap_high"]:
# # #         #     modelledtides_1d_peaks = argrelmax(
# # #         #         modelledtides_1d.values, order=order
# # #         #     )[0]
# # #         # if x in ["spring_low", "neap_low"]:
# # #         #     modelledtides_1d_peaks = argrelmin(
# # #         #         modelledtides_1d.values, order=order
# # #         #     )[0]
# # #         # if x == "neap_high":
# # #         #     ## apply the peak detection routine to calculate all the high tide maxima
# # #         #     tide_maxima = argrelmax(modelledtides_1d.values)[0]
# # #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         #     ## extract neap high tides based on a half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# # #         #     order_nh = int(
# # #         #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# # #         #     )
# # #         #     ## apply the peak detection routine to calculate all the neap high tide minima within the high tide peaks
# # #         #     neap_peaks = argrelmin(tide_maxima.tide_height.values, order=order_nh)[0]

# # #         # if x == "neap_low":
# # #         #     ## apply the peak detection routine to calculate all the low tide maxima
# # #         #     tide_maxima = argrelmin(modelledtides_1d.values)[0]
# # #         #     tide_maxima = modelledtides_1d.isel(time=tide_maxima).to_dataset()
# # #         #     ## extract neap low tides based on 14 day half lunar cycle - determined as the fraction of all high tide points relative to the number of spring high tide values
# # #         #     order_nl = int(
# # #         #         ceil((len(tide_maxima.time) / (len(modelledtides_1d_peaks)) / 2))
# # #         #     )
# # #         #     ## apply the peak detection routine to calculate all the neap low tide maxima within the low tide peaks
# # #         #     neap_peaks = argrelmax(tide_maxima.tide_height.values, order=order_nl)[0]
# # #         #     # neap_peaks = argrelmax(tide_maxima.values, order=order_nl)[0]

        
# # #         # if x in ["neap_high", "neap_low"]:
# # #         #     ## extract neap high tides
# # #         #     neappeaks = tide_maxima.isel(time=neap_peaks)
# # #         #     time_range = pd.to_datetime(neappeaks.time)
# # #         #     time_range = fill_peak_gaps(time_range, modelledtides_1d, use_min=(x == "neap_low"))
# # #         #     return time_range
# # #         #     # Extract the peak height dates
# # #         #     # tide_cq = neappeaks.quantile(q=calculate_quantiles, dim="time")

# # #         # if x in ["spring_high", "spring_low"]:
# # #         #     # select for indices associated with peaks
# # #         #     springpeaks = modelledtides_1d.isel(
# # #         #         time=modelledtides_1d_peaks
# # #         #     ).to_dataset()
# # #         #     # Save datetimes for calculation of combined filter exposure
# # #         #     time_range = pd.to_datetime(springpeaks.time)
# # #         #     time_range = fill_peak_gaps(time_range, modelledtides_1d, use_min=(x == "spring_low"))
# # #         #     return time_range
# # #         #     # Extract the peak height dates
# # #         #     # tide_cq = springpeaks.quantile(q=calculate_quantiles, dim="time")






# # #     # if x == "hightide":
# # #     #     # calculate all the high tide maxima
# # #     #     high_peaks = argrelmax(modelledtides_1d.values)[0]
# # #     #     # extract all hightide peaks
# # #     #     high_peaks2 = modelledtides_1d.isel(time=high_peaks)
# # #     #     # identify all lower hightide peaks
# # #     #     lowhigh_peaks = argrelmin(high_peaks2.values)[0]
# # #     #     # extract all lower hightide peaks
# # #     #     lowhigh_peaks2 = high_peaks2.isel(time=lowhigh_peaks)

# # #     #     # Test for diurnal tidal regimes on the assumption that semi-diurnal and mixed tidal settings
# # #     #     # should have approximately equal proportions of daytime and nighttime hightide peaks
# # #     #     if len(lowhigh_peaks) / len(high_peaks) < 0.2:
# # #     #         # timeranges[str(x)] = pd.to_datetime(high_peaks2.time)
# # #     #         return pd.to_datetime(high_peaks2.time)
# # #     #         # tide_cq = high_peaks2.quantile(
# # #     #         #     q=calculate_quantiles, dim="time"
# # #     #         # ).to_dataset()
# # #     #     else:
# # #     #         # interpolate the lower hightide curve
# # #     #         low_high_linear = interp(
# # #     #             np.arange(0, len(modelledtides_1d)),
# # #     #             high_peaks[lowhigh_peaks],
# # #     #             lowhigh_peaks2.values,
# # #     #         )
# # #     #         # Extract all tides higher than/equal to the extrapolated lowest high tide line
# # #     #         hightide = modelledtides_1d.where(
# # #     #             modelledtides_1d >= low_high_linear, drop=True
# # #     #         )
# # #     #         ## Save datetimes for calculation of combined filter exposure
# # #     #         time_range = pd.to_datetime(hightide.time)
# # #     #         return time_range
# # #     #         # tide_cq = hightide.quantile(q=calculate_quantiles, dim="time").to_dataset()

# # #     # if x == "lowtide":
# # #     #     # calculate all the low tide maxima
# # #     #     low_peaks = argrelmin(modelledtides_1d.values)[0]
# # #     #     # extract all lowtide peaks
# # #     #     low_peaks2 = modelledtides_1d.isel(time=low_peaks)
# # #     #     # identify all higher lowtide peaks
# # #     #     highlow_peaks = argrelmax(low_peaks2.values)[0]
# # #     #     # extract all higher lowtide peaks
# # #     #     highlow_peaks2 = low_peaks2.isel(time=highlow_peaks)

# # #     #     # Test for diurnal tidal regimes on the assumption that semi-diurnal and mixed tidal settings
# # #     #     # should have approximately equal proportions of daytime and nighttime lowtide peaks
# # #     #     if len(highlow_peaks) / len(low_peaks) < 0.2:
# # #     #         # timeranges[str(x)] = pd.to_datetime(low_peaks2.time)
# # #     #         return pd.to_datetime(low_peaks2.time)
# # #     #         # tide_cq = low_peaks2.quantile(
# # #     #         #     q=calculate_quantiles, dim="time"
# # #     #         # ).to_dataset()
# # #     #     else:
# # #     #         # interpolate the higher lowtide curve
# # #     #         high_low_linear = interp(
# # #     #             np.arange(0, len(modelledtides_1d)),
# # #     #             low_peaks[highlow_peaks],
# # #     #             highlow_peaks2.values,
# # #     #         )
# # #     #         # Extract all tides lower than/equal to the extrapolated higher lowtide line
# # #     #         lowtide = modelledtides_1d.where(
# # #     #             modelledtides_1d <= high_low_linear, drop=True
# # #     #         )
# # #     #         ## Save datetimes for calculation of combined filter exposure
# # #     #         time_range = pd.to_datetime(lowtide.time)
# # #     #         return time_range
# # #     #         # tide_cq = lowtide.quantile(q=calculate_quantiles, dim="time").to_dataset()
