import datetime
from math import ceil

import numpy as np
import pandas as pd
import pytz
import xarray as xr
from eo_tides.eo import _pixel_tides_resample, pixel_tides

from intertidal.utils import configure_logging, round_date_strings


def temporal_filters(x, time_range, dem):
    """Identify and extract temporal-specific dates and times to feed into
    tidal modelling for custom exposure calculations.

    Parameters
    ----------
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
    # Import sunriset here to avoid dependency issues if function not used
    import sunriset

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
    if x == "wet":
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
    if x == "summer":
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
    if x == "autumn":
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
    if x == "winter":
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
    if x == "spring":
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
    if x == "jan":
        return time_range.drop(time_range[time_range.month != 1])
    if x == "feb":
        return time_range.drop(time_range[time_range.month != 2])
    if x == "mar":
        return time_range.drop(time_range[time_range.month != 3])
    if x == "apr":
        return time_range.drop(time_range[time_range.month != 4])
    if x == "may":
        return time_range.drop(time_range[time_range.month != 5])
    if x == "jun":
        return time_range.drop(time_range[time_range.month != 6])
    if x == "jul":
        return time_range.drop(time_range[time_range.month != 7])
    if x == "aug":
        return time_range.drop(time_range[time_range.month != 8])
    if x == "sep":
        return time_range.drop(time_range[time_range.month != 9])
    if x == "oct":
        return time_range.drop(time_range[time_range.month != 10])
    if x == "nov":
        return time_range.drop(time_range[time_range.month != 11])
    if x == "dec":
        return time_range.drop(time_range[time_range.month != 12])
    if x in ["daylight", "night"]:
        # Identify the central coordinate directly from the dem GeoBox
        tidepost_lon_4326, tidepost_lat_4326 = dem.odc.geobox.extent.centroid.to_crs("EPSG:4326").coords[0]

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
        sun_df = sunriset.to_pandas(startdate, tidepost_lat_4326, tidepost_lon_4326, local_tz, diff)

        # Set the index as a datetimeindex to match the ModTides ds
        sun_df = sun_df.set_index(pd.DatetimeIndex(sun_df.index))

        # Append the date to each Sunrise and Sunset time
        sun_df["Sunrise dt"] = sun_df.index + sun_df["Sunrise"]
        sun_df["Sunset dt"] = sun_df.index + sun_df["Sunset"]

        # Create new dataframes where daytime and nightime datetimes are
        # recorded, then merged on a new `Sunlight` column
        daytime = pd.DataFrame(data="Sunrise", index=sun_df["Sunrise dt"], columns=["Sunlight"])
        nighttime = pd.DataFrame(data="Sunset", index=sun_df["Sunset dt"], columns=["Sunlight"])
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
    **model_tides_kwargs,
):
    """Calculate intertidal exposure, indicating the proportion of time
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
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

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

    # Create empty xarray.Datasets to store outputs into
    exposure_ds = xr.Dataset(coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values)))
    modelledtides_ds = xr.Dataset(coords=dict(y=(["y"], dem.y.values), x=(["x"], dem.x.values)))

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
    all_filters = temp_filters + ["unfiltered"]
    for x in filters:
        assert x in all_filters, f'Nominated filter "{x}" is not in {all_filters}. Check spelling and retry'

    # Run tide model at low resolution
    modelledtides_lowres = pixel_tides(
        data=dem,
        time=time_range,
        model=tide_model,
        directory=tide_model_dir,
        resample=False,
        **model_tides_kwargs,
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
        modelledtides_x = modelledtides_x.quantile(q=calculate_quantiles, dim="time").to_dataset().tide_m

        # Add modelledtides_x to output dataset
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

        # Reorder dimensions
        if "time" in list(idxmin.dims):
            idxmin = idxmin.transpose("time", "y", "x")
        else:
            idxmin = idxmin.transpose("y", "x")

        # Convert to percentage and add as variable in exposure dataset
        exposure_ds[str(x)] = idxmin * 100

    if return_tide_modelling:
        return exposure_ds, modelledtides_ds, modelledtides_1d, timeranges
    return exposure_ds, modelledtides_ds
