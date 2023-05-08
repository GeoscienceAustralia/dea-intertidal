import logging
import yaml
import fsspec
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd
from pathlib import Path


def configure_logging(name: str = "DEA Intertidal") -> logging.Logger:
    """
    Configure logging for the application.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file and returns data as a nested dictionary.

    config_path can be a path or URL to a web accessible YAML file
    """
    with fsspec.open(config_path, mode="r") as f:
        config = yaml.safe_load(f)
    return config


def round_date_strings(date, round_type="end"):
    """
    Round a date string up or down to the start or end of a given time
    period.

    Parameters
    ----------
    date : str
        Date string of variable precision (e.g. "2020", "2020-01",
        "2020-01-01").
    round_type : str, optional
        Type of rounding to perform. Valid options are "start" or "end".
        If "start", date is rounded down to the start of the time period.
        If "end", date is rounded up to the end of the time period.
        Default is "end".

    Returns
    -------
    date_rounded : str
        The rounded date string.

    Examples
    --------
    >>> round_date_strings('2020')
    '2020-12-31 00:00:00'

    >>> round_date_strings('2020-01', round_type='start')
    '2020-01-01 00:00:00'

    >>> round_date_strings('2020-01', round_type='end')
    '2020-01-31 00:00:00'
    """

    # Determine precision of input date string
    date_segments = len(date.split("-"))

    # If provided date has no "-", treat it as having year precision
    if date_segments == 1 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + YearBegin(0))
    elif date_segments == 1 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + YearEnd(0))

    # If provided date has one "-", treat it as having month precision
    elif date_segments == 2 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + MonthBegin(0))
    elif date_segments == 2 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + MonthEnd(0))

    # If more than one "-", then return date as-is
    elif date_segments > 2:
        date_rounded = date

    return date_rounded


def export_intertidal_rasters(
    ds,
    prefix="data/interim/testing",
    int_bands=None,
    int_nodata=-999,
    int_dtype=np.int16,
    float_dtype=np.float32,
    overwrite=True,
):
    """
    Export outputs of the DEA Intertidal workflow to COG GeoTIFF files.

    If a band contains "elevation" in the name it is exported as a
    float32 data type. Otherwise, the band is exported as an integer16
    data type, after filling NaN with the nodata value and setting the
    nodata attribute on the layer.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the bands to be exported.
    prefix : str, optional
        A string that will be used as a prefix for the output file
        names (default is "testing").
    int_bands : tuple or list, optional
        A list of bands to export as integer datatype. If None, will use
        the following list of bands: ("exposure", "extents",
        "offset_hightide", "offset_lowtide", "spread")
    int_nodata : int, optional
        An integer that represents nodata values for integer bands
        (default is -999).
    int_dtype : string or numpy data type, optional
        The data type to use for integer layers (default is
        np.int16).
    float_dtype : string or numpy data type, optional
        The data type to use for floating point layers (default is
        np.float32).
    overwrite : bool, optional
        A boolean value that determines whether or not to overwrite
        existing files (default is True).

    Returns
    -------
    None
    """

    # Use default list of bands to convert to integers if none provided
    if int_bands is None:
        int_bands = (
            "exposure",
            "extents",
            "offset_hightide",
            "offset_lowtide",
            "spread",
        )

    for band in ds:
        # Export specific bands as integer16 data types by first filling
        # NaN with nodata value before converting to int, then setting
        # nodata attribute on layer
        if band in int_bands:
            band_da = ds[band].fillna(int_nodata).astype(int_dtype)
            band_da.attrs["nodata"] = int_nodata

        # Export other bands as float32 data types
        else:
            band_da = ds[band].astype(float_dtype)

        # Export band to file
        band_da.odc.write_cog(fname=f"{prefix}_{band}.tif", overwrite=overwrite)


def intertidal_hillshade(
    elevation,
    extents,
    azdeg=315,
    altdeg=45,
    dyx=10,
    vert_exag=100,
    **shade_kwargs,
):
    """
    Create a hillshade array for an intertidal zone given an elevation
    array and an extents array.

    Parameters
    ----------
    elevation : str or xr.DataArray
        An xr.DataArray or a path to the elevation raster file.
    extents : str or xr.DataArray
        xr.DataArray or a path to the extents raster file.
    azdeg : float, optional
        The azimuth angle of the light source, in degrees. Default is 315.
    altdeg : float, optional
        The altitude angle of the light source, in degrees. Default is 45.
    dyx : float, optional
        The distance between pixels in the x and y directions, in meters.
        Default is 10.
    vert_exag : float, optional
        The vertical exaggeration of the hillshade. Default is 100.
    **shade_kwargs : optional
        Additional keyword arguments to pass to
        `matplotlib.colors.LightSource.shade()`.

    Returns
    -------
    xr.DataArray
        The hillshade array for the intertidal zone.
    """

    from matplotlib.colors import LightSource, Normalize
    import matplotlib.pyplot as plt
    import xarray as xr

    # Read data
    if isinstance(elevation, str):
        elevation = xr.open_rasterio(elevation).squeeze("band")
    if isinstance(extents, str):
        extents = xr.open_rasterio(extents).squeeze("band")

    # Fill upper and bottom of intertidal zone with min and max heights
    # so that hillshade can be applied across the entire raster
    elevation_filled = xr.where(extents == 0, elevation.min(), elevation)
    elevation_filled = xr.where(extents == 2, elevation.max(), elevation_filled)

    # Create hillshade based on elevation data
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hillshade = ls.shade(
        elevation_filled.values,
        cmap=plt.cm.viridis,
        blend_mode=lambda x, y: x * y,
        vert_exag=vert_exag,
        dx=dyx,
        dy=dyx,
        **shade_kwargs,
    )

    # Mask out non-intertidal pixels
    hillshade = np.where(
        np.expand_dims(extents.values == 1, axis=-1), hillshade, np.nan
    )

    # Create a new xarray data array from the numpy array
    hillshaded_da = xr.DataArray(
        hillshade,
        dims=["y", "x", "variables"],
        coords={
            "y": elevation.y,
            "x": elevation.x,
            "variables": ["red", "green", "blue", "alpha"],
        },
    )

    return hillshaded_da


def pixel_ebb_flow(satellite_ds, tide_m, offset_min=15):
    """
    Computes whether each pixel in a satellite dataset represents ebb or
    flow tide conditions.

    This function compares the original modelled tide heights from the
    moment of satellite data acquisition with new tide hieghts modelled
    after after shifting the original timesteps forward by `offset_min`
    minutes. If a pixel's tide height is lower at the `tide_m_offset`
    time than at the original time, it is considered to be flowing
    (rising) over time. Otherwise, it is considered to be ebbing
    (falling) over time.

    Parameters
    ----------
    tide_m : xarray.DataArray
        An array containing modelled tide heights for each pixel and
        timestep in a satellite dataset.
    offset_min : int, optional
        The time offset in minutes to use for the comparison (default
        is 15).

    Returns
    -------
    ebb_flow_da : xarray.DataArray
        A new DataArray with the same dimensions as `tide_m` containing
        a boolean value for each pixel indicating whether the tide is
        ebbing (False) or flowing (True).
    tide_m_offset : xarray.DataArray
        An array containing modelled tides offset by `offset_min`
        minutes in time, for reference.
    """

    # Offset times in original array by X minutes forward in time
    times_offset = tide_m.time + np.timedelta64(offset_min, "m")

    # Model tides into every pixel in the three-dimensional (x, y, time)
    # input array
    tide_m_offset, _ = pixel_tides(tide_m, times=times_offset, resample=True)

    # Restore original times so both arrays can be combined using xarray
    tide_m_offset["time"] = tide_m["time"]

    # For each pixel, test whether original tides were lower than
    # tides offset to X minutes later. If they were lower, the tide was
    # "flowing" (rising) over time, if they were higher the tide was
    # "ebbing" (falling) over time
    ebb_flow_da = (tide_m < tide_m_offset).rename("ebb_flow")

    return ebb_flow_da, tide_m_offset
