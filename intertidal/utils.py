import logging

import bottleneck
import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd


def configure_logging(name: str = "DEA Intertidal") -> logging.Logger:
    """Configure logging for the application."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def round_date_strings(date, round_type="end"):
    """Round a date string up or down to the start or end of a given time
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
    >>> round_date_strings("2020")
    '2020-12-31 00:00:00'

    >>> round_date_strings("2020-01", round_type="start")
    '2020-01-01 00:00:00'

    >>> round_date_strings("2020-01", round_type="end")
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


def intertidal_hillshade(
    elevation,
    freq,
    azdeg=315,
    altdeg=45,
    dyx=10,
    vert_exag=100,
    **shade_kwargs,
):
    """Create a hillshade array for an intertidal zone given an elevation
    array and a frequency array.

    Parameters
    ----------
    elevation : str or xr.DataArray
        Elevation data
    freq : str or xr.DataArray
        NDWI frequency data
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
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import LightSource

    # Fill upper and bottom of intertidal zone with min and max heights
    # so that hillshade can be applied across the entire raster
    elev_min, elev_max = elevation.quantile([0, 1])
    elevation_filled = xr.where(elevation.isnull() & (freq < 50), elev_max, elevation).fillna(elev_min)

    from scipy.ndimage import gaussian_filter

    input_data = gaussian_filter(elevation_filled, sigma=1)

    # Create hillshade based on elevation data
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hillshade = ls.shade(
        input_data,
        cmap=plt.cm.viridis,
        blend_mode=lambda x, y: x * y,
        vert_exag=vert_exag,
        dx=dyx,
        dy=dyx,
        **shade_kwargs,
    )

    # Mask out non-intertidal pixels
    hillshade = np.where(np.expand_dims(elevation.notnull().values, axis=-1), hillshade, np.nan)

    # Create a new xarray data array from the numpy array
    hillshaded_da = xr.DataArray(
        hillshade * 255,
        dims=["y", "x", "variables"],
        coords={
            "y": elevation.y,
            "x": elevation.x,
            "variables": ["r", "g", "b", "a"],
        },
    )

    return hillshaded_da


def spearman_correlation(x, y, dim):
    """Fast Spearman correlation using bottleneck and apply_ufunc."""

    def _covariance_gufunc(x, y):
        return np.nanmean(
            (x - np.nanmean(x, axis=-1, keepdims=True)) * (y - np.nanmean(y, axis=-1, keepdims=True)),
            axis=-1,
        )

    def _pearson_correlation_gufunc(x, y):
        return _covariance_gufunc(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))

    def _spearman_correlation_gufunc(x, y):
        x_ranks = bottleneck.nanrankdata(x, axis=-1)
        y_ranks = bottleneck.nanrankdata(y, axis=-1)
        return _pearson_correlation_gufunc(x_ranks, y_ranks)

    return xr.apply_ufunc(
        _spearman_correlation_gufunc,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        dask="parallelized",
        output_dtypes=[float],
    )
