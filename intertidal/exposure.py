import xarray as xr
import numpy as np

from dea_tools.coastal import pixel_tides


def exposure(
    dem,
    timerange,
):
    """
    Calculate exposure percentage for each pixel based on tide-height 
    differences between the NIDEM value and percentile values of the 
    tide model for a given time range.

    Parameters
    ----------
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data 
        and coordinates and attributes metadata.
    timerange : tuple
        Tuple containing start and end time of time range to be used for
        tide model in the format of "YYYY-MM-DD".

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

    # Create the tide-height percentiles from which to calculate 
    # exposure statistics
    pc_range = np.linspace(0, 1, 101)

    # Run the pixel_tides function with the calculate_quantiles option. 
    # For each pixel, an array of tideheights is returned, corresponding
    # to the percentiles from pc_range of the timerange-tide model that 
    # each tideheight appears in the model.
    tide_cq, _ = pixel_tides(
        dem, resample=True, calculate_quantiles=pc_range, times=timerange
    )

    # Calculate the tide-height difference between the NIDEM value and 
    # each percentile value per pixel
    diff = abs(tide_cq - dem)

    # Take the percentile of the smallest tide-height difference as the 
    # exposure % per pixel
    idxmin = diff.idxmin(dim="quantile")

    # Convert to percentage 
    exposure = (idxmin * 100)

    return exposure, tide_cq
