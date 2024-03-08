import xarray as xr
import numpy as np
import geopandas as gpd

from intertidal.tide_modelling import pixel_tides_ensemble
from intertidal.utils import configure_logging


def exposure(
    dem,
    times,
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
    run_id=None,
    log=None,
):
    """
    Calculate intertidal exposure for each pixel, indicating the 
    proportion of time that each pixel was "exposed" from tidal
    inundation during the time period of interest.
    
    The exposure calculation is based on tide-height differences between
    the elevation value and modelled tide height percentiles.

    Parameters
    ----------
    dem : xarray.DataArray
        xarray.DataArray containing Digital Elevation Model (DEM) data.
    times : pandas.DatetimeIndex or list of pandas.Timestamps
        High frequency times used to model tides across the period of
        interest. These are used to evaluate how frequently each pixel
        in the DEM was exposed or inundated by the tide.
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
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    exposure : xarray.DataArray
        An array containing the percentage time 'exposure' of
        each pixel from tidal inundation for the duration of the modelling
        period.
    tide_cq : xarray.DataArray
        An array containing the quantiled high temporal resolution tide
        modelling for each pixel. Dimensions should be 'quantile', 'x' and 'y'.

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
    
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id

    # Create the tide-height percentiles from which to calculate
    # exposure statistics
    tide_percentiles = np.linspace(0, 1, 101)     
    
    # Run the `pixel_tides_ensemble` function with the `calculate_quantiles`
    # option. For each pixel, an array of tide heights is returned, 
    # corresponding to the percentiles of the tide range specified by
    # `tide_percentiles`. 
    log.info(f"{run_id}: Modelling tide heights for each pixel")
    tide_cq, _ = pixel_tides_ensemble(
        ds=dem,
        ancillary_points="data/raw/tide_correlations_2017-2019.geojson",
        model=tide_model,
        directory=tide_model_dir,
        calculate_quantiles=tide_percentiles,
        times=times,
    )

    # Calculate the tide-height difference between the elevation value and
    # each percentile value per pixel
    diff = abs(tide_cq - dem)

    # Take the percentile of the smallest tide-height difference as the
    # exposure % per pixel
    idxmin = diff.idxmin(dim="quantile")

    # Convert to percentage
    exposure = idxmin * 100

    return exposure, tide_cq
