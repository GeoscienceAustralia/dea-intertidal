import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat
import click

import datacube
import odc.geo.xr
from odc.algo import mask_cleanup, xr_quantile
from datacube.utils.geometry import Geometry
from datacube.utils.aws import configure_s3_access

from dea_tools.coastal import pixel_tides
from dea_tools.dask import create_local_dask_cluster

from intertidal.utils import (
    load_config,
    configure_logging,
    round_date_strings,
    export_intertidal_rasters,
)
from intertidal.extents import extents
from intertidal.exposure import exposure
from intertidal.tidal_bias_offset import bias_offset, tidal_offset_tidelines


def load_data(
    dc,
    geom,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:3577",
    s2_prod="s2_nbart_ndwi",
    ls_prod="ls_nbart_ndwi",
    config_path="configs/dea_virtual_product_landsat_s2.yaml",
    filter_gqa=True,
):
    """
    Load cloud-masked Landsat and Sentinel-2 NDWI data for a given
    spatial and temporal extent.

    Parameters
    ----------
    dc : Datacube
        A datacube instance connected to a database.
    geom : Geometry object from datacube.utils.geometry
        A geometry object from `datacube.utils.geometry` that defines
        the spatial extent of interest.
    time_range : tuple, optional
        A tuple containing the start and end date for the time range of
        interest, in the format (start_date, end_date). The default is
        ("2019", "2021").
    resolution : int or float, optional
        The spatial resolution (in metres) to load data at. The default
        is 10.
    crs : str, optional
        The coordinate reference system (CRS) to project data into. The
        default is Australian Albers "EPSG:3577".
    s2_prod : str, optional
        The name of the virtual product to use for Sentinel-2 data. The
        default is "s2_nbart_ndwi".
    ls_prod : str, optional
        The name of the virtual product to use for Landsat data. The
        default is "ls_nbart_ndwi".
    config_path : str, optional
        The path to the virtual product configuration file. The default is
        "configs/dea_virtual_product_landsat_s2.yaml".
    filter_gqa : bool, optional
        Whether or not to filter Sentinel-2 data using the GQA filter.
        The default is True.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Landsat and Sentinel-2
        data, converted to NDWI with cloud masking applied.
    """

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry

    # Load in virtual product catalogue
    catalog = catalog_from_file(config_path)

    # Create the 'query' dictionary object
    query_params = {
        "geopolygon": geom,
        "time": time_range,
        "resolution": (-resolution, resolution),
        "output_crs": crs,
        "dask_chunks": {"time": 1, "x": 2048, "y": 2048},
        "resampling": {
            "*": "cubic",
            "oa_nbart_contiguity": "nearest",
            "oa_fmask": "nearest",
            "oa_s2cloudless_mask": "nearest",
        },
    }

    # Optionally add GQA
    # TODO: Remove once Sentinel-2 GQA issue is resolved
    if filter_gqa:
        query_params["gqa_iterative_mean_xy"] = (0, 1)

    # Output list
    data_list = []

    # If Sentinel-2 data is requested
    if s2_prod is not None:
        # Load Sentinel-2 data
        product = catalog[s2_prod]
        s2_ds = product.load(dc, **query_params)

        # Apply cloud mask and contiguity mask
        s2_ds_masked = s2_ds.where(s2_ds.cloud_mask == 1 & s2_ds.contiguity)
        data_list.append(s2_ds_masked)

    # If Landsat data is requested
    if ls_prod is not None:
        # Load Landsat data
        product = catalog[ls_prod]
        ls_ds = product.load(dc, **query_params)

        # Clean cloud mask by applying morphological closing to all
        # valid (non cloud, shadow or nodata) pixels. This removes
        # long, narrow features like false positives over bright beaches.
        good_data_cleaned = mask_cleanup(
            mask=ls_ds.cloud_mask.isin([1, 4, 5]),
            mask_filters=[("closing", 5)],
        )

        # Dilate cloud and shadow. To ensure that nodata areas (e.g.
        # Landsat 7 SLC off stripes) are not also dilated, only dilate
        # mask pixels (e.g. values 0 in `good_data_cleaned`) that are
        # outside of the original nodata pixels (e.g. not 0 in
        # `ls_ds.cloud_mask`)
        good_data_mask = mask_cleanup(
            mask=(good_data_cleaned == 0) & (ls_ds.cloud_mask != 0),
            mask_filters=[("dilation", 5)],
        )

        # Apply cloud mask and contiguity mask
        ls_ds_masked = ls_ds.where(~good_data_mask & ls_ds.contiguity)
        data_list.append(ls_ds_masked)

    # Combine into a single ds, sort and drop no longer needed bands
    satellite_ds = (
        xr.concat(data_list, dim="time")
        .sortby("time")
        .drop(["cloud_mask", "contiguity"])
    )
    return satellite_ds


def ds_to_flat(
    satellite_ds,
    ndwi_thresh=0.1,
    index="ndwi",
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.2,
):
    """
    Flattens a three-dimensional array (x, y, time) to a two-dimensional
    array (time, z) by selecting only pixels with positive correlations
    between water observations and tide height. This greatly improves
    processing time by ensuring only a narrow strip of pixels along the
    coastline are analysed, rather than the entire x * y array.
    The x and y dimensions are stacked into a single dimension (z)

    Parameters
    ----------
    satellite_ds : xr.Dataset
        Three-dimensional (x, y, time) xarray dataset with variable
        "tide_m" and a water index variable as provided by `index`.
    ndwi_thresh : float, optional
        Threshold for NDWI index used to identify wet or dry pixels.
        Default is 0.1.
    index : str, optional
        Name of the water index variable. Default is "ndwi".
    min_freq : float, optional
        Minimum frequency of wetness required for a pixel to be included
        in the output. Default is 0.01.
    max_freq : float, optional
        Maximum frequency of wetness required for a pixel to be included
        in the output. Default is 0.99.
    min_correlation : float, optional
        Minimum correlation between water index values and tide height
        required for a pixel to be included in the output. Default is
        0.2.

    Returns
    -------
    flat_ds : xr.Dataset
        Two-dimensional xarray dataset with dimensions (time, z)
    freq : xr.DataArray
        Frequency of wetness for each pixel.
    good_mask : xr.DataArray
        Boolean mask indicating which pixels meet the inclusion criteria.
    corr : xr.DataArray
        Correlation of pixel wetness to tide height
    """

    # Calculate frequency of wet per pixel, then threshold
    # to exclude always wet and always dry
    freq = (
        (satellite_ds[index] > ndwi_thresh)
        .where(~satellite_ds[index].isnull())
        .mean(dim="time")
        .drop_vars("variable")
    )
    good_mask = (freq >= min_freq) & (freq <= max_freq)

    # Flatten to 1D
    flat_ds = satellite_ds.stack(z=("x", "y")).where(
        good_mask.stack(z=("x", "y")), drop=True
    )

    # Calculate correlations, and keep only pixels with positive
    # correlations between water observations and tide height
    correlations = xr.corr(flat_ds[index] > ndwi_thresh, flat_ds.tide_m, dim="time")
    flat_ds = flat_ds.where(correlations > min_correlation, drop=True)
    
    # Return correlations to 3D array for use in later intertidal modules
    corr = correlations.unstack("z").reindex_like(satellite_ds).transpose("y","x")
    
    print(
        f"Reducing analysed pixels from {freq.count().item()} to {len(flat_ds.z)} ({len(flat_ds.z) * 100 / freq.count().item():.2f}%)"
    )
    return flat_ds, freq, good_mask, corr


def rolling_tide_window(
    i,
    flat_ds,
    window_spacing,
    window_radius,
    tide_min,
    statistic="median",
):
    """
    Filter observations from a flattened array that fall within a
    specific tide window, and summarise these values using a given
    statistic (median, mean, or quantile).

    This is used to smooth NDWI values along the tide dimension so that
    we can more easily identify the transition from dry to wet pixels
    with increasing tide height.

    Parameters
    ----------
    i : int
        Index of the current window.
    flat_ds : xarray.Dataset
        Input dataset with tide observations (tide_m) as a dimension.
    window_spacing : float
        Provides the spacing of each rolling window interval in tide
        units (e.g. metres).
    window_radius : float
        Provides the radius/width of each rolling window in tide units
        (e.g. metres).
    tide_min : float
        Bottom edge of the rolling window in tide units (e.g. metres).
    statistic : str, optional
        Statistic to apply on the values within each window. One of
        ["median", "mean", "quantile"]. Default is "median".

    Returns
    -------
    xarray.Dataset
        Aggregated dataset of the selected statistic and additional
        information on the window. The returned dataset includes the
        aggregated NDWI values within the window.

    """

    # Set min and max thresholds to filter dataset
    thresh_centre = tide_min + (i * window_spacing)
    thresh_min = thresh_centre - window_radius
    thresh_max = thresh_centre + window_radius

    # Filter dataset
    masked_ds = flat_ds.where(
        (flat_ds.tide_m >= thresh_min) & (flat_ds.tide_m <= thresh_max)
    )

    # Apply median or quantile
    if statistic == "quantile":
        ds_agg = xr_quantile(
            src=masked_ds.dropna(dim="time", how="all"),
            quantiles=[0.1, 0.5, 0.9],
            nodata=np.nan,
        )
    elif statistic == "median":
        ds_agg = masked_ds.median(dim="time")
    elif statistic == "mean":
        ds_agg = masked_ds.mean(dim="time")

    return ds_agg


def pixel_rolling_median(flat_ds, windows_n=100, window_prop_tide=0.15, max_workers=64):
    """
    Calculate rolling medians for each pixel in an xarray.Dataset from
    low to high tide, using a set number of rolling windows (defined
    by `windows_n`) with radius determined by the proportion of the tide
    range specified by `window_prop_tide`.
    
    For each window, the function returns the median of all tide heights
    and NDWI index values within the window, and returns an array with a
    new "interval" dimension that summarises these values from low to
    high tide.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened two dimensional (time, z) xr.Dataset containing
        variables "ndwi" and "tide_height", as produced by the
        `ds_to_flat` function
    windows_n : int, optional
        Number of rolling windows to iterate over, by default 100
    window_prop_tide : float, optional
        Proportion of the tide range to use for each window radius,
        by default 0.15
    max_workers : int, optional
        Maximum number of worker processes to use for parallel
        execution, by default 64

    Returns
    -------
    xarray.Dataset
        An two dimensional (interval, z) xarray.Dataset containing the
        rolling median for each pixel from low to high tide.
    """

    # First obtain some required statistics on the satellite-observed
    # min, max and tide range per pixel
    tide_max = flat_ds.tide_m.max(dim="time")
    tide_min = flat_ds.tide_m.min(dim="time")
    tide_range = tide_max - tide_min

    # To conduct a pixel-wise rolling median, we first need to calculate
    # some statistics on the tides observed for each individual pixel in
    # the study area. These are then used to calculate rolling windows
    # that are unique/tailored for the tidal regime of each pixel:
    #
    #     - window_radius_tide: Provides the radius/width of each
    #       rolling window in tide units (e.g. metres).
    #     - window_spacing_tide: Provides the spacing of each rolling
    #       window interval in tide units (e.g. metres)
    #     - window_offset: Ensures that analysis covers the entire tide
    #       range by starting the first rolling window beneath the
    #       lowest tide, and finishing the final rolling window after
    #       the highest tide
    #
    window_radius_tide = tide_range * window_prop_tide
    window_spacing_tide = tide_range / windows_n
    window_offset = int((windows_n * window_prop_tide) / 2.0)

    # Parallelise pixel-based rolling median using `concurrent.futures`
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create rolling intervals to iterate over
        rolling_intervals = range(-window_offset, windows_n + window_offset)

        # Place itervals in a iterable along with params for each call
        to_iterate = (
            rolling_intervals,
            *(
                repeat(i, len(rolling_intervals))
                for i in [flat_ds, window_spacing_tide, window_radius_tide, tide_min]
            ),
        )

        # Apply func in parallel
        out_list = list(
            tqdm(
                executor.map(rolling_tide_window, *to_iterate),
                total=len(list(rolling_intervals)),
            )
        )

    # Combine to match the shape of the original dataset, then sort from
    # low to high tide
    interval_ds = xr.concat(out_list, dim="interval").sortby("interval")

    return interval_ds


def pixel_dem(interval_ds, flat_ds, ndwi_thresh=0.1, smooth_radius=20):
    """
    Calculates an estimate of intertidal elevation based on satellite
    imagery and tide data. Elevation is calculated by identifying the
    max tide per pixel where a rolling median of NDWI == land.

    Parameters
    ----------
    interval_ds : xarray.Dataset
        A flattened 2D xarray Dataset containing the rolling median for
        each pixel from low to high tide for the given area, with
        variables 'tide_m' and 'ndwi'.
    flat_ds : xarray.Dataset
        A flattened two dimensional (time, z) xr.Dataset containing
        variables "ndwi" and "tide_height", as produced by the
        `ds_to_flat` function
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI), above which pixels are considered water. Defaults to
        0.1, which appears to more reliably capture the transition from
        dry to wet pixels than 0.0.
    smooth_radius : int, optional
        A rolling mean filter can be applied to smooth data along the
        tide interval dimension. This produces smoother DEM surfaces
        than using the rolling median directly. Defaults to 20; set to
        0 to deactivate.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the DEM for the given area, with
        a single variable 'elevation'.

    Notes
    -----
    This function can additionally apply a rolling mean to smooth the
    interval data before identifying the max tide per pixel where
    NDWI == land. This produces a cleaner and less noisy output.
    """

    # Smooth tidal intervals using a rolling mean
    if smooth_radius > 1:
        smoothed_ds = interval_ds.rolling(
            interval=smooth_radius, center=False, min_periods=1
        ).mean()
    else:
        smoothed_ds = interval_ds

    # Identify the max tide per pixel where rolling median NDWI == land.
    # This represents the tide height at which the pixel transitions from
    # dry to wet as it gets inundated by tidal waters.
    tide_dry = smoothed_ds.tide_m.where(smoothed_ds.ndwi <= ndwi_thresh)
    tide_thresh = tide_dry.max(dim="interval")

    # Remove any pixel where tides max out (i.e. always land)
    tide_max = smoothed_ds.tide_m.max(dim="interval")
    always_dry = tide_thresh >= tide_max
    dem_flat = tide_thresh.where(~always_dry).drop("variable")

    # Export as xr.Dataset
    return dem_flat.to_dataset(name="elevation")


def pixel_uncertainty(flat_ds, flat_dem, ndwi_thresh=0.1, min_q=0.25, max_q=0.75, output_auxiliaries=False):
    """
    Calculate uncertainty bounds around a modelled elevation based on
    observations that were misclassified by a given NDWI threshold.

    The function identifies observations that were misclassified by the
    modelled elevation, i.e., wet observations (NDWI > threshold) at
    lower tide heights than the modelled elevation, or dry observations
    (NDWI < threshold) at higher tide heights than the modelled
    elevation. It calculates the interquartile tide height range of
    these misclassified observations.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened (2D) dataset containing dimensions "time" and "z",
        and variables "ndwi" (Normalized Difference Water Index) and
        "tide_m" (tide height) for each satellite observation.
    flat_dem : xarray.DataArray
        A 2D array containing modelled elevations per pixel, as
        generated by `intertidal.elevation.pixel_dem`.
    ndwi_thresh : float, optional
        A threshold value for NDWI, below which an observation is
        considered "dry", and above which it is considered "wet". The
        default is 0.1.
    min_q, max_q : float, optional
        The minimum and maximum quantiles used to estimate uncertainty
        bounds based on misclassified points. Defaults to interquartile
        range, or 0.25, 0.75. This provides a balance between capturing
        the range of uncertainty at each pixel, while not being overly
        influenced by outliers in `flat_ds`.
    output_auxiliaries : bool, optional
        True if auxiliary outputs are required for debugging

    Returns
    -------
    tuple of xarray.DataArray
        The lower and upper uncertainty bounds around the modelled
        elevation, and the summary uncertainty range between them.
    misclassified_ds : xr.DataSet
        If output_auxiliaries = True, a flattened Dataset is returned
        showing all identified misclassified pixels in both the `ndwi`
        and `tide_m` arrays.
    """

    # Identify observations that were misclassifed by our modelled
    # elevation: e.g. wet observations (NDWI > threshold) at lower tide
    # heights than our modelled elevation, or dry observations (NDWI <
    # threshold) at higher tide heights than our modelled elevation.
    misclassified_wet = (flat_ds.ndwi > ndwi_thresh) & (
        flat_ds.tide_m < flat_dem.elevation
    )
    misclassified_dry = (flat_ds.ndwi < ndwi_thresh) & (
        flat_ds.tide_m > flat_dem.elevation
    )
    misclassified_all = misclassified_wet | misclassified_dry
    misclassified_ds = flat_ds.where(misclassified_all).drop("variable")

    # Calculate interquartile tide height range of our misclassified
    # observations to obtain lower and upper uncertainty bounds around our
    # modelled elevation.
    misclassified_q = xr_quantile(
        src=misclassified_ds.dropna(dim="time", how="all")[["tide_m"]],
        quantiles=[min_q, max_q],
        nodata=np.nan,
    ).tide_m.fillna(flat_dem.elevation)

    # Clip min and max uncertainty to modelled elevation to ensure lower
    # bounds are not above modelled elevation (and vice versa)
    dem_flat_low = np.minimum(
        misclassified_q.sel(quantile=min_q, drop=True), flat_dem.elevation
    )
    dem_flat_high = np.maximum(
        misclassified_q.sel(quantile=max_q, drop=True), flat_dem.elevation
    )

    # Subtract low from high DEM to summarise uncertainy range
    dem_flat_uncertainty = dem_flat_high - dem_flat_low

    if output_auxiliaries:
        return (
        dem_flat_low,
        dem_flat_high,
        dem_flat_uncertainty,
        misclassified_ds,
    )
    
    else:
        return (
            dem_flat_low,
            dem_flat_high,
            dem_flat_uncertainty,
        )


def flat_to_ds(flat_ds, template, stacked_dim="z"):
    """
    Convert a flattened xarray Dataset with a stacked dimension to its
    original spatial dimensions, based on a given template.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened xarray.Dataset, i.e., a dataset where each "y", "x"
        pixel is stacked into a single "z" dimension.
    template : xarray.Dataset
        An xarray.Dataset containing the original spatial dimensions and
        coordinates of the data, used as a template to reshape the flattened
        data back to the spatial dimensions.
    stacked_dim : str, optional
        The name of the stacked dimension in the flattened dataset. The
        default is "z".

    Returns
    -------
    xarray.Dataset
        The unflattened xarray Dataset, with the same spatial dimensions
        (e.g. "y", "x") as the template.

    Notes
    -----
    The function unstacks the flattened dataset along the stacked
    dimension, reindexes the resulting dataset to match the spatial
    dimensions and coordinates of the template, and transposes the
    dimensions to match the order of the template's "y", "x", and
    variable dimensions.
    """

    # Unstack back to match template array
    unstacked_ds = (
        flat_ds.unstack(stacked_dim)
        .reindex_like(template)
        .transpose(*template.odc.spatial_dims)
    )

    return unstacked_ds


def elevation(
    study_area,
    start_date="2020",
    end_date="2022",
    resolution=10,
    crs="EPSG:3577",
    ndwi_thresh=0.1,
    include_s2=True,
    include_ls=True,
    filter_gqa=False,
    config_path="configs/dea_intertidal_config.yaml",
    log=None,
    output_auxiliaries = False,
):
    """
    Calculates DEA Intertidal Elevation using satellite imagery and
    tidal modeling.

    Parameters
    ----------
    study_area : int or str or Geometry
        Study area polygon represented as either the ID of a tile grid
        cell, or a Geometry object.
    start_date : str, optional
        Start date of data to load (inclusive), by default '2020'. Can
        be any string supported by datacube (e.g. '2020-01-01')
    end_date : str, optional
        End date of data to load (inclusive), by default '2022'. Can
        be any string supported by datacube (e.g. '2022-12-31')
    resolution : int, optional
        Pixel size in meters, by default 10.
    crs : str, optional
        Coordinate reference system, by default "EPSG:3577".
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI) above which pixels are considered water, by default 0.1.
    include_s2 : bool, optional
        Whether to include Sentinel-2 data, by default True.
    include_ls : bool, optional
        Whether to include Landsat data, by default True.
    filter_gqa : bool, optional
        Whether to apply the GQA filter to the dataset, by default False.
    config_path : str, optional
        Path to the configuration file, by default
        "configs/dea_intertidal_config.yaml".
    log : logging.Logger, optional
        Logger object, by default None.
    output_auxiliaries : bool, optional
        True if auxiliary outputs are required for debugging

    Returns
    -------
    ds : xarray.Dataset
        xarray.Dataset object containing intertidal elevation and
        confidence values for each pixel in the study area.
    freq : xarray.DataArray
        The frequency layer summarising how frequently a pixel was
        wet across the time series.
    tide_m : xarray.DataArray
        An xarray.DataArray object containing the modeled tide
        heights for each pixel in the study area.
    good_mask : xarray.DataArray
        An xarray.DataArray identifiying candidate intertidal pixels.
        Only returned if `output_auxiliaries` is True.        
    misclassified_ds : xarray.DataArray
        An xarray.DataArray derived from a xarray.Dataset of the same 
        name. The returned variable is a count of misclassified ndwi
        pixels, reshaped after satellite_ds. Only returned if
        'output_auxiliaries' is True.
    """

    if log is None:
        log = configure_logging()

    # Create local dask cluster to improve data load time
    client = create_local_dask_cluster(return_client=True)

    # Connect to datacube
    dc = datacube.Datacube(app="Intertidal_elevation")

    # Load analysis params from config file
    config = load_config(config_path)

    # Load study area from tile grid if passed a string
    if isinstance(study_area, (int, str)):
        # Load study area
        gridcell_gdf = (
            gpd.read_file(config["Input files"]["grid_path"])
            .to_crs(epsg=4326)
            .set_index("id")
        )
        gridcell_gdf.index = gridcell_gdf.index.astype(str)
        gridcell_gdf = gridcell_gdf.loc[[str(study_area)]]

        # Create geom as input for dc.load
        geom = Geometry(geom=gridcell_gdf.iloc[0].geometry, crs="EPSG:4326")
        log.info(f"Study area {study_area}: Loaded study area grid")

    # Otherwise, use supplied geom
    else:
        geom = study_area
        study_area = "testing"
        log.info(f"Study area {study_area}: Loaded custom study area")

    # Load data
    log.info(f"Study area {study_area}: Loading satellite data")
    satellite_ds = load_data(
        dc=dc,
        geom=geom,
        time_range=(start_date, end_date),
        resolution=resolution,
        crs=crs,
        s2_prod="s2_nbart_ndwi" if include_s2 else None,
        ls_prod="ls_nbart_ndwi" if include_ls else None,
        config_path=config["Virtual product"]["virtual_product_path"],
        filter_gqa=filter_gqa,
    )[["ndwi"]]

    # Load data and close dask client
    satellite_ds.load()
    client.close()

    # Model tides into every pixel in the three-dimensional (x by y by
    # time) satellite dataset
    log.info(f"Study area {study_area}: Modelling tide heights for each pixel")
    tide_m, _ = pixel_tides(satellite_ds, resample=True)

    # Set tide array pixels to nodata if the satellite data array pixels
    # contain nodata. This ensures that we ignore any tide observations
    # where we don't have matching satellite imagery
    log.info(
        f"Study area {study_area}: Masking nodata and adding tide heights to satellite data array"
    )
    satellite_ds["tide_m"] = tide_m.where(
        ~satellite_ds.to_array().isel(variable=0).isnull()
    )

    # Flatten array from 3D (time, y, x) to 2D (time, z) and drop pixels
    # with no correlation with tide. This greatly improves processing
    # time by ensuring only a narrow strip of tidally influenced pixels
    # along the coast are analysed, rather than the entire study area.
    # (This step is later reversed using the `flat_to_ds` function)
    log.info(
        f"Study area {study_area}: Flattening satellite data array and filtering to tide influenced pixels"
    )
    flat_ds, freq, good_mask, corr = ds_to_flat(
        satellite_ds, ndwi_thresh=0.0, min_freq=0.01, max_freq=0.99, min_correlation=0.2
    )

    # Calculate per-pixel rolling median.
    log.info(f"Study area {study_area}: Running per-pixel rolling median")
    interval_ds = pixel_rolling_median(
        flat_ds, windows_n=100, window_prop_tide=0.15, max_workers=64
    )

    # Model intertidal elevation
    log.info(f"Study area {study_area}: Modelling intertidal elevation")
    flat_dem = pixel_dem(interval_ds, flat_ds, ndwi_thresh)

    # Model intertidal uncertainty and add arrays into elevation dataset
    log.info(f"Study area {study_area}: Modelling intertidal uncertainty")
    if output_auxiliaries:
        (elevation_low, 
         elevation_high, 
         elevation_uncertainty, 
         misclassified_ds) = pixel_uncertainty(flat_ds, flat_dem, ndwi_thresh, output_auxiliaries=output_auxiliaries)
        
        flat_dem[["elevation_low", 
                  "elevation_high", 
                  "elevation_uncertainty"]] = (elevation_low, 
                                               elevation_high, 
                                               elevation_uncertainty)
        
    else:
        flat_dem[
            ["elevation_low", "elevation_high", "elevation_uncertainty"]
        ] = pixel_uncertainty(flat_ds, flat_dem, ndwi_thresh)

    # Unstack into original spatial dimensions to create master dataset
    ds = flat_to_ds(flat_dem, satellite_ds)

    # Return master ds and frequency layer
    log.info(
        f"Study area {study_area}: Successfully completed intertidal elevation modelling"
    )
    if output_auxiliaries:
        
        # Unstack misclassified_ds
        # Note: not using the flat_to_ds func as the time dimension on misclassified_ds
        # was interfering with the transpose step
        misclassified_ds = misclassified_ds.unstack('z').reindex_like(satellite_ds)
        
        # Count misclassified pixels
        misclassified_ds = misclassified_ds.ndwi.count(dim='time').transpose("y", "x")
        
        return ds, freq, corr, tide_m, good_mask, misclassified_ds
    else:
        return ds, freq, corr, tide_m


@click.command()
@click.option(
    "--config_path",
    type=str,
    required=True,
    help="Path to the YAML config file defining inputs to "
    "use for this analysis. These are typically located in "
    "the `dea-intertidal/configs/` directory.",
)
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a unique ID of an analysis "
    "gridcell that will be used to run the analysis. This "
    'should match a row in the "id" column of the provided '
    "analysis gridcell vector file.",
)
@click.option(
    "--start_date",
    type=str,
    default="2020",
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. ",
)
@click.option(
    "--end_date",
    type=str,
    default="2022",
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. ",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="The spatial resolution in metres used to load satellite "
    "data and produce intertidal outputs. Defaults to 10 metre "
    "Sentinel-2 resolution.",
)
@click.option(
    "--ndwi_thresh",
    type=float,
    default=0.1,
    help="NDWI threshold used to identify the transition from dry to "
    "wet in the intertidal elevation calculation. Defaults to 0.1, "
    "which appears to more reliably capture this transition than 0.0.",
)
@click.option(
    "--modelled_freq",
    type=str,
    default="30min",
    help="The frequency at which to model tides across the entire "
    "analysis period as inputs to the exposure, LAT (lowest "
    "astronomical tide), HAT (highest astronomical tide), and "
    "spread/offset calculations. Defaults to '30min' which will "
    "generate a timestep every 30 minutes between 'start_date' and "
    "'end_date'.",
)
@click.option(
    "--tideline_offset_distance",
    type=int,
    default=500,
    help="The distance along each high and low tideline "
    "at which the respective high or low tide satellite "
    "offset will be calculated. By default, the distance "
    "is set to 500m.",
)
@click.option(
    "--exposure_offsets/--no-exposure_offsets",
    is_flag=True,
    default=True,
    help="Whether to run the Exposure and spread/offsets/tidelines "
    "steps of the Intertidal workflow. Defaults to True; can be set "
    "to False by passing `--no-exposure_offsets`.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    type=bool,
    default=True,
    help="Whether to use sign AWS requests for S3 access",
)
@click.option(
    "--output_auxiliaries",
    type=bool,
    default=False,
    help="Whether to output auxiliary files for debugging",
)
def intertidal_cli(
    config_path,
    study_area,
    start_date,
    end_date,
    resolution,
    ndwi_thresh,
    modelled_freq,
    tideline_offset_distance,
    exposure_offsets,
    aws_unsigned,
    output_auxiliaries,
):
    log = configure_logging(f"Intertidal processing for study area {study_area}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned) 

    try:
        if output_auxiliaries:
            # Calculate elevation
            (ds, 
             freq, 
             corr, 
             tide_m, 
             good_mask, 
             misclassified_ds) = elevation(
                study_area,
                start_date=start_date,
                end_date=end_date,
                resolution=resolution,
                crs="EPSG:3577",
                ndwi_thresh=ndwi_thresh,
                include_s2=True,
                include_ls=True,
                filter_gqa=False,
                config_path=config_path,
                log=log,
                output_auxiliaries=output_auxiliaries,
            )
            
            # Compile auxiliary files into xr.Dataset
            ds_debug = xr.Dataset()
            ds_debug['NDWI_freq'] = freq
            ds_debug['NDWI_tide_corr'] = corr
            ds_debug['intertidal_candidate_px'] = good_mask
            ds_debug['misclassified_px_count'] = misclassified_ds
            
            # Calculate extents
            log.info(f"Study area {study_area}: Calculating Extents layer")
            ds["extents"] = extents(freq, ds.elevation, corr)

            if exposure_offsets:
                # Calculate exposure
                log.info(f"Study area {study_area}: Calculating Exposure layer")
                all_timerange = pd.date_range(
                    start=round_date_strings(start_date, round_type="start"),
                    end=round_date_strings(end_date, round_type="end"),
                    freq=modelled_freq,
                )
                ds["exposure"], tide_cq = exposure(ds.elevation, all_timerange)

                # Calculate spread, offsets and HAT/LAT/LOT/HOT
                log.info(
                    f"Study area {study_area}: Calculating spread, offset "
                    "and HAT/LAT/LOT/HOT layers"
                )
                (
                    ds["lat"],
                    ds["hat"],
                    ds["lot"],
                    ds["hot"],
                    ds["spread"],
                    ds["offset_lowtide"],
                    ds["offset_hightide"],
                ) = bias_offset(
                    tide_m=tide_m,
                    tide_cq=tide_cq,
                    extents=ds.extents,
                    lot_hot=True,
                    lat_hat=True,
                )

                # Calculate tidelines
                log.info(
                    f"Study area {study_area}: Calculating high and low tidelines "
                    "and associated satellite offsets"
                )
                (hightideline, lowtideline, tidelines_gdf) = tidal_offset_tidelines(
                    extents=ds.extents,
                    offset_hightide=ds.offset_hightide,
                    offset_lowtide=ds.offset_lowtide,
                    distance=tideline_offset_distance,
                )

                # Export high and low tidelines and the offset data
                log.info(
                    f"Study area {study_area}: Exporting high and low tidelines with satellite offset"
                )
                hightideline.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_offset_hightide.geojson"
                )
                lowtideline.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_offset_lowtide.geojson"
                )
                tidelines_gdf.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_tidelines_highlow.geojson"
                )

            else:
                log.info(
                    f"Study area {study_area}: Skipping Exposure and spread/offsets/tidelines calculation"
                )

            # Export layers as GeoTIFFs with optimised data types
            log.info(f"Study area {study_area}: Exporting outputs to GeoTIFFs")
            export_intertidal_rasters(
                ds, prefix=f"data/interim/{study_area}_{start_date}_{end_date}"
            )
            
            # Export auxiliary debug layers as GeoTIFFs with optimised data types
            log.info(f"Study area {study_area}: Exporting debugging outputs to GeoTIFFs")
            export_intertidal_rasters(
                ds_debug, prefix=f"data/interim/Debug_{study_area}_{start_date}_{end_date}"
            )

            # Workflow completed
            log.info(f"Study area {study_area}: Completed DEA Intertidal workflow")
            
            return freq, corr, good_mask, misclassified_ds
            
        else:
             # Calculate elevation
            ds, freq, corr, tide_m = elevation(
                study_area,
                start_date=start_date,
                end_date=end_date,
                resolution=resolution,
                crs="EPSG:3577",
                ndwi_thresh=ndwi_thresh,
                include_s2=True,
                include_ls=True,
                filter_gqa=False,
                config_path=config_path,
                log=log,
            )

            # Calculate extents
            log.info(f"Study area {study_area}: Calculating Extents layer")
            ds["extents"] = extents(freq, ds.elevation, corr)

            if exposure_offsets:
                # Calculate exposure
                log.info(f"Study area {study_area}: Calculating Exposure layer")
                all_timerange = pd.date_range(
                    start=round_date_strings(start_date, round_type="start"),
                    end=round_date_strings(end_date, round_type="end"),
                    freq=modelled_freq,
                )
                ds["exposure"], tide_cq = exposure(ds.elevation, all_timerange)

                # Calculate spread, offsets and HAT/LAT/LOT/HOT
                log.info(
                    f"Study area {study_area}: Calculating spread, offset "
                    "and HAT/LAT/LOT/HOT layers"
                )
                (
                    ds["lat"],
                    ds["hat"],
                    ds["lot"],
                    ds["hot"],
                    ds["spread"],
                    ds["offset_lowtide"],
                    ds["offset_hightide"],
                ) = bias_offset(
                    tide_m=tide_m,
                    tide_cq=tide_cq,
                    extents=ds.extents,
                    lot_hot=True,
                    lat_hat=True,
                )

                # Calculate tidelines
                log.info(
                    f"Study area {study_area}: Calculating high and low tidelines "
                    "and associated satellite offsets"
                )
                (hightideline, lowtideline, tidelines_gdf) = tidal_offset_tidelines(
                    extents=ds.extents,
                    offset_hightide=ds.offset_hightide,
                    offset_lowtide=ds.offset_lowtide,
                    distance=tideline_offset_distance,
                )

                # Export high and low tidelines and the offset data
                log.info(
                    f"Study area {study_area}: Exporting high and low tidelines with satellite offset"
                )
                hightideline.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_offset_hightide.geojson"
                )
                lowtideline.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_offset_lowtide.geojson"
                )
                tidelines_gdf.to_crs("EPSG:4326").to_file(
                    f"data/interim/{study_area}_{start_date}_{end_date}_tidelines_highlow.geojson"
                )

            else:
                log.info(
                    f"Study area {study_area}: Skipping Exposure and spread/offsets/tidelines calculation"
                )

            # Export layers as GeoTIFFs with optimised data types
            log.info(f"Study area {study_area}: Exporting outputs to GeoTIFFs")
            export_intertidal_rasters(
                ds, prefix=f"data/interim/{study_area}_{start_date}_{end_date}"
            )

            # Workflow completed
            log.info(f"Study area {study_area}: Completed DEA Intertidal workflow")

    except Exception as e:
        log.exception(f"Study area {study_area}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    intertidal_cli()


# def pixel_tide_sort(ds, tide_var="tide_height", ndwi_var="ndwi", tide_dim="tide_n"):

#     # NOT CURRENTLY USED

#     # Return indicies to sort each pixel by tide along time dim
#     sort_indices = np.argsort(ds[tide_var].values, axis=0)

#     # Use indices to sort both tide and NDWI array
#     tide_sorted = np.take_along_axis(ds[tide_var].values, sort_indices, axis=0)
#     ndwi_sorted = np.take_along_axis(ds[ndwi_var].values, sort_indices, axis=0)

#     # Update values in array
#     ds[tide_var][:] = tide_sorted
#     ds[ndwi_var][:] = ndwi_sorted

#     return (
#         ds.assign_coords(coords={tide_dim: ("time", np.linspace(0, 1, len(ds.time)))})
#         .swap_dims({"time": tide_dim})
#         .drop("time")
#     )


# def create_dask_gateway_cluster(profile="r5_L", workers=2):
#     """
#     Create a cluster in our internal dask cluster.
#     Parameters
#     ----------
#     profile : str
#         Possible values are:
#             - r5_L (2 cores, 15GB memory)
#             - r5_XL (4 cores, 31GB memory)
#             - r5_2XL (8 cores, 63GB memory)
#             - r5_4XL (16 cores, 127GB memory)
#     workers : int
#         Number of workers in the cluster.
#     """

#     try:
#         from dask_gateway import Gateway

#         gateway = Gateway()

#         # Close any existing clusters
#         if len(cluster_names) > 0:
#             print("Cluster(s) still running:", cluster_names)
#             for n in cluster_names:
#                 cluster = gateway.connect(n.name)
#                 cluster.shutdown()

#         # Connect to new cluster
#         options = gateway.cluster_options()
#         options["profile"] = profile
#         options["jupyterhub_user"] = "robbi"
#         cluster = gateway.new_cluster(options)
#         cluster.scale(workers)

#         return cluster

#     except ClientConnectionError:
#         raise ConnectionError("Access to dask gateway cluster unauthorized")


# def abslmp_gauge(
#     coords, start_year=2019, end_year=2021, data_path="data/raw/ABSLMP", plot=True
# ):
#     """
#     Loads water level data from the nearest Australian Baseline Sea Level
#     Monitoring Project gauge.
#     """

#     from shapely.ops import nearest_points
#     from shapely.geometry import Point

#     # Standardise coords format
#     if isinstance(coords, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)):
#         print("Using dataset bounds to load gauge data")
#         coords = coords.odc.geobox.geographic_extent.geom
#     elif isinstance(coords, tuple):
#         coords = Point(coords)

#     # Convert coords to GeoDataFrame
#     coords_gdf = gpd.GeoDataFrame(geometry=[coords], crs="EPSG:4326").to_crs(
#         "EPSG:3577"
#     )

#     # Load station metadata
#     site_metadata_df = pd.read_csv(
#         f"{data_path}/ABSLMP_station_metadata.csv", index_col="ID CODE"
#     )

#     # Convert metadata to GeoDataFrame
#     sites_metadata_gdf = gpd.GeoDataFrame(
#         data=site_metadata_df,
#         geometry=gpd.points_from_xy(
#             site_metadata_df.LONGITUDE, site_metadata_df.LATITUDE
#         ),
#         crs="EPSG:4326",
#     ).to_crs("EPSG:3577")

#     # Find nearest row
#     site_metadata_gdf = gpd.sjoin_nearest(coords_gdf, sites_metadata_gdf).iloc[0]
#     site_id = site_metadata_gdf["index_right"]
#     site_name = site_metadata_gdf["TOWN / DISTRICT"]

#     # Read all tide data
#     print(f"Loading ABSLMP gauge {site_id} ({site_name})")
#     available_paths = glob(f"{data_path}/{site_id}_*.csv")
#     available_years = sorted([int(i[-8:-4]) for i in available_paths])

#     loaded_data = [
#         pd.read_csv(
#             f"{data_path}/{site_id}_{year}.csv",
#             index_col=0,
#             parse_dates=True,
#             na_values=-9999,
#         )
#         for year in range(start_year, end_year)
#         if year in available_years
#     ]

#     try:
#         # Combine loaded data
#         df = pd.concat(loaded_data).rename(
#             {" Adjusted Residuals": "Adjusted Residuals"}, axis=1
#         )

#         # Extract water level and residuals
#         clean_df = df[["Sea Level", "Adjusted Residuals"]].rename_axis("time")
#         clean_df.columns = ["sea_level", "residuals"]
#         clean_df["sea_level"] = clean_df.sea_level - site_metadata_gdf.AHD
#         clean_df["sea_level_noresiduals"] = clean_df.sea_level - clean_df.residuals

#         # Summarise non-residual waterlevels by week to assess seasonality
#         seasonal_df = (
#             clean_df[["sea_level_noresiduals"]]
#             .groupby(clean_df.index.isocalendar().week)
#             .mean()
#         )

#         # Plot
#         if plot:
#             fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#             axes = axes.flatten()
#             clean_df["sea_level"].plot(ax=axes[0], lw=0.2)
#             axes[0].set_title("Water levels (AHD)")
#             axes[0].set_xlabel("")
#             clean_df["residuals"].plot(ax=axes[1], lw=0.3)
#             axes[1].set_title("Adjusted residuals")
#             axes[1].set_xlabel("")
#             clean_df["sea_level_noresiduals"].plot(ax=axes[2], lw=0.2)
#             axes[2].set_title("Water levels, no residuals (AHD)")
#             axes[2].set_xlabel("")
#             seasonal_df.plot(ax=axes[3])
#             axes[3].set_title("Seasonal")

#         return clean_df, seasonal_df

#     except ValueError:
#         print(
#             f"\nNo data for selected start and end year. Available years include:\n{available_years}"
#         )


# def abslmp_correction(ds, start_year=2010, end_year=2021):
#     """
#     Applies a seasonal correction to tide height data based on the nearest
#     Australian Baseline Sea Level Monitoring Project gauge.
#     """

#     # Load seasonal data from ABSLMP
#     _, abslmp_seasonal_df = abslmp_gauge(
#         coords=ds, start_year=start_year, end_year=end_year, plot=False
#     )

#     # Apply weekly offsets to tides
#     df_correction = abslmp_seasonal_df.loc[ds.time.dt.weekofyear].reset_index(drop=True)
#     df_correction.index = ds.time
#     da_correction = (
#         df_correction.rename_axis("time")
#         .rename({"sea_level_noresiduals": "tide_m"}, axis=1)
#         .to_xarray()
#     )
#     ds["tide_m"] = ds["tide_m"] + da_correction.tide_m

#     return ds
