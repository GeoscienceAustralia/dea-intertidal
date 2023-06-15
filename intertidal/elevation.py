import os
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


def extract_geom(study_area, config, id_col="id"):
    """
    Handles extraction of a datacube Geometry object from either a
    string or integer tile ID.

    If passed as a string or integer, a Geometry object will be
    extracted based on the extent of the relevant study area grid cell.
    If a Geometry object is passed in, it will be returned as-is.

    TODO: Refactor this func to use a Gridspec directly instead of a
    study area polygon dataset.

    Parameters
    ----------
    study_area : int, str or Geometry
        Study area polygon represented as either the ID of a tile grid
        cell, or a `datacube.utils.geometry.Geometry` object defining
        the spatial extent of interest.
    config : dict
        A loaded configuration file, used to obtain the path to the
        study area grid ("grid_path").
    id_col : str, optional
        The name of the study area grid column containing each grid
        cell ID. Defaults to "id".

    Returns
    -------
    geom : datacube.utils.geometry.Geometry
        A datacube Geometry object providing the analysis extents.
    study_area : str
        Returns either the previously provided `study_area` ID, or the
        string "custom" if a custom Geometry object is passed in.
    """
    # Load study area from tile grid if passed a string
    if isinstance(study_area, (int, str)):
        # Load study area
        gridcell_gdf = gpd.read_file(config["Input files"]["grid_path"]).set_index(
            id_col
        )
        gridcell_gdf.index = gridcell_gdf.index.astype(str)
        gridcell_gdf = gridcell_gdf.loc[[str(study_area)]]

        # Create geom as input for dc.load
        geom = Geometry(geom=gridcell_gdf.iloc[0].geometry, crs=gridcell_gdf.crs)

    # Otherwise, use supplied geom
    elif isinstance(study_area, Geometry):
        geom = study_area
        study_area = "custom"

    else:
        raise Exception(
            "Unsupported input type for `study_area`; please "
            "provide either a string, integer or dataube Geometry "
            "object."
        )

    return geom, study_area


def load_data(
    dc,
    study_area,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:3577",
    s2_prod="s2_nbart_ndwi",
    ls_prod="ls_nbart_ndwi",
    config_path="configs/dea_intertidal_config.yaml",
    filter_gqa=True,
    log=None,
):
    """
    Load cloud-masked Landsat and Sentinel-2 NDWI data for a given
    spatial and temporal extent.

    Parameters
    ----------
    dc : Datacube
        A datacube instance connected to a database.
    study_area : int, str or Geometry
        Study area polygon represented as either the ID of a tile grid
        cell, or a `datacube.utils.geometry` object defining the spatial
        extent of interest.
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
        Path to the configuration file, used to obtain the virtual
        products config to load ("virtual_product_path").
        Defaults to "configs/dea_intertidal_config.yaml".
    filter_gqa : bool, optional
        Whether or not to filter Sentinel-2 data using the GQA filter.
        Defaults to True.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Landsat and Sentinel-2
        data, converted to NDWI with cloud masking applied.
    """

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry

    if log is None:
        log = configure_logging()

    # Load product and virtual product catalogue configs
    config = load_config(config_path)
    catalog = catalog_from_file(config["Virtual product"]["virtual_product_path"])

    # Load study area geometry object, and project to match `crs`
    geom, study_area = extract_geom(study_area, config)
    geom = geom.to_crs(crs)

    # Set up query params
    query_params = {
        "geopolygon": geom,
        "time": time_range,
    }

    # Set up load params
    load_params = {
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
        s2_ds = product.load(dc, **query_params, **load_params)

        # Apply cloud mask and contiguity mask
        s2_ds_masked = s2_ds.where(s2_ds.cloud_mask == 1 & s2_ds.contiguity)
        data_list.append(s2_ds_masked)

    # If Landsat data is requested
    if ls_prod is not None:
        # Load Landsat data
        product = catalog[ls_prod]
        ls_ds = product.load(dc, **query_params, **load_params)

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
    ndwi_thresh=0.0,
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
        Default is 0.0.
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
        Two-dimensional xarray dataset with dimensions (time, z),
        containing NDWI and tide height variables.
    freq : xr.DataArray
        Frequency of wetness for each pixel (where NDWI > `ndwi_thresh`).
    corr : xr.DataArray
        Correlation of NDWI pixel wetness with tide height.
    intertidal_candidates : xr.DataArray
        Pixels identified as potential intertidal candidates for
        subsequent elevation modelling by the above frequency and
        correlation thresholds.
    """

    # Flatten satellite dataset by stacking "y" and "x" dimensions
    flat_ds = satellite_ds.stack(z=("y", "x"))

    # Calculate frequency of wet per pixel, then threshold
    # to exclude always wet and always dry
    freq = (
        (flat_ds[index] > ndwi_thresh)
        .where(~flat_ds[index].isnull())
        .mean(dim="time")
        .drop_vars("variable")
        .rename("ndwi_wet_freq")
    )
    freq_mask = (freq >= min_freq) & (freq <= max_freq)

    # Flatten to 1D, dropping any pixels that are not in frequency mask
    flat_ds = flat_ds.where(freq_mask, drop=True)

    # Calculate correlations between NDWI water observations and tide
    # height. Because we are only interested in pixels with inundation
    # patterns (e.g.transitions from dry to wet) are driven by tide, we
    # first convert NDWI into a boolean dry/wet layer before running the
    # correlation. This prevents small changes in NDWI beneath the water
    # surface from producing correlations with tide height.
    wet_dry = flat_ds[index] > ndwi_thresh
    corr = xr.corr(wet_dry, flat_ds.tide_m, dim="time").rename("ndwi_tide_corr")

    # Keep only pixels with correlations that meet min threshold
    corr_mask = corr >= min_correlation
    flat_ds = flat_ds.where(corr_mask, drop=True)

    # Return pixels identified as intertidal candidates
    intertidal_candidates = corr_mask.where(corr_mask, drop=True).rename(
        "intertidal_candidate_px"
    )

    print(
        f"Reducing analysed pixels from {freq.count().item()} to "
        f"{len(intertidal_candidates.z)} ({len(intertidal_candidates.z) * 100 / freq.count().item():.2f}%)"
    )

    return flat_ds, freq, corr, intertidal_candidates


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


def pixel_rolling_median(
    flat_ds, windows_n=100, window_prop_tide=0.15, max_workers=None
):
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

    # TODO: Implement interpolation of intervals
    # interval_ds = interval_ds.interp(interval=np.linspace(0, 56, 100), method="linear")

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


def pixel_uncertainty(
    flat_ds, flat_dem, ndwi_thresh=0.1, method="mad", min_q=0.25, max_q=0.75
):
    """
    Calculate uncertainty bounds around a modelled elevation based on
    observations that were misclassified by a given NDWI threshold.

    The function identifies observations that were misclassified by the
    modelled elevation, i.e., wet observations (NDWI > threshold) at
    lower tide heights than the modelled elevation, or dry observations
    (NDWI < threshold) at higher tide heights than the modelled
    elevation.

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
    method : string, optional
        Whether to calculate uncertainty using Median Absolute Deviation
        (MAD) of the tide heights of all misclassified points, or by
        taking upper/lower tide height quantiles of miscalssified points.
        Defaults to "mad" for Median Absolute Deviation; use "quantile"
        to use quantile calculation instead.
    min_q, max_q : float, optional
        If `method == "quantile": the minimum and maximum quantiles used
        to estimate uncertainty bounds based on misclassified points.
        Defaults to interquartile range, or 0.25, 0.75. This provides a
        balance between capturing the range of uncertainty at each
        pixel, while not being overly influenced by outliers in `flat_ds`.

    Returns
    -------
    dem_flat_low, dem_flat_high, dem_flat_uncertainty : xarray.DataArray
        The lower and upper uncertainty bounds around the modelled
        elevation, and the summary uncertainty range between them.
    misclassified_sum : xarray.DataArray
        The sum of individual satellite observations misclassified by
        the modelled elevation and NDWI threshold.
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

    # Calculate uncertainty by taking the Median Absolute Deviation of
    # all misclassified points.
    if method == "mad":
        # Calculate median of absolute deviations
        # TODO: Account for large MAD on pixels with very few
        # misclassified points. Set < X misclassified points to 0 MAD?
        mad = abs(misclassified_ds.tide_m - flat_dem.elevation).median(dim="time")

        # Calculate low and high bounds
        uncertainty_low = flat_dem.elevation - mad
        uncertainty_high = flat_dem.elevation + mad

    # Calculate interquartile tide height range of our misclassified
    # observations to obtain lower and upper uncertainty bounds around our
    # modelled elevation.
    elif method == "quantile":
        # Use xr_quantile (faster than built-in .quantile)
        misclassified_q = xr_quantile(
            src=misclassified_ds.dropna(dim="time", how="all")[["tide_m"]],
            quantiles=[min_q, max_q],
            nodata=np.nan,
        ).tide_m.fillna(flat_dem.elevation)

        # Extract low and high bounds
        uncertainty_low = misclassified_q.sel(quantile=min_q, drop=True)
        uncertainty_high = misclassified_q.sel(quantile=max_q, drop=True)

    # Clip min and max uncertainty to modelled elevation to ensure lower
    # bounds are not above modelled elevation (and vice versa)
    dem_flat_low = np.minimum(uncertainty_low, flat_dem.elevation)
    dem_flat_high = np.maximum(uncertainty_high, flat_dem.elevation)

    # Subtract low from high DEM to summarise uncertainy range
    dem_flat_uncertainty = dem_flat_high - dem_flat_low

    # Calculate sum of misclassified points
    misclassified_sum = (
        misclassified_all.sum(dim="time")
        .rename("misclassified_px_count")
        .where(~flat_dem.elevation.isnull())
    )

    return (
        dem_flat_low,
        dem_flat_high,
        dem_flat_uncertainty,
        misclassified_sum,
    )


def flat_to_ds(flat_ds, template, stacked_dim="z"):
    """
    Convert a flattened xarray Dataset with a stacked dimension to its
    original spatial dimensions, based on a given template.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened xarray.Dataset, i.e., a dataset where each y/x
        pixel is stacked into a single "z" dimension.
    template : xarray.Dataset or xarray.Dataarray
        A dataset  containing the original spatial dimensions and
        coordinates of the data, used as a template to reshape the
        flattened data back to the spatial dimensions.
    stacked_dim : str, optional
        The name of the stacked y/x dimension in the flattened dataset.
        The default is "z".

    Returns
    -------
    xarray.Dataset
        The unflattened xarray Dataset, with the same spatial dimensions
        (e.g. y/x) as the template.

    Notes
    -----
    The function unstacks the flattened dataset along the stacked
    dimension, reindexes the resulting dataset to match the coordinates
    of the template, and transposes the dimensions to match the order of
    the template's spatial y/x dimensions.
    """

    unstacked_ds = (
        # First, unstack back into y/x dimensions
        flat_ds.unstack(stacked_dim)
        # After unstacking, our output can be missing entire y/x
        # coordinates contained in `template`. To address this, we need
        # to "reindex" our unstacked data so that it has exactly the
        # same coordinates as `template`. Affected pixels will be filled
        # with np.nan
        .reindex_like(template)
        # Finally, we ensure that our spatial y/x dimensions have not
        # been rotated during the unstack. The `...` preserves any extra
        # non-spatial dimensions (like "time") if they exist
        .transpose(..., *template.odc.spatial_dims)
    )

    return unstacked_ds


def elevation(
    satellite_ds,
    ndwi_thresh=0.1,
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.2,
    windows_n=100,
    window_prop_tide=0.15,
    max_workers=None,
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
    config_path="configs/dea_intertidal_config.yaml",
    study_area=None,
    log=None,
):
    """
    Calculates DEA Intertidal Elevation using satellite imagery and
    tidal modeling.

    Parameters
    ----------
    satellite_ds : xarray.Dataset
        A satellite data time series containing an "ndwi" water index
        variable.
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI) above which pixels are considered water, by default 0.1.
    min_freq, max_freq : float, optional
        Minimum and maximum frequency of wetness required for a pixel to
        be included in the analysis, by default 0.01 and 0.99.
    min_correlation : float, optional
        Minimum correlation between water index and tide height required
        for a pixel to be included in the analysis, by default 0.2.
    windows_n : int, optional
        Number of rolling windows to iterate over in the per-pixel
        rolling median calculation, by default 100
    window_prop_tide : float, optional
        Proportion of the tide range to use for each window radius in
        the per-pixel rolling median calculation, by default 0.15
    max_workers : int, optional
        Maximum number of worker processes to use for parallel execution
        in the per-pixel rolling median calculation. Defaults to None,
        which uses built-in methods from `concurrent.futures` to
        determine workers.
    tide_model : str, optional
        The tide model used to model tides, as supported by the `pyTMD`
        Python package. Options include:
        - "FES2014" (default; pre-configured on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `dea_tools.coastal.model_tides`.
    config_path : str, optional
        Path to the configuration file, by default
        "configs/dea_intertidal_config.yaml".
    study_area : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    ds : xarray.Dataset
        A dataset containing intertidal elevation and
        confidence values for each pixel in the study area.
    ds_aux : xarray.Dataset
        A dataset containg auxiliary layers used for subsequent
        workflows and debugging. These include information about the
        frequency of inundation for each pixel, correlations between
        NDWI and tide height, the number of misclassified observations
        resulting from the modelled elevation value, and the intertidal
        candidate pixels passed to the elevation modelling code.
    tide_m : xarray.DataArray
        An xarray.DataArray object containing the modeled tide
        heights for each pixel in the study area.
    """

    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use study area name for logs if it exists
    if study_area is not None:
        log_prefix = f"Study area {study_area}: "
    else:
        log_prefix = ""

    # Model tides into every pixel in the three-dimensional (x by y by
    # time) satellite dataset
    log.info(f"{log_prefix}Modelling tide heights for each pixel")
    tide_m, _ = pixel_tides(
        satellite_ds, resample=True, model=tide_model, directory=tide_model_dir
    )

    # Set tide array pixels to nodata if the satellite data array pixels
    # contain nodata. This ensures that we ignore any tide observations
    # where we don't have matching satellite imagery
    log.info(
        f"{log_prefix}Masking nodata and adding tide heights to satellite data array"
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
        f"{log_prefix}Flattening satellite data array and filtering to intertidal candidate pixels"
    )
    flat_ds, freq, corr, intertidal_candidates = ds_to_flat(
        satellite_ds,
        min_freq=min_freq,
        max_freq=max_freq,
        min_correlation=min_correlation,
    )

    # Calculate per-pixel rolling median.
    log.info(f"{log_prefix}Running per-pixel rolling median")
    interval_ds = pixel_rolling_median(
        flat_ds,
        windows_n=windows_n,
        window_prop_tide=window_prop_tide,
        max_workers=max_workers,
    )

    # Model intertidal elevation
    log.info(f"{log_prefix}Modelling intertidal elevation")
    flat_dem = pixel_dem(interval_ds, flat_ds, ndwi_thresh)

    # Model intertidal elevation uncertainty
    log.info(f"{log_prefix}Modelling intertidal uncertainty")
    (
        elevation_low,
        elevation_high,
        elevation_uncertainty,
        misclassified,
    ) = pixel_uncertainty(flat_ds, flat_dem, ndwi_thresh)

    # Add uncertainty array to dataset
    # TODO: decide whether we want to also keep low and high bounds
    flat_dem["elevation_uncertainty"] = elevation_uncertainty

    # Combine auxiliary layers into a new auxilary dataset. Using
    # `xr.combine_by_coords` is required because each of our debug
    # layers have different lengths/coordinates along the "z" dimension
    flat_ds_aux = xr.combine_by_coords(
        [freq, corr, intertidal_candidates, misclassified],
        fill_value={"intertidal_candidates": False},
    )

    # Unstack all layers back into their original spatial dimensions
    log.info(f"{log_prefix}Unflattening data back to its original spatial dimensions")
    ds = flat_to_ds(flat_dem, satellite_ds)
    ds_aux = flat_to_ds(flat_ds_aux, satellite_ds)

    # Return master dataset and debug dataset
    log.info(f"{log_prefix}Successfully completed intertidal elevation modelling")

    return ds, ds_aux, tide_m


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
    "--min_freq",
    type=float,
    default=0.01,
    help="Minimum frequency of wetness required for a pixel to be "
    "included in the analysis, by default 0.01.",
)
@click.option(
    "--max_freq",
    type=float,
    default=0.99,
    help="Maximum frequency of wetness required for a pixel to be "
    "included in the analysis, by default 0.99.",
)
@click.option(
    "--min_correlation",
    type=float,
    default=0.2,
    help="Minimum correlation between water index and tide height "
    "required for a pixel to be included in the analysis, by default "
    "0.2.",
)
@click.option(
    "--windows_n",
    type=int,
    default=100,
    help="Number of rolling windows to iterate over in the per-pixel "
    "rolling median calculation, by default 100.",
)
@click.option(
    "--window_prop_tide",
    type=float,
    default=0.15,
    help="Proportion of the tide range to use for each window radius "
    "in the per-pixel rolling median calculation, by default 0.15.",
)
@click.option(
    "--tide_model",
    type=str,
    default="FES2014",
    help="The tide model used to model tides, as supported by the "
    "`pyTMD` Python package. Options include 'FES2014' (default), "
    "'TPXO8-atlas' and 'TPXO9-atlas-v5'.",
)
@click.option(
    "--tide_model_dir",
    type=str,
    default="/var/share/tide_models",
    help="The directory containing tide model data files. Defaults to "
    "'/var/share/tide_models'; for more information about the required "
    "directory structure, refer to `dea_tools.coastal.model_tides`.",
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
    "is set to 500 m.",
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
    "--output_auxiliaries",
    is_flag=True,
    default=False,
    help="Whether to output auxiliary files for debugging. Defaults to "
    "False; can be set to True by passing `--output_auxiliaries`.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    is_flag=True,
    default=True,
    help="Whether to sign AWS requests for S3 access. Defaults to "
    "True; can be set to False by passing `--no-aws_unsigned`.",
)
def intertidal_cli(
    config_path,
    study_area,
    start_date,
    end_date,
    resolution,
    ndwi_thresh,
    min_freq,
    max_freq,
    min_correlation,
    windows_n,
    window_prop_tide,
    tide_model,
    tide_model_dir,
    modelled_freq,
    tideline_offset_distance,
    exposure_offsets,
    output_auxiliaries,
    aws_unsigned,
):
    log = configure_logging(f"Intertidal processing for study area {study_area}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    # Create output folder. If it doesn't exist, create it
    output_dir = f"data/interim/{study_area}/{start_date}-{end_date}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        log.info(f"Study area {study_area}: Loading satellite data")

        # Connect to datacube to load data
        dc = datacube.Datacube(app="Intertidal_CLI")

        # Create local dask cluster to improve data load time
        client = create_local_dask_cluster(return_client=True)

        satellite_ds = load_data(
            dc=dc,
            study_area=study_area,
            time_range=(start_date, end_date),
            resolution=resolution,
            crs="EPSG:3577",
            s2_prod="s2_nbart_ndwi",
            ls_prod="ls_nbart_ndwi",
            filter_gqa=False,
            config_path=config_path,
        )[["ndwi"]]

        # Load data and close dask client
        satellite_ds.load()
        client.close()

        # Calculate elevation
        log.info(f"Study area {study_area}: Calculating Intertidal Elevation")
        ds, ds_aux, tide_m = elevation(
            satellite_ds,
            ndwi_thresh=ndwi_thresh,
            min_freq=min_freq,
            max_freq=max_freq,
            min_correlation=min_correlation,
            windows_n=windows_n,
            window_prop_tide=window_prop_tide,
            tide_model=tide_model,
            tide_model_dir=tide_model_dir,
            config_path=config_path,
            study_area=study_area,
            log=log,
        )

        # Calculate extents
        log.info(f"Study area {study_area}: Calculating Intertidal Extents")
        ds["extents"] = extents(
            ds_aux.ndwi_wet_freq, ds.elevation, ds_aux.ndwi_tide_corr
        )

        if exposure_offsets:
            log.info(f"Study area {study_area}: Calculating Intertidal Exposure")

            # Set time range
            all_timerange = pd.date_range(
                start=round_date_strings(start_date, round_type="start"),
                end=round_date_strings(end_date, round_type="end"),
                freq=modelled_freq,
            )

            # Calculate exposure
            ds["exposure"], tide_cq = exposure(
                dem=ds.elevation,
                time_range=all_timerange,
                tide_model=tide_model,
                tide_model_dir=tide_model_dir,
            )

            # Calculate spread, offsets and HAT/LAT/LOT/HOT
            log.info(
                f"Study area {study_area}: Calculating spread, offset "
                "and HAT/LAT/LOT/HOT layers"
            )
            (
                ds["oa_lat"],
                ds["oa_hat"],
                ds["oa_lot"],
                ds["oa_hot"],
                ds["oa_spread"],
                ds["oa_offset_lowtide"],
                ds["oa_offset_hightide"],
            ) = bias_offset(
                tide_m=tide_m,
                tide_cq=tide_cq,
                extents=ds.extents,
                lot_hot=True,
                lat_hat=True,
            )

        #             # Calculate tidelines
        #             log.info(
        #                 f"Study area {study_area}: Calculating high and low tidelines "
        #                 "and associated satellite offsets"
        #             )
        #             (hightideline, lowtideline, tidelines_gdf) = tidal_offset_tidelines(
        #                 extents=ds.extents,
        #                 offset_hightide=ds.oa_offset_hightide,
        #                 offset_lowtide=ds.oa_offset_lowtide,
        #                 distance=tideline_offset_distance,
        #             )

        #             # Export high and low tidelines and the offset data
        #             log.info(
        #                 f"Study area {study_area}: Exporting high and low tidelines with satellite offset to {output_dir}"
        #             )
        #             hightideline.to_crs("EPSG:4326").to_file(
        #                 f"{output_dir}/{study_area}_{start_date}_{end_date}_offset_hightide.geojson"
        #             )
        #             lowtideline.to_crs("EPSG:4326").to_file(
        #                 f"{output_dir}/{study_area}_{start_date}_{end_date}_offset_lowtide.geojson"
        #             )
        #             tidelines_gdf.to_crs("EPSG:4326").to_file(
        #                 f"{output_dir}/{study_area}_{start_date}_{end_date}_tidelines_highlow.geojson"
        #             )

        else:
            log.info(
                f"Study area {study_area}: Skipping Exposure and spread/offsets/tidelines calculation"
            )

        # Export layers as GeoTIFFs with optimised data types
        log.info(f"Study area {study_area}: Exporting output GeoTIFFs to {output_dir}")
        export_intertidal_rasters(
            ds, prefix=f"{output_dir}/DEV_{study_area}_{start_date}_{end_date}"
        )

        if output_auxiliaries:
            # Export auxiliary debug layers as GeoTIFFs with optimised data types
            log.info(
                f"Study area {study_area}: Exporting debugging GeoTIFFs to {output_dir}"
            )
            export_intertidal_rasters(
                ds_aux,
                prefix=f"{output_dir}/DEV_{study_area}_{start_date}_{end_date}_debug",
            )

        # Workflow completed
        log.info(f"Study area {study_area}: Completed DEA Intertidal workflow")

    except Exception as e:
        log.exception(f"Study area {study_area}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    intertidal_cli()
