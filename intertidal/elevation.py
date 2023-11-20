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
from odc.geo.geom import Geometry
from odc.geo.geobox import GeoBox
from odc.geo.gridspec import GridSpec
from odc.geo.types import xy_
from odc.algo import (
    mask_cleanup,
    xr_quantile,
    enum_to_bool,
    keep_good_only,
    erase_bad,
    to_f32,
)
from datacube.utils.aws import configure_s3_access

from dea_tools.coastal import pixel_tides, glint_angle, _pixel_tides_resample
from dea_tools.dask import create_local_dask_cluster
from dea_tools.spatial import interpolate_2d
from dea_tools.temporal import lag_linregress_3D

from intertidal.utils import (
    configure_logging,
    round_date_strings,
    export_intertidal_rasters,
)
from intertidal.extents import extents
from intertidal.exposure import exposure
from intertidal.tidal_bias_offset import bias_offset, tidal_offset_tidelines


def extract_geobox(
    study_area=None,
    geom=None,
    resolution=10,
    crs="EPSG:3577",
    tile_width=32000,
    gridspec_origin_x=-2688000,
    gridspec_origin_y=-5472000,
):
    """
    Handles extraction of a GeoBox pixel grid from either a GridSpec
    tile ID (in the form 'x143y56'), or a provided Geometry object.

    If a tile ID string is passed to `study_area`, a GeoBox will be
    extracted based on relevant GridSpec tile. If a custom Geometry
    object is passed using `geom`, it will be converted to a GeoBox.

    (Either `study_area` or `geom` is required; `geom` will override
    `study_area` if provided).

    Parameters
    ----------
    study_area : str, optional
        Tile ID string to process. This should be the ID of a GridSpec
        analysis tile in the format "x143y56". If `geom` is provided,
        this will have no effect.
    geom : Geometry, optional
        A datacube Geometry object defining a custom spatial extent of
        interest. If `geom` is provided, this will overrule any study
        area ID passed to `study_area` and will be returned as-is.
    resolution : int, optional
        The desired resolution of the GeoBox grid, in units of the
        coordinate reference system (CRS). Defaults to 10.
    crs : str, optional
        The coordinate reference system (CRS) to use for the GeoBox.
        Defaults to "EPSG:3577".
    tile_width : int, optional
        The width of a GridSpec tile, in units of the coordinate
        reference system (CRS). Defaults to 32000 metres.
    gridspec_origin_x : int, optional
        The x-coordinate of the origin (bottom-left corner) of the
        GridSpec tile grid. Defaults to -2688000.
    gridspec_origin_y : int, optional
        The y-coordinate of the origin (bottom-left corner) of the
        GridSpec tile grid. Defaults to -5472000.

    Returns
    -------
    geobox : odc.geo.geobox.GeoBox
        A GeoBox defining the pixel grid to use to load data (defining
        the CRS, resolution, shape and extent of the study area).
    """

    def _id_to_tuple(id_str):
        """
        Converts a tile ID in form 'x143y56' to a ix, iy tuple so it
        can be passed to a GridSpec (e.g. `gs[ix, iy]`)
        """
        try:
            ix, iy = id_str.replace("x", "").split("y")
            return int(ix), int(iy)
        except ValueError:
            raise ValueError(
                "Supplied study area ID is not in the form 'x143y56'. If "
                "you meant to provide an ID matching a feature from a "
                "custom vector file, make sure you run the 'Optional: "
                "load study area from vector file' notebook cell."
            )

    # List of valid input geometry types (from `odc-geo` or `datacube-core`)
    GEOM_TYPES = (odc.geo.geom.Geometry, datacube.utils.geometry._base.Geometry)

    # Either `study_area` or `geom` must be provided
    if study_area is None and geom is None:
        raise ValueError(
            "Please provide either a study area ID (using `study_area`), "
            "or a datacube Geometry object (using `geom`)."
        )

    # If custom geom is provided, verify it is a geometry
    elif geom is not None and not isinstance(geom, GEOM_TYPES):
        raise ValueError(
            "Unsupported input type for `geom`; please provide a "
            "datacube Geometry object."
        )

    # Otherwise, extract GeoBox from geometry
    elif geom is not None and isinstance(geom, GEOM_TYPES):
        geobox = GeoBox.from_geopolygon(geom, crs=crs, resolution=resolution)

    # If no custom geom provided, load tile from GridSpec tile grid
    elif geom is None:
        # Verify that resolution fits evenly inside tile width
        if tile_width % resolution != 0:
            raise ValueError(
                "Ensure that `resolution` divides into `tile_width` evenly."
            )

        # Calculate tile pixels
        n_pixels = tile_width / resolution

        # Create GridSpec tile grid
        gs = GridSpec(
            crs=crs,
            resolution=resolution,
            tile_shape=(n_pixels, n_pixels),
            origin=xy_(gridspec_origin_x, gridspec_origin_y),
        )

        # Extract GeoBox from GridSpec
        geobox = gs[_id_to_tuple(study_area)]

    return geobox


def load_data(
    dc,
    study_area=None,
    geom=None,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:3577",
    include_s2=True,
    include_ls=True,
    filter_gqa=True,
    max_cloudcover=90,
    ndwi=True,
    mask_sunglint=None,
    dask_chunks=None,
    dtype="float32",
    log=None,
    **query,
):
    """
    Loads cloud-masked Sentinel-2 and Landsat satellite data for a given
    study area/geom and time range.

    Supports optionally converting to Normalised Difference Water Index
    and masking sunglinted pixels.

    Parameters
    ----------
    dc : datacube.Datacube()
        A datacube instance to load data from.
    study_area : str, optional
        Tile ID string to process. This should be the ID of a GridSpec
        analysis tile in the format "x143y56". If `geom` is provided,
        this will have no effect.
    geom : Geometry, optional
        A datacube Geometry object defining a custom spatial extent of
        interest. If `geom` is provided, this will overrule any study
        area ID passed to `study_area` and will be returned as-is.
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
    include_s2 : bool, optional
        Whether to load Sentinel-2 data.
    include_ls : bool, optional
        Whether to load Landsat data.
    filter_gqa : bool, optional
        Whether or not to filter Sentinel-2 data using the GQA filter.
        Defaults to True.
    max_cloudcover : float, optional
        The maximum cloud cover metadata value used to load data.
        Defaults to 90 (i.e. 90% cloud cover).
    ndwi : bool, optional
        Whether to convert spectral bands to Normalised Difference Water
        Index values before returning them. Note that this must be set
        to True if both `include_s2` and `include_ls` are True.
    mask_sunglint : int, optional
        EXPERIMENTAL: Whether to mask out pixels that are likely to be
        affected by sunglint using glint angles. Low glint angles
        (e.g. < 20) often correspond with sunglint. Defaults to None;
        set to e.g. "20" to mask out all pixels with a glint angle of
        less than 20.
    dask_chunks : dict, optional
        Optional custom Dask chunks to load data with. Defaults to None,
        which will use '{"x": 1600, "y": 1600}'.
    dtype : str, optional
        Desired data type for output data. Valid values are "int16"
        (default) and "float32". If `ndwi=True`, then "float32" will be
        used regardless of what is set here (as nodata values must be
        set to 'NaN' before calculating NDWI).
    **query :
        Optional datacube.load keyword argument parameters used to
        query data.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Landsat or Sentinel-2
        data.
    """

    # Set spectral bands to load
    s2_spectral_bands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_red_edge_1",
        "nbart_red_edge_2",
        "nbart_red_edge_3",
        "nbart_nir_1",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
    ]
    ls_spectral_bands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir",
        "nbart_swir_1",
        "nbart_swir_2",
    ]

    # Set masking bands to load
    s2_masking_bands = ["oa_s2cloudless_mask", "oa_nbart_contiguity"]
    ls_masking_bands = ["oa_fmask", "oa_nbart_contiguity"]

    # Set sunglint bands to load
    if mask_sunglint is not None:
        sunglint_bands = [
            "oa_solar_zenith",
            "oa_solar_azimuth",
            "oa_satellite_azimuth",
            "oa_satellite_view",
        ]
    else:
        sunglint_bands = []

    # Load study area, defined as a GeoBox pixel grid
    geobox = extract_geobox(
        study_area=study_area, geom=geom, resolution=resolution, crs=crs
    )

    # Set up query params
    query_params = {
        "like": geobox.compat,  # Load into the exact GeoBox pixel grid
        "time": time_range,
        **query,  # Optional additional query parameters
    }

    # Set up load params
    load_params = {
        "group_by": "solar_day",
        "dask_chunks": {"x": 1600, "y": 1600} if dask_chunks is None else dask_chunks,
        "resampling": {
            "*": "cubic",
            "oa_fmask": "nearest",
            "oa_s2cloudless_mask": "nearest",
        },
    }

    # Optionally add GQA
    # TODO: Remove once Sentinel-2 GQA issue is resolved
    if filter_gqa:
        query_params["gqa_iterative_mean_xy"] = (0, 1)

    # Output data
    data_list = []

    # If Sentinel-2 data is requested
    if include_s2:
        ds_s2 = dc.load(
            product=["ga_s2am_ard_3", "ga_s2bm_ard_3"],
            measurements=s2_spectral_bands + s2_masking_bands + sunglint_bands,
            s2cloudless_cloud=(0, max_cloudcover),
            **query_params,
            **load_params,
        )

        # Create cloud mask, treating nodata and clouds as bad pixels
        cloud_mask = enum_to_bool(
            mask=ds_s2.oa_s2cloudless_mask, categories=["nodata", "cloud"]
        )

        # Identify non-contiguous pixels
        noncontiguous_mask = enum_to_bool(ds_s2.oa_nbart_contiguity, categories=[False])

        # Set cloud mask and non-contiguous pixels to nodata
        combined_mask = cloud_mask | noncontiguous_mask
        ds_s2 = erase_bad(
            x=ds_s2[s2_spectral_bands + sunglint_bands], where=combined_mask
        )

        # Optionally, apply sunglint mask
        if mask_sunglint is not None:
            # Calculate glint angle
            glint_array = glint_angle(
                solar_azimuth=ds_s2.oa_solar_azimuth,
                solar_zenith=ds_s2.oa_solar_zenith,
                view_azimuth=ds_s2.oa_satellite_azimuth,
                view_zenith=ds_s2.oa_satellite_view,
            )

            # Apply glint angle threshold and set affected pixels to nodata
            glint_mask = glint_array > mask_sunglint
            ds_s2 = keep_good_only(x=ds_s2[s2_spectral_bands], where=glint_mask)

        # Optionally convert to float, setting all nodata pixels to `np.nan`
        # (required for NDWI, so will be applied even if `dtype="int16"`)
        if (dtype == "float32") or ndwi:
            ds_s2 = to_f32(ds_s2)

        # Convert to NDWI
        if ndwi:
            # Calculate NDWI
            ds_s2["ndwi"] = (ds_s2.nbart_green - ds_s2.nbart_nir_1) / (
                ds_s2.nbart_green + ds_s2.nbart_nir_1
            )
            data_list.append(ds_s2[["ndwi"]])
        else:
            data_list.append(ds_s2)

    # If Landsat data is requested
    if include_ls:
        ds_ls = dc.load(
            product=[
                "ga_ls5t_ard_3",
                "ga_ls7e_ard_3",
                "ga_ls8c_ard_3",
                "ga_ls9c_ard_3",
            ],
            measurements=ls_spectral_bands + ls_masking_bands + sunglint_bands,
            cloud_cover=(0, max_cloudcover),
            **query_params,
            **load_params,
        )

        # First, we identify all bad pixels: nodata, cloud and shadow.
        # We then apply morphological opening to clean up narrow false
        # positive clouds (e.g. bright sandy beaches). By including
        # nodata, we make sure that small areas of cloud next to Landsat
        # 7 SLC-off nodata gaps are not accidently removed (at the cost
        # of not being able to clean false positives next to SLC-off gaps)
        bad_data = enum_to_bool(
            ds_ls.oa_fmask, categories=["nodata", "cloud", "shadow"]
        )
        bad_data_cleaned = mask_cleanup(bad_data, mask_filters=[("opening", 5)])

        # We now dilate ONLY pixels in our cleaned bad data dask that
        # are outside of our iriginal nodata pixels. This ensures that
        # Landsat 7 SLC-off nodata stripes are not also dilated.
        nodata_mask = enum_to_bool(ds_ls.oa_fmask, categories=["nodata"])
        bad_data_mask = mask_cleanup(
            mask=bad_data_cleaned & ~nodata_mask,
            mask_filters=[("dilation", 5)],
        )

        # Identify non-contiguous pixels
        noncontiguous_mask = enum_to_bool(ds_ls.oa_nbart_contiguity, categories=[False])

        # Set cleaned bad pixels and non-contiguous pixels to nodata
        combined_mask = bad_data_mask | noncontiguous_mask
        ds_ls = erase_bad(ds_ls[ls_spectral_bands + sunglint_bands], combined_mask)

        # Optionally, apply sunglint mask
        if mask_sunglint is not None:
            # Calculate glint angle
            glint_array = glint_angle(
                solar_azimuth=ds_ls.oa_solar_azimuth,
                solar_zenith=ds_ls.oa_solar_zenith,
                view_azimuth=ds_ls.oa_satellite_azimuth,
                view_zenith=ds_ls.oa_satellite_view,
            )

            # Apply glint angle threshold and set affected pixels to nodata
            glint_mask = glint_array > mask_sunglint
            ds_ls = keep_good_only(x=ds_ls[ls_spectral_bands], where=glint_mask)

        # Optionally convert to float, setting all nodata pixels to `np.nan`
        # (required for NDWI, so will be applied even if `dtype="int16"`)
        if (dtype == "float32") or ndwi:
            ds_ls = to_f32(ds_ls)

        # Convert to NDWI
        if ndwi:
            # Calculate NDWI
            ds_ls["ndwi"] = (ds_ls.nbart_green - ds_ls.nbart_nir) / (
                ds_ls.nbart_green + ds_ls.nbart_nir
            )
            data_list.append(ds_ls[["ndwi"]])
        else:
            data_list.append(ds_ls)

    # Combine into a single ds, sort and drop no longer needed bands
    satellite_ds = xr.concat(data_list, dim="time").sortby("time")

    return satellite_ds


def load_topobathy(
    dc,
    satellite_ds,
    product="ga_multi_ausbath_0",
    resampling="bilinear",
    mask_invalid=True,
):
    """
    Loads a topo-bathymetric DEM for the extents of the loaded satellite
    data. This is used as a coarse mask to constrain the analysis to the
    coastal zone, improving run time and reducing clear false positives.

    Parameters
    ----------
    dc : Datacube
        A Datacube instance for loading data.
    satellite_ds : ndarray
        The loaded satellite data, used to obtain the spatial extents
        of the data.
    product : str, optional
        The name of the topo-bathymetric DEM product to load from the
        datacube. Defaults to "ga_multi_ausbath_0".
    resampling : str, optional
        The resampling method to use, by default "bilinear".
    mask_invalid : bool, optional
        Whether to mask invalid/nodata values in the array by setting
        them to NaN, by default True.

    Returns
    -------
    topobathy_ds : xarray.Dataset
        The loaded topo-bathymetric DEM.
    """
    from datacube.utils.masking import mask_invalid_data

    topobathy_ds = dc.load(
        product=product, like=satellite_ds.odc.geobox.compat, resampling=resampling
    ).squeeze("time")

    # Mask invalid data
    if mask_invalid:
        topobathy_ds = mask_invalid_data(topobathy_ds)

    return topobathy_ds


def pixel_tides_ensemble(
    satellite_ds,
    directory,
    ancillary_points,
    top_n=3,
    models=None,
    interp_method="nearest",
):
    """
    Generate an ensemble tide model, choosing the best three tide models
    for any coastal location using ancillary point data (e.g. altimetry
    observations or NDWI correlations along the coastline).

    This function generates an ensemble of tidal height predictions for
    each pixel in a satellite dataset. Firstly, tides from multiple tide
    models are modelled into a low resolution grid using `pixel_tides`.
    Ancillary point data is then loaded and interpolated to the same
    grid to serve as weightings. These weightings are used to retain
    only the top three tidal models, and remaining top models are
    combined into a single ensemble output for each time/x/y.
    The resulting ensemble tides are then resampled and reprojected to
    match the high-resolution satellite data.

    Parameters:
    -----------
    satellite_ds : xarray.Dataset
        Three-dimensional dataset containing satellite-derived
        information (x by y by time).
    directory : str
        Directory containing tidal model data; see `pixel_tides`.
    ancillary_points : str
        Path to a file containing point correlations for different tidal
        models.
    top_n : integer, optional
        The number of top models to use in the ensemble calculation.
        Default is 3, which will calculate a median of the top 3 models.
    models : list or None, optional
        An optional list of tide models to use for the ensemble model.
        Default is None, which will use "FES2014", "FES2012", "EOT20",
        "TPXO8-atlas-v1", "TPXO9-atlas-v5", "HAMTIDE11", "GOT4.10".
    interp_method : str, optional
        Interpolation method used to interpolate correlations onto the
        low-resolution tide grid. Default is "nearest".

    Returns:
    --------
    tides_highres : xarray.Dataset
        High-resolution ensemble tidal heights dataset.
    weights_ds : xarray.Dataset
        Dataset containing weights for each tidal model used in the ensemble.
    """
    # Use default models if none provided
    if models is None:
        models = [
            "FES2014",
            "FES2012",
            "TPXO8-atlas-v1",
            "TPXO9-atlas-v5",
            "EOT20",
            "HAMTIDE11",
            "GOT4.10",
        ]

    # Model tides into every pixel in the three-dimensional
    # (x by y by time) satellite dataset
    tide_lowres = pixel_tides(
        satellite_ds,
        resample=False,
        model=models,
        directory=directory,
    )

    # Load ancillary points from file, reproject to match satellite
    # data, and drop empty points
    print("Generating ensemble tide model from point inputs")
    corr_gdf = (
        gpd.read_file(ancillary_points)[models + ["geometry"]]
        .to_crs(satellite_ds.odc.crs)
        .dropna()
    )

    # Loop through each model, interpolating correlations into
    # low-res tide grid
    out_list = []

    for model in models:
        out = interpolate_2d(
            tide_lowres,
            x_coords=corr_gdf.geometry.x,
            y_coords=corr_gdf.geometry.y,
            z_coords=corr_gdf[model],
            method=interp_method,
        ).expand_dims({"tide_model": [model]})

        out_list.append(out)

    # Combine along tide model dimension into a single xarray.Dataset
    weights_ds = xr.concat(out_list, dim="tide_model")

    # Mask out all but the top N models, then take median of remaining
    # to produce a single ensemble output for each time/x/y
    tide_lowres_ensemble = tide_lowres.where(
        (weights_ds.rank(dim="tide_model") > (len(models) - top_n))
    ).median("tide_model")

    # Resample/reproject ensemble tides to match high-res satellite data
    tides_highres, tides_lowres = _pixel_tides_resample(
        tides_lowres=tide_lowres_ensemble,
        ds=satellite_ds,
    )

    return tides_highres, weights_ds


def ds_to_flat(
    satellite_ds,
    ndwi_thresh=0.0,
    index="ndwi",
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.15,
    valid_mask=None,
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
        0.15.
    valid_mask : xr.DataArray, optional
        A boolean mask used to optionally constrain the analysis area,
        with the same spatial dimensions as `satellite_ds`. For example,
        this could be a mask generated from a topo-bathy DEM, used to
        limit the analysis to likely intertidal pixels. Default is None,
        which will not apply a mask.

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

    # If an overall valid data mask is provided, apply to the data first
    if valid_mask is not None:
        satellite_ds = satellite_ds.where(valid_mask)

    # Flatten satellite dataset by stacking "y" and "x" dimensions, then
    # drop any pixels that are empty across all-of-time
    flat_ds = satellite_ds.stack(z=("y", "x")).dropna(dim="time", how="all")

    # Calculate frequency of wet per pixel, then threshold
    # to exclude always wet and always dry
    freq = (
        (flat_ds[index] > ndwi_thresh)
        .where(~flat_ds[index].isnull())
        .mean(dim="time")
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
    corr = lag_linregress_3D(x=flat_ds.tide_m, y=wet_dry).cor.rename("ndwi_tide_corr")

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
    interval_ds : xarray.Dataset
        An two dimensional (interval, z) xarray.Dataset containing
        rolling medians for each pixel along intervals from low to high
        tide.
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


def pixel_dem(interval_ds, ndwi_thresh=0.1, interp_intervals=200, smooth_radius=10):
    """
    Calculates an estimate of intertidal elevation based on satellite
    imagery and tide data. Elevation is modelled by identifying the
    tide height at which a pixel transitions from dry to wet; calculated
    here as the maximum tide at which a rolling median of NDWI is
    characterised as land (e.g. NDWI <= `ndwi_thresh`).

    This function can additionally interpolate to a higher number of
    intertidal intervals and/or apply a rolling mean to smooth data
    before the elevation extraction. This can produce a cleaner output.

    Parameters
    ----------
    interval_ds : xarray.Dataset
        A flattened 2D xarray Dataset containing the rolling median for
        each pixel from low to high tide for the given area, with
        variables 'tide_m' and 'ndwi'.
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI), above which pixels are considered water. Defaults to
        0.1, which appears to more reliably capture the transition from
        dry to wet pixels than 0.0.
    interp_intervals : int, optional
        Whether to interpolate to an increased density of intervals.
        This can be useful for reducing the impact of "terrace"-like
        artefacts across very low sloping intertidal flats where we have
        minimal satellite observations. Defaults to 200; set to None to
        deactivate.
    smooth_radius : int, optional
        A rolling mean filter can be applied to smooth data along the
        tide interval dimension. This produces smoother DEM surfaces
        than using the rolling median directly. Defaults to 10; set to
        None to deactivate.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the DEM for the given area, with
        a single variable 'elevation'.
    """

    # Apply optional interval interpolation
    if interp_intervals is not None:
        print(f"Applying tidal interval interpolation to {interp_intervals} intervals")
        interval_ds = interval_ds.interp(
            interval=np.linspace(0, interval_ds.interval.max(), interp_intervals),
            method="linear",
        )

    # Smooth tidal intervals using a rolling mean
    if smooth_radius is not None:
        print(f"Applying rolling mean smoothing with radius {smooth_radius}")
        smoothed_ds = interval_ds.rolling(
            interval=smooth_radius,
            center=True,
            min_periods=int(smooth_radius / 2.0),
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
    dem_flat = tide_thresh.where(~always_dry)

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
    misclassified_ds = flat_ds.where(misclassified_all)

    # Calculate uncertainty by taking the Median Absolute Deviation of
    # all misclassified points.
    if method == "mad":
        # Calculate median of absolute deviations
        # TODO: Account for large MAD on pixels with very few
        # misclassified points. Set < n misclassified points to 0 MAD?
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


def pixel_dem_debug(
    x,
    y,
    flat_ds,
    interval_ds,
    ndwi_thresh=0.1,
    interp_intervals=200,
    smooth_radius=20,
    certainty_method="mad",
):
    # Unstack data back to x, y so we can select pixels by their coordinates
    flat_unstacked = flat_ds[["tide_m", "ndwi"]].unstack().sortby(["time", "x", "y"])
    interval_unstacked = (
        interval_ds[["tide_m", "ndwi"]].unstack().sortby(["interval", "x", "y"])
    )

    # Extract nearest pixel to x and y coords
    flat_pixel = flat_unstacked.sel(x=x, y=y, method="nearest")
    interval_pixel = interval_unstacked.sel(x=x, y=y, method="nearest")

    # Apply interval interpolation and rolling mean
    interval_clean_pixel = (
        interval_pixel.interp(
            interval=np.linspace(0, interval_ds.interval.max(), interp_intervals),
            method="linear",
        )[["tide_m", "ndwi"]]
        .rolling(
            interval=smooth_radius,
            center=False,
            min_periods=int(smooth_radius / 2.0),
        )
        .mean()
    )

    if not isinstance(ndwi_thresh, float):
        # Experiment with variable threshold
        ndwi_thresh = xr.DataArray(
            np.linspace(ndwi_thresh[0], ndwi_thresh[-1], interp_intervals),
            coords={"interval": interval_clean_pixel.interval},
        )

    # Calculate DEM
    flat_dem_pixel = pixel_dem(
        interval_clean_pixel,
        ndwi_thresh=ndwi_thresh,
        interp_intervals=None,
        smooth_radius=None,
    )

    # Calculate certainty
    elev_low_mad, elev_high_mad, _, _ = pixel_uncertainty(
        flat_pixel,
        flat_dem_pixel,
        ndwi_thresh,
        method=certainty_method,
    )

    # Plot
    flat_pixel.to_dataframe().plot.scatter(x="tide_m", y="ndwi", color="black", s=3)
    interval_pixel.to_dataframe().rename({"ndwi": "rolling median"}, axis=1).plot(
        x="tide_m", y="rolling median", ax=plt.gca()
    )
    interval_clean_pixel.to_dataframe().rename({"ndwi": "smoothed"}, axis=1).plot(
        x="tide_m", y="smoothed", ax=plt.gca()
    )

    if not isinstance(ndwi_thresh, float):
        plt.plot(
            interval_clean_pixel.tide_m.sel(
                interval=~interval_clean_pixel.tide_m.isnull()
            ),
            ndwi_thresh.sel(interval=~interval_clean_pixel.tide_m.isnull()),
            color="black",
            linestyle="--",
            lw=1,
            alpha=1,
        )
    else:
        plt.gca().axvspan(
            elev_low_mad.item(), elev_high_mad.item(), color="lightgrey", alpha=0.3
        )
        plt.gca().axhline(ndwi_thresh, color="black", linestyle="--", lw=1, alpha=1)

    plt.gca().axvline(
        flat_dem_pixel.elevation, color="black", linestyle="--", lw=1, alpha=1
    )
    plt.gca().set_ylim(-1, 1)



def elevation(
    satellite_ds,
    valid_mask=None,
    ndwi_thresh=0.1,
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.15,
    windows_n=100,
    window_prop_tide=0.15,
    max_workers=None,
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
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
    valid_mask : xr.DataArray, optional
        A boolean mask used to optionally constrain the analysis area,
        with the same spatial dimensions as `satellite_ds`. For example,
        this could be a mask generated from a topo-bathy DEM, used to
        limit the analysis to likely intertidal pixels. Default is None,
        which will not apply a mask.
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI) above which pixels are considered water, by default 0.1.
    min_freq, max_freq : float, optional
        Minimum and maximum frequency of wetness required for a pixel to
        be included in the analysis, by default 0.01 and 0.99.
    min_correlation : float, optional
        Minimum correlation between water index and tide height required
        for a pixel to be included in the analysis, by default 0.15.
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

    # Model tides into every pixel in the three-dimensional satellite
    # dataset (x by y by time)
    log.info(f"{log_prefix}Modelling tide heights for each pixel")
    if tide_model[0] == "ensemble":
        # Use ensemble model combining multiple input ocean tide models
        tide_m, _ = pixel_tides_ensemble(
            satellite_ds,
            directory=tide_model_dir,
            ancillary_points="data/raw/corr_points.geojson",
        )

    else:
        # Use single input ocean tide model
        tide_m, _ = pixel_tides(
            satellite_ds,
            resample=True,
            model=tide_model,
            directory=tide_model_dir,
        )

    # Set tide array pixels to nodata if the satellite data array pixels
    # contain nodata. This ensures that we ignore any tide observations
    # where we don't have matching satellite imagery
    log.info(
        f"{log_prefix}Masking nodata and adding tide heights to satellite data array"
    )
    satellite_ds["tide_m"] = tide_m.where(
        ~satellite_ds.to_array().isel(variable=0).isnull().drop("variable")
    )

    # Flatten array from 3D (time, y, x) to 2D (time, z) and drop pixels
    # with no correlation with tide. This greatly improves processing
    # time by ensuring only a narrow strip of tidally influenced pixels
    # along the coast are analysed, rather than the entire study area.
    # (This step is later reversed using the `flat_to_ds` function)
    log.info(
        f"{log_prefix}Flattening satellite data array and filtering to intertidal candidate pixels"
    )
    if valid_mask is not None:
        log.info(f"{log_prefix}Applying valid data mask to constrain study area")
    flat_ds, freq, corr, intertidal_candidates = ds_to_flat(
        satellite_ds,
        min_freq=min_freq,
        max_freq=max_freq,
        min_correlation=min_correlation,
        valid_mask=valid_mask,
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
    flat_dem = pixel_dem(interval_ds, ndwi_thresh)

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
    "--study_area",
    type=str,
    required=True,
    help="A string providing a GridSpec tile ID (e.g. in the form "
    "'x143y56') to run the analysis on.",
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
    "which typically captures this transition more reliably than 0.0.",
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
    default=0.15,
    help="Minimum correlation between water index and tide height "
    "required for a pixel to be included in the analysis, by default "
    "0.15.",
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
    multiple=True,
    default=["FES2014"],
    help="The model used for tide modelling, as supported by the "
    "`pyTMD` Python package. Options include 'FES2014' (default), "
    "'TPXO9-atlas-v5', 'TPXO8-atlas-v1', 'EOT20', 'HAMTIDE11', 'GOT4.10'. ",
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
            include_s2=True,
            include_ls=True,
            filter_gqa=True,
            max_cloudcover=90,
            skip_broken_datasets=True,
        )

        # Load data
        satellite_ds.load()

        # Load data from GA's Australian Bathymetry and Topography Grid 2009
        topobathy_ds = load_topobathy(
            dc, satellite_ds, product="ga_multi_ausbath_0", resampling="bilinear"
        )

        # Calculate elevation
        log.info(f"Study area {study_area}: Calculating Intertidal Elevation")
        ds, ds_aux, tide_m = elevation(
            satellite_ds,
            valid_mask=topobathy_ds.height_depth > -20,
            ndwi_thresh=ndwi_thresh,
            min_freq=min_freq,
            max_freq=max_freq,
            min_correlation=min_correlation,
            windows_n=windows_n,
            window_prop_tide=window_prop_tide,
            tide_model=tide_model,
            tide_model_dir=tide_model_dir,
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
            ds, prefix=f"{output_dir}/{study_area}_{start_date}_{end_date}"
        )

        if output_auxiliaries:
            # Export auxiliary debug layers as GeoTIFFs with optimised data types
            log.info(
                f"Study area {study_area}: Exporting debugging GeoTIFFs to {output_dir}"
            )
            export_intertidal_rasters(
                ds_aux,
                prefix=f"{output_dir}/{study_area}_{start_date}_{end_date}_debug",
            )

        # Workflow completed; close Dask client
        client.close()
        log.info(f"Study area {study_area}: Completed DEA Intertidal workflow")

    except Exception as e:
        log.exception(f"Study area {study_area}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    intertidal_cli()
