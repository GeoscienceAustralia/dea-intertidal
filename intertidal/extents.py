import datacube
import odc.geo.xr

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from skimage import graph
from odc.geo.geobox import GeoBox
from odc.geo.gridspec import GridSpec
from odc.geo.types import xy_
from odc.algo import mask_cleanup
from odc.geo.xr import xr_zeros
from dea_tools.spatial import xr_rasterize
from rasterio.features import sieve
from intertidal.io import (
    load_aclum_mask,
    extract_geobox,
)


def _cost_distance(
    cost_surface, start_array, sampling=None, geometric=True, **mcp_kwargs
):
    """
    Calculate accumulated least-cost distance through a cost surface
    array from a set of starting cells to every other cell in an array,
    using methods from `skimage.graph.MCP` or `skimage.graph.MCP_Geometric`.

    Parameters
    ----------
    cost_surface : ndarray
        A 2D array representing the cost surface.
    start_array : ndarray
        A 2D array with the same shape as `cost_surface` where non-zero
        values indicate start points.
    sampling : tuple, optional
        For each dimension, specifies the distance between two cells.
        If not given or None, unit distance is assumed.
    geometric : bool, optional
        If True, `skimage.graph.MCP_Geometric` will be used to calculate
        costs, accounting for the fact that diagonal vs. axial moves
        are of different lengths and weighting path costs accordingly.
        If False, costs will be calculated simply as the sum of the
        values of the costs array along the minimum cost path.
    **mcp_kwargs :
        Any additional keyword arguments to pass to `skimage.graph.MCP`
        or `skimage.graph.MCP_Geometric`.

    Returns
    -------
    lcd : ndarray
        A 2D array of the least-cost distances from the start cell to all other cells.
    """

    # Initialise relevant least cost graph
    if geometric:
        lc_graph = graph.MCP_Geometric(
            costs=cost_surface,
            sampling=sampling,
            **mcp_kwargs,
        )
    else:
        lc_graph = graph.MCP(
            costs=cost_surface,
            sampling=sampling,
            **mcp_kwargs,
        )

    # Extract starting points from the array (pixels with non-zero values)
    starts = list(zip(*np.nonzero(start_array)))

    # Calculate the least-cost distance from the start cell to all other cells
    lcd = lc_graph.find_costs(starts=starts)[0]

    return lcd


def xr_cost_distance(cost_da, starts_da, use_cellsize=False, geometric=True):
    """
    Calculate accumulated least-cost distance through a cost surface
    array from a set of starting cells to every other cell in an
    xarray.DataArray, returning results as an xarray.DataArray.

    Parameters
    ----------
    cost_da : xarray.DataArray
        An xarray.DataArray representing the cost surface, where pixel
        values represent the cost of moving through each pixel.
    starts_da : xarray.DataArray
        An xarray.DataArray with the same shape as `cost_da` where non-
        zero values indicate start points for the distance calculation.
    use_cellsize : bool, optional
        Whether to incorporate cell size when calculating the distance
        between two cells, based on the spatial resolution of the array.
        Default is False, which will assume distances between cells will
        be based on cost values only.
    geometric : bool, optional
        If True, `skimage.graph.MCP_Geometric` will be used to calculate
        costs, accounting for the fact that diagonal vs. axial moves
        are of different lengths and weighting path costs accordingly.
        If False, costs will be calculated simply as the sum of the
        values of the costs array along the minimum cost path.

    Returns
    -------
    costdist_da : xarray.DataArray
        An xarray.DataArray providing least-cost distances between every
        cell and the nearest start cell.
    """

    # Use resolution from input arrays if requested
    if use_cellsize:
        x, y = cost_da.odc.geobox.resolution.xy
        cellsize = (abs(y), abs(x))
    else:
        cellsize = None

    # Compute least cost array
    costdist_array = _cost_distance(
        cost_da, starts_da.values, sampling=cellsize, geometric=geometric
    )

    # Wrap as xarray
    costdist_da = xr.DataArray(costdist_array, coords=cost_da.coords)

    return costdist_da


def load_connectivity_mask(
    dc,
    geobox,
    product="ga_srtm_dem1sv1_0",
    elevation_band="dem_h",
    resampling="bilinear",
    buffer=20000,
    max_threshold=100,
    add_mangroves=False,
    mask_filters=[("dilation", 3)],
    **cost_distance_kwargs,
):
    """
    Generates a mask based on connectivity to ocean pixels, using least-
    cost distance weighted by elevation. By incorporating elevation,
    this mask will extend inland further in areas of low lying elevation
    and less far inland in areas of steep terrain.

    Parameters
    ----------
    dc : Datacube
        A Datacube instance for loading data.
    geobox : ndarray
        The GeoBox defining the pixel grid to load data into (e.g.
        resolution, extents, CRS).
    product : str, optional
        The name of the DEM product to load from the datacube.
        Defaults to "ga_srtm_dem1sv1_0".
    elevation_band : str, optional
        The name of the band containing elevation data. Defaults to
        "height_depth".
    resampling : str, optional
        The resampling method to use, by default "bilinear".
    buffer : int, optional
        The distance by which to buffer the input GeoBox to reduce edge
        effects. This buffer will eventually be removed and clipped back
        to the original GeoBox extent. Defaults to 20,000 metres.
    max_threshold: int, optional
        Value used to threshold the resulting cost distance to produce
        a mask.
    mask_filters : list of tuples, optional
        An optional list of morphological processing steps to pass to
        the `mask_cleanup` function. The default is `[("dilation", 3)]`,
        which will dilate True pixels by a radius of 3 pixels.
    **cost_distance_kwargs :
        Optional keyword arguments to pass to the ``xr_cost_distance``
        cost-distance function.

    Returns
    -------
    costdist_mask : xarray.DataArray
        An output boolean mask, where True represent pixels located in
        close cost-distance proximity to the ocean.
    costdist_da : xarray.DataArray
        The output cost-distance array, reflecting distance from the
        ocean weighted by elevation.
    """

    # Buffer input geobox and reduce resolution to ensure that the
    # connectivity analysis is less affected by edge effects
    geobox_buffered = GeoBox.from_bbox(
        geobox.buffered(xbuff=buffer, ybuff=buffer).boundingbox,
        resolution=30,
        tight=True,
    )
    
    # Exclude tiles that fall outside of the DEM
    try:
        # Load DEM data
        dem_da = dc.load(
            product="ga_srtm_dem1sv1_0",
            measurements=[elevation_band],
            resampling="bilinear",
            like=geobox_buffered,
        ).squeeze()[elevation_band]
    except Exception as e:
        print(f'An error occurred in {study_area}: {e}')
        return

    # Identify starting points (ocean nodata points)
    if add_mangroves:
        print("Adding GMW mangroves to starting points")
        try:
            gmw_da = load_gmw_mask(dem_da)
            starts_da = (dem_da == dem_da.nodata) | gmw_da
        except:
            starts_da = dem_da == dem_da.nodata
    else:
        starts_da = dem_da == dem_da.nodata

    # Calculate cost surface (negative values are not allowed, so
    # negative nodata values are resolved by clipping values to between
    # 0 and infinity)
    costs_da = dem_da.clip(0, np.inf)

    # Run cost distance surface
    costdist_da = xr_cost_distance(
        cost_da=costs_da,
        starts_da=starts_da,
        **cost_distance_kwargs,
    )

    # Reproject back to original geobox extents and resolution
    costdist_da = costdist_da.odc.reproject(how=geobox)

    # Apply threshold
    costdist_mask = costdist_da < max_threshold

    # If requested, apply cleanup
    if mask_filters is not None:
        costdist_mask = mask_cleanup(costdist_mask, mask_filters=mask_filters)

    return costdist_mask, costdist_da


def load_gmw_mask(ds, gmw_path="/gdata1/data/mangroves/gmw_v3_2007_vec_aus.geojson"):
    """
    Experiment with loading GMW data.
    """
    gmw_gdf = gpd.read_file(
        gmw_path, bbox=ds.odc.geobox.boundingbox.to_crs("EPSG:4326")
    )
    gmw_da = xr_rasterize(gmw_gdf, ds)
    return gmw_da


def extents(
    dem, freq, corr, coastal_mask, urban_mask, min_correlation=0.15, sieve_size=5
):
    """
    Classify coastal ecosystems into broad classes based on
    wetting frequency, proximity to ocean, and relationships
    to tidal inundation, elevation, and urban land use.

    Parameters
    ----------
    dem : xarray.DataArray
        An intertidal Digital Elevation Model, as produced by
        the `intertidal.elevation` algorithm.
    freq : xarray.DataArray
        Frequency of wetness for each pixel, generated by the
        `intertidal.elevation` algorithm.
    corr : xarray.DataArray
        Correlation data between water index and tide height,
        as generated by the `intertidal.elevation` algorithm.
    coastal_mask : xarray.DataArray
        A boolean mask identifying likely coastal pixels. This
        is used to separate inland vs. ocean and coastal waters.
        For Australia, this is obtained using the
        `intertidal.extents.load_connectivity_mask` function,
        which uses cost-distance modelling to identify likely
        coastal pixels.
    urban_mask : xarray.DataArray
        A boolean mask identifying urban pixels that are applied
        to remove false positive water observations over urban
        areas. For Australia, this is obtained using the
        `intertidal.io.load_aclum_mask` function.
    min_correlation : float, optional
        Minimum correlation between water index and tide height
        for identifying low confidence intertidal pixels. Default
        is 0.15.
    sieve_size : int, optional
        Maximum number of grouped pixels to be sieved out to
        remove small noisy features. Default is 5.

    Returns
    -------
    xarray.DataArray
        A categorical DataArray with the following classes:
        - 255: Nodata
        - 1: Ocean and coastal waters (wet ≥50% of observations, within coastal mask)
        - 2: Exposed intertidal (low confidence) (correlation > 0.15, within coastal mask)
        - 3: Exposed intertidal (high confidence) (included in intertidal elevation dataset)
        - 4: Inland waters (wet >50% of observations, outside coastal mask)
        - 5: Land (wet <50% of observations)
    """

    # Identify dataset geobox
    geobox = dem.odc.geobox

    # Identify any pixels that are nodata in frequency
    is_nan = freq.isnull()

    # Spilt pixels into those that were mostly wet vs mostly dry.
    # Identify subset of mostly wet pixels that were inland
    mostly_dry = (freq < 0.50) & ~is_nan
    mostly_wet = (freq >= 0.50) & ~is_nan
    mostly_wet_inland = mostly_wet & ~coastal_mask

    # Identify low-confidence pixels as those with greater than 0.15
    # correlation. Use connectivity mask to mask out any that are "inland"
    intertidal_lc = (corr >= min_correlation) & coastal_mask

    # Identify high confidence intertidal as those in our elevation data
    intertidal_hc = dem.notnull()

    # Identify likely misclassified urban pixels as those that overlap with
    # the urban mask
    urban_misclass = mostly_wet_inland & urban_mask

    # Combine all classifications - this is done one-by-one, pasting each
    # new layer over the top of the existing data
    extents = xr_zeros(geobox=geobox, dtype="int16") + 255  # start with 255
    extents.values[mostly_wet] = 1  # Add in mostly wet pixels
    extents.values[mostly_wet_inland] = 4  # Add in mostly wet inland pixels on top
    extents.values[urban_misclass] = (
        5  # Set any pixels in the misclassified urban class to land
    )
    extents.values[mostly_dry] = 5  # Add mostly dry on top
    extents.values[intertidal_lc] = 2  # Add low confidence intertidal on top

    # Sieve out small noisy features. This is applied to everything but the
    # high confidence intertidal class, to keep a 1:1 match with elevation
    extents.values[:] = sieve(extents.values, size=sieve_size)

    # Add high confidence intertidal on top
    extents.values[intertidal_hc] = 3

    # Export to file
    extents.attrs["nodata"] = 255

    return extents
