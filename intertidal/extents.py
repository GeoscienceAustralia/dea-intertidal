def load_data(dc, geom, time_range=("2019", "2021"), resolution=10, crs="epsg:32753"):

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry
    from odc.algo import mask_cleanup
    import xarray as xr

    # Load in virtual product catalogue and select MNDWI product
    catalog = catalog_from_file("configs/dea_virtual_product_landsat_s2.yaml")

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

    # Load Sentinel-2 data
    product = catalog["s2_nbart_ndwi"]
    s2_ds = product.load(dc, **query_params)

    # Apply cloud mask and contiguity mask
    s2_ds_masked = s2_ds.where(s2_ds.cloud_mask == 1 & s2_ds.contiguity)[["ndwi"]]

    # Load Landsat data
    product = catalog["ls_nbart_ndwi"]
    ls_ds = product.load(dc, **query_params)

    # Clean cloud mask by applying morphological closing to all
    # valid (non cloud, shadow or nodata) pixels. This removes
    # long, narrow features like false positives over bright beaches.
    a = ls_ds.cloud_mask.isin([1, 4, 5])
    good_data_cleaned = mask_cleanup(
        mask=a,
        mask_filters=[("closing", 5)],
    )

    # Dilate cloud and shadow. To ensure that nodata areas (e.g.
    # Landsat 7 SLC off stripes) are not also dilated, only dilate
    # mask pixels (e.g. values 0 in `good_data_cleaned`) that are
    # outside of the original nodata pixels (e.g. not 0 in
    # `ls_ds.cloud_mask`)
    b = (good_data_cleaned == 0) & (ls_ds.cloud_mask != 0)
    good_data_mask = mask_cleanup(
        mask=b,
        mask_filters=[("dilation", 5)],
    )

    # Apply cloud mask and contiguity mask
    ls_ds_masked = ls_ds.where(~good_data_mask & ls_ds.contiguity)[["ndwi"]]

    # Combine into a single ds
    ds = xr.concat([s2_ds_masked, ls_ds_masked], dim="time").sortby("time")
    return ds


import dask
import xarray as xr
from dea_tools.spatial import interpolate_2d
import numpy as np

@dask.delayed
def interpolate_tide(timestep, tidepoints_gdf, method="rbf", factor=500):
    """
    Extract a subset of tide modelling point data for a given time-step,
    then interpolate these tides into the extent of the xarray dataset.
    Parameters:
    -----------
    timestep_tuple : tuple
        A tuple of x, y and time values sourced from `ds`. These values
        are used to set up the x and y grid into which tide heights for
        each timestep are interpolated. For example:
        `(ds.x.values, ds.y.values, ds.time.values)`
    tidepoints_gdf : geopandas.GeoDataFrame
        An `geopandas.GeoDataFrame` containing modelled tide heights
        with an index based on each timestep in `ds`.
    method : string, optional
        The method used to interpolate between point values. This string
        is either passed to `scipy.interpolate.griddata` (for 'linear',
        'nearest' and 'cubic' methods), or used to specify Radial Basis
        Function interpolation using `scipy.interpolate.Rbf` ('rbf').
        Defaults to 'rbf'.
    factor : int, optional
        An optional integer that can be used to subsample the spatial
        interpolation extent to obtain faster interpolation times, then
        up-sample this array back to the original dimensions of the
        data as a final step. For example, setting `factor=10` will
        interpolate ata into a grid that has one tenth of the
        resolution of `ds`. This approach will be significantly faster
        than interpolating at full resolution, but will potentially
        produce less accurate or reliable results.
    Returns:
    --------
    out_tide : xarray.DataArray
        A 2D array containing tide heights interpolated into the extent
        of the input data.
    """

    # Extract subset of observations based on timestamp of imagery
    time_string = str(timestep.time.values)[0:19].replace("T", " ")
    tidepoints_subset = tidepoints_gdf.loc[time_string]

    # Get lists of x, y and z (tide height) data to interpolate
    x_coords = tidepoints_subset.geometry.x.values.astype("float32")
    y_coords = tidepoints_subset.geometry.y.values.astype("float32")
    z_coords = tidepoints_subset.tide_m.values.astype("float32")

    # Interpolate tides into the extent of the satellite timestep
    out_tide = interpolate_2d(
        ds=timestep,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        method=method,
        factor=factor,
    )

    # Return data as a Float32 to conserve memory
    return out_tide.astype("float32")


def interpolate_tides(ds, tidepoints_gdf):
    """
    Interpolates tide heights into the spatial extent of each
    timestep of a satellite dataset, and return data as a lazily
    evaluated `xr.DataArray`.
    Parameters:
    -----------
    ds : xarray.Dataset
        An `xarray.Dataset` containing a time series of water index
        data (e.g. MNDWI) for the provided datacube query.
    tidepoints_gdf : geopandas.GeoDataFrame
        An `geopandas.GeoDataFrame` containing modelled tide heights
        with an index based on each timestep in `ds`.
    Returns:
    --------
    tide_m : xarray.DataArray
        A Dask-aware `xarray.DataArray` matching the dimensions of `ds`,
        containing a spatially interpolated tide height for each pixel.
    """

    # Function to lazily apply tidal interpolation to each timestep
    def _interpolate_tide_dask(x):
        return xr.DataArray(
            dask.array.from_delayed(
                interpolate_tide(x, tidepoints_gdf=tidepoints_gdf),
                ds.geobox.shape,
                np.float32,
            ),
            dims=["y", "x"],
        )

    # Apply func to each timestep
    tide_m = ds.groupby("time").apply(_interpolate_tide_dask)

    return tide_m