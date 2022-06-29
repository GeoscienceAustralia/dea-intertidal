import numpy as np
import pandas as pd
import xarray as xr
from odc.algo import mask_cleanup


def load_data(dc, geom, time_range=("2019", "2021"), resolution=10, crs="epsg:32753"):

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry

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
    ls_ds_masked = ls_ds.where(~good_data_mask & ls_ds.contiguity)[["ndwi"]]

    # Combine into a single ds
    ds = xr.concat([s2_ds_masked, ls_ds_masked], dim="time").sortby("time")
    return ds


def pixel_tide_sort(ds, tide_var="tide_height", ndwi_var="ndwi", tide_dim="tide_n"):

    # Return indicies to sort each pixel by tide along time dim
    sort_indices = np.argsort(ds[tide_var].values, axis=0)

    # Use indices to sort both tide and NDWI array
    tide_sorted = np.take_along_axis(ds[tide_var].values, sort_indices, axis=0)
    ndwi_sorted = np.take_along_axis(ds[ndwi_var].values, sort_indices, axis=0)

    # Update values in array
    ds[tide_var][:] = tide_sorted
    ds[ndwi_var][:] = ndwi_sorted

    return (
        ds.assign_coords(coords={tide_dim: ("time", np.linspace(0, 1, len(ds.time)))})
        .swap_dims({"time": tide_dim})
        .drop("time")
    )


def item(ds, tide_dim="tide_n", intervals=10):
    
    # Bin tide heights into 9 tidal intervals from low (1) to high tide (9)
    tide_intervals = pd.cut(
        ds[tide_dim],
        bins=intervals,
        labels=range(1, intervals + 1),
        include_lowest=True,
    )

    # Add interval to dataset
    ds["tide_interval"] = xr.DataArray(tide_intervals, coords=[ds[tide_dim]])

    # For each interval, compute the median water index and tide height value
    ds_intervals = (
        ds[["tide_interval", "ndwi", "tide_m"]]
        .groupby("tide_interval")
        .median(dim=tide_dim)
        .compute()
    )
    
    # For each interval, compute the median water index and tide height value
    ds_confidence = (
        ds[["tide_interval", "ndwi"]]
        .groupby("tide_interval")
        .std(dim=tide_dim)
        .mean(dim='tide_interval')
        .compute()
    )

    # Plot the resulting set of tidal intervals
    item_da = ((ds_intervals.ndwi < 0) * ds_intervals.tide_interval).max(
        dim="tide_interval"
    )
    return item_da, ds_intervals, ds_confidence


def nidem(item_da, ds_intervals, intervals, clean=[("opening", 20), ("dilation", 10)]):

    from dea_tools.spatial import subpixel_contours, interpolate_2d, contours_to_arrays

    # Set up attributes to assign to each waterline
    attribute_df = pd.DataFrame(
        {"tide_m": ds_intervals.tide_m.median(dim=["x", "y"]).values}
    )

    # Extract waterlines
    contours_gdf = subpixel_contours(
        da=item_da,
        z_values=[i - 0.5 for i in range(1, intervals + 1)],
        crs=item_da.odc.geobox.crs,
        affine=item_da.odc.geobox.transform,
        attribute_df=attribute_df,
        min_vertices=2,
        output_path="tide_intervals.geojson",
        dim="tide_interval",
    )

    # First convert our contours shapefile into an array of XYZ points
    xyz_array = contours_to_arrays(contours_gdf, "tide_m")

    # Interpolate these XYZ points over the spatial extent of the Landsat dataset
    intertidal_dem = interpolate_2d(
        ds=ds_intervals,
        x_coords=xyz_array[:, 0],
        y_coords=xyz_array[:, 1],
        z_coords=xyz_array[:, 2],
    ).astype(np.float32)

    # Identify areas that are always wet (e.g. below low tide), or always dry
    above_lowest = item_da > item_da.min()
    below_highest = item_da < item_da.max()

    # Keep only pixels between high and low tide
    intertidal_dem_clean = intertidal_dem.where(above_lowest & below_highest)

    # Clean to remove noisy, narrow non-open coast pixels
    if clean:
        to_keep = mask_cleanup(below_highest, mask_filters=clean)
        intertidal_dem_clean = intertidal_dem_clean.where(to_keep)

    return intertidal_dem_clean
