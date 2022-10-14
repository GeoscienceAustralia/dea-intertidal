import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from odc.algo import mask_cleanup
import odc.geo.xr


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
        .mean(dim="tide_interval")
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


