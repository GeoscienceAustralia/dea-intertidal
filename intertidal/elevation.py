import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from odc.algo import mask_cleanup
import odc.geo.xr

from dea_tools.coastal import model_tides


def load_data(
    dc,
    geom,
    time_range=("2019", "2021"),
    resolution=10,
    crs="epsg:32753",
    s2_prod="s2_nbart_ndwi",
    ls_prod="ls_nbart_ndwi",
    config_path="configs/dea_virtual_product_landsat_s2.yaml",
):

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
    ds = (
        xr.concat(data_list, dim="time")
        .sortby("time")
        .drop(["cloud_mask", "contiguity"])
    )
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


def ds_to_flat(ds, ndwi_thresh=0.1, index='ndwi', min_freq=0.01, max_freq=0.99, min_correlation=0.2):

    """
    Converts a three dimensional (x, y, time) array to a two
    dimensional (z, time) array by selecting only pixels along
    a narrow coastal strip. The x and y dimensions are stacked into
    a single dimension.

    Selected pixels must show a pattern of wetting and drying over
    the time series, and have a positive correlation between water
    observations and tide height.
    """

    # Calculate frequency of wet per pixel, then threshold
    # to exclude always wet and always dry
    freq = (ds[index] > ndwi_thresh).where(~ds[index].isnull()).mean(dim="time")
    good_mask = (freq >= min_freq) & (freq <= max_freq)

    # Flatten to 1D
    ds_flat = ds.stack(z=("x", "y")).where(good_mask.stack(z=("x", "y")), drop=True)

    # Calculate correlations, and keep only pixels with positive
    # correlations between water observations and tide height
    correlations = xr.corr(ds_flat[index] > ndwi_thresh, ds_flat.tide_m, dim="time")
    ds_flat = ds_flat.where(correlations > min_correlation, drop=True)

    print(
        f"Reducing analysed pixels from {freq.count().item()} to {len(ds_flat.z)} ({len(ds_flat.z) * 100 / freq.count().item():.2f}%)"
    )
    return ds_flat, freq, good_mask


def create_dask_gateway_cluster(profile="r5_L", workers=2):
    """
    Create a cluster in our internal dask cluster.
    Parameters
    ----------
    profile : str
        Possible values are:
            - r5_L (2 cores, 15GB memory)
            - r5_XL (4 cores, 31GB memory)
            - r5_2XL (8 cores, 63GB memory)
            - r5_4XL (16 cores, 127GB memory)
    workers : int
        Number of workers in the cluster.
    """

    try:

        from dask_gateway import Gateway

        gateway = Gateway()

        # Close any existing clusters
        if len(cluster_names) > 0:
            print("Cluster(s) still running:", cluster_names)
            for n in cluster_names:
                cluster = gateway.connect(n.name)
                cluster.shutdown()

        # Connect to new cluster
        options = gateway.cluster_options()
        options["profile"] = profile
        options["jupyterhub_user"] = "robbi"
        cluster = gateway.new_cluster(options)
        cluster.scale(workers)

        return cluster

    except ClientConnectionError:
        raise ConnectionError("Access to dask gateway cluster unauthorized")


def abslmp_gauge(
    coords, start_year=2019, end_year=2021, data_path="data/raw/ABSLMP", plot=True
):

    """
    Loads water level data from the nearest Australian Baseline Sea Level
    Monitoring Project gauge.
    """

    from shapely.ops import nearest_points
    from shapely.geometry import Point

    # Standardise coords format
    if isinstance(coords, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)):
        print("Using dataset bounds to load gauge data")
        coords = coords.odc.geobox.geographic_extent.geom
    elif isinstance(coords, tuple):
        coords = Point(coords)

    # Convert coords to GeoDataFrame
    coords_gdf = gpd.GeoDataFrame(geometry=[coords], crs="EPSG:4326").to_crs(
        "EPSG:3577"
    )

    # Load station metadata
    site_metadata_df = pd.read_csv(
        f"{data_path}/ABSLMP_station_metadata.csv", index_col="ID CODE"
    )

    # Convert metadata to GeoDataFrame
    sites_metadata_gdf = gpd.GeoDataFrame(
        data=site_metadata_df,
        geometry=gpd.points_from_xy(
            site_metadata_df.LONGITUDE, site_metadata_df.LATITUDE
        ),
        crs="EPSG:4326",
    ).to_crs("EPSG:3577")

    # Find nearest row
    site_metadata_gdf = gpd.sjoin_nearest(coords_gdf, sites_metadata_gdf).iloc[0]
    site_id = site_metadata_gdf["index_right"]
    site_name = site_metadata_gdf["TOWN / DISTRICT"]

    # Read all tide data
    print(f"Loading ABSLMP gauge {site_id} ({site_name})")
    available_paths = glob(f"{data_path}/{site_id}_*.csv")
    available_years = sorted([int(i[-8:-4]) for i in available_paths])

    loaded_data = [
        pd.read_csv(
            f"{data_path}/{site_id}_{year}.csv",
            index_col=0,
            parse_dates=True,
            na_values=-9999,
        )
        for year in range(start_year, end_year)
        if year in available_years
    ]

    try:

        # Combine loaded data
        df = pd.concat(loaded_data).rename(
            {" Adjusted Residuals": "Adjusted Residuals"}, axis=1
        )

        # Extract water level and residuals
        clean_df = df[["Sea Level", "Adjusted Residuals"]].rename_axis("time")
        clean_df.columns = ["sea_level", "residuals"]
        clean_df["sea_level"] = clean_df.sea_level - site_metadata_gdf.AHD
        clean_df["sea_level_noresiduals"] = clean_df.sea_level - clean_df.residuals

        # Summarise non-residual waterlevels by week to assess seasonality
        seasonal_df = (
            clean_df[["sea_level_noresiduals"]]
            .groupby(clean_df.index.isocalendar().week)
            .mean()
        )

        # Plot
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            clean_df["sea_level"].plot(ax=axes[0], lw=0.2)
            axes[0].set_title("Water levels (AHD)")
            axes[0].set_xlabel("")
            clean_df["residuals"].plot(ax=axes[1], lw=0.3)
            axes[1].set_title("Adjusted residuals")
            axes[1].set_xlabel("")
            clean_df["sea_level_noresiduals"].plot(ax=axes[2], lw=0.2)
            axes[2].set_title("Water levels, no residuals (AHD)")
            axes[2].set_xlabel("")
            seasonal_df.plot(ax=axes[3])
            axes[3].set_title("Seasonal")

        return clean_df, seasonal_df

    except ValueError:
        print(
            f"\nNo data for selected start and end year. Available years include:\n{available_years}"
        )


def abslmp_correction(ds, start_year=2010, end_year=2021):
    """
    Applies a seasonal correction to tide height data based on the nearest
    Australian Baseline Sea Level Monitoring Project gauge.
    """

    # Load seasonal data from ABSLMP
    _, abslmp_seasonal_df = abslmp_gauge(
        coords=ds, start_year=start_year, end_year=end_year, plot=False
    )

    # Apply weekly offsets to tides
    df_correction = abslmp_seasonal_df.loc[ds.time.dt.weekofyear].reset_index(drop=True)
    df_correction.index = ds.time
    da_correction = (
        df_correction.rename_axis("time")
        .rename({"sea_level_noresiduals": "tide_m"}, axis=1)
        .to_xarray()
    )
    ds["tide_m"] = ds["tide_m"] + da_correction.tide_m

    return ds