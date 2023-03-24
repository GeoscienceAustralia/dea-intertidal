import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from odc.algo import mask_cleanup
import odc.geo.xr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat

import datacube
from datacube.utils.geometry import Geometry

from intertidal.utils import load_config, configure_logging

# from dea_tools.coastal import model_tides
from dea_tools.coastal import pixel_tides
from dea_tools.dask import create_local_dask_cluster


def load_data(
    dc,
    geom,
    time_range=("2019", "2021"),
    resolution=10,
    crs="epsg:32753",
    s2_prod="s2_nbart_ndwi",
    ls_prod="ls_nbart_ndwi",
    config_path="configs/dea_virtual_product_landsat_s2.yaml",
    filter_gqa=True,
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
    ds = (
        xr.concat(data_list, dim="time")
        .sortby("time")
        .drop(["cloud_mask", "contiguity"])
    )
    return ds


def ds_to_flat(
    ds, ndwi_thresh=0.1, index="ndwi", min_freq=0.01, max_freq=0.99, min_correlation=0.2
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
    ds : xr.Dataset
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
    ds_flat : xr.Dataset
        Two-dimensional xarray dataset with dimensions (time, z)
    freq : xr.DataArray
        Frequency of wetness for each pixel.
    good_mask : xr.DataArray
        Boolean mask indicating which pixels meet the inclusion criteria.
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
    
    ## Preparation for extents workflow
    correlations3D = correlations.unstack("z").reindex_like(ds).transpose("y", "x")
    ds['freq_corr'] = freq.where(correlations3D > min_freq, drop=True)

    print(
        f"Reducing analysed pixels from {freq.count().item()} to {len(ds_flat.z)} ({len(ds_flat.z) * 100 / freq.count().item():.2f}%)"
    )
    return ds_flat, freq, good_mask, ds


def rolling_tide_window(
    i,
    ds,
    window_spacing,
    window_radius,
    tide_min,
    statistic="median",
):
    """
    This function takes a rolling window of tide observations from
    our flattened tide array, and returns a summary of these values.

    This is used to smooth our NDWI values along the tide dimension
    (e.g. rolling medians or quantiles).
    """

    # Set min and max thresholds to filter dataset
    thresh_centre = tide_min + (i * window_spacing)
    thresh_min = thresh_centre - window_radius
    thresh_max = thresh_centre + window_radius

    # Filter dataset
    masked_ds = ds.where((ds.tide_m >= thresh_min) & (ds.tide_m <= thresh_max))

    # Apply median or quantile
    if statistic == "quantile":
        ds_agg = xr_quantile(src=masked_ds, quantiles=[0.1, 0.5, 0.9], nodata=np.nan)
    elif statistic == "median":
        ds_agg = masked_ds.median(dim="time")  # .expand_dims(quantile=[0.5])
    elif statistic == "mean":
        ds_agg = masked_ds.mean(dim="time")  # .expand_dims(quantile=[0.5])

    # Add standard deviation
    ds_agg["ndwi_std"] = masked_ds.ndwi.std(dim="time")
    ds_agg["ndwi_count"] = (~masked_ds.ndwi.isnull()).sum(dim="time")

    return ds_agg


def pixel_rolling_median(ds_flat, windows_n=100, window_prop_tide=0.15, max_workers=64):
    """
    Calculate rolling medians for each pixel in an xarray.Dataset from
    low to high tide, using a set number of rolling windows (defined
    by `windows_n`) with radius determined by the proportion of the tide
    range specified by `window_prop_tide`.

    Parameters
    ----------
    ds_flat : xarray.Dataset
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
    tide_max = ds_flat.tide_m.max(dim="time")
    tide_min = ds_flat.tide_m.min(dim="time")
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
                for i in [ds_flat, window_spacing_tide, window_radius_tide, tide_min]
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
    interval_ds = xr.concat(out_list, dim="interval").sortby(["interval", "x", "y"])

    return interval_ds


def pixel_dem(interval_ds, ds, ndwi_thresh, fname, export_geotiff=True):
    
    # Use standard deviation as measure of confidence
    confidence = interval_ds.ndwi_std

    # Smooth using a rolling mean
    smoothed_ds = interval_ds.rolling(
        interval=20, center=False, min_periods=1
    ).mean()

    # Outputs
    output_list = []

    # Export DEM for rolling median and half a standard deviation either side
    for q in [-0.5, 0, 0.5]:
        
        suffix = {-0.5: "dem_low", 0: "dem", 0.5: "dem_high"}[q]        
        print(f"Processing {suffix}")

        # Identify the max tide per pixel where NDWI == land
        tide_dry = (smoothed_ds.tide_m + (confidence * q)).where(
            smoothed_ds.ndwi <= ndwi_thresh
        )
        tide_thresh = tide_dry.max(dim="interval")
        tide_max = smoothed_ds.tide_m.max(dim="interval")

        # Remove any pixel where tides max out (i.e. always land), and 
        # unstack back to 3D (x, y, time) array
        always_dry = tide_thresh >= tide_max
        dem = tide_thresh.where(~always_dry)
        dem = dem.unstack("z").reindex_like(ds).transpose("y", "x")

        # Add name and add outputs to list
        dem = dem.rename(suffix)        
        output_list.append(dem)
        
    # Merge into a single xarray.Dataset
    dem_ds = xr.merge(output_list).drop('variable')
    
    # Subtract low from high DEM to get a single confidence layer
    # Note: This may produce unexpected results at the top and bottom
    # of the intertidal zone, as the low and high DEMs may not be 
    # currently be properly masked to remove always wet/dry terrain
    dem_ds['confidence'] = (dem_ds.dem_high - dem_ds.dem_low)
    
    # # Export as GeoTIFFs
    # if export_geotiff:
    #     print(f"\nExporting GeoTIFF files to 'data/interim/pixel_{fname}_....tif'")
    #     dem_ds.map(
    #         lambda x: x.odc.write_cog(
    #             fname=f"data/interim/pixel_{fname}_{x.name}.tif", overwrite=True
    #         )
    #     )    
    
    # Merge dem_ds into ds
    ds = ds.merge(dem_ds)
    
    return ds #dem_ds


def elevation(study_area,
              start_year=2020,
              end_year=2022,
              resolution=10,
              crs="EPSG:3577",
              ndwi_thresh=0.1,
              include_s2=True,
              include_ls=True,
              filter_gqa=False,
              config_path='configs/dea_intertidal_config.yaml',
              log=None):
    
    if log is None:
        log = configure_logging()
    
    # Create local dask cluster to improve data load time
    client = create_local_dask_cluster(return_client=True)

    # Connect to datacube
    dc = datacube.Datacube(app="Intertidal_elevation")

    # Load analysis params from config file
    config = load_config(config_path)
    
    # Load study area from tile grid if passed a string
    if isinstance(study_area, int):
        
        # Load study area
        gridcell_gdf = (
            gpd.read_file(config['Input files']['grid_path']).to_crs(
                epsg=4326).set_index('id'))
        gridcell_gdf.index = gridcell_gdf.index.astype(int).astype(str)
        gridcell_gdf = gridcell_gdf.loc[[str(study_area)]]

        # Create geom as input for dc.load
        geom = Geometry(geom=gridcell_gdf.iloc[0].geometry, crs='EPSG:4326')
        fname = f"{study_area}_{start_year}-{end_year}"
        log.info(f"Study area {study_area}: Loaded study area grid")
    
    # Otherwise, use supplied geom
    else:        
        geom = study_area
        study_area = 'testing'
        fname = f"{study_area}_{start_year}-{end_year}"
        log.info(f"Study area {study_area}: Loaded custom study area")
    
    # Load data
    log.info(f"Study area {study_area}: Loading satellite data")
    ds = load_data(dc=dc, 
               geom=geom, 
               time_range=(str(start_year), str(end_year)), 
               resolution=resolution, 
               crs=crs,
               s2_prod="s2_nbart_ndwi" if include_s2 else None,
               ls_prod="ls_nbart_ndwi" if include_ls else None,
               config_path=config['Virtual product']['virtual_product_path'],
               filter_gqa=filter_gqa)[['ndwi']]
    ds.load()
    
    # Model tides into every pixel in the three-dimensional (x by y by time) satellite dataset
    log.info(f"Study area {study_area}: Modelling tide heights for each pixel")
    ds["tide_m"], _ = pixel_tides(ds, resample=True)

    # Set tide array pixels to nodata if the satellite data array pixels contain
    # nodata. This ensures that we ignore any tide observations where we don't
    # have matching satellite imagery 
    log.info(f"Study area {study_area}: Masking nodata and adding tide heights to satellite data array")
    ds["tide_m"] = ds["tide_m"].where(~ds.to_array().isel(variable=0).isnull())

    # Flatten array from 3D to 2D and drop pixels with no correlation with tide
    log.info(f"Study area {study_area}: Flattening satellite data array and filtering to tide influenced pixels")
    ds_flat, freq, good_mask, ds = ds_to_flat(
        ds, ndwi_thresh=0.0, min_freq=0.01, max_freq=0.99, min_correlation=0.2)
    
    # Per-pixel rolling median
    log.info(f"Study area {study_area}: Running per-pixel rolling median")
    interval_ds = pixel_rolling_median(
    ds_flat, windows_n=100, window_prop_tide=0.15, max_workers=64)
    
    # Model intertidal elevation and confidence
    log.info(f"Study area {study_area}: Modelling intertidal elevation and confidence")
    ds = pixel_dem(interval_ds, ds, ndwi_thresh, fname)
    
    # Close dask client
    client.close()
    
    log.info(f"Study area {study_area}: Successfully completed intertidal elevation modelling")    
    return ds

    
    
    
    
    
    
    
    
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
