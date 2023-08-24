import logging
import yaml
import fsspec
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd
from pathlib import Path


def configure_logging(name: str = "DEA Intertidal") -> logging.Logger:
    """
    Configure logging for the application.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def round_date_strings(date, round_type="end"):
    """
    Round a date string up or down to the start or end of a given time
    period.

    Parameters
    ----------
    date : str
        Date string of variable precision (e.g. "2020", "2020-01",
        "2020-01-01").
    round_type : str, optional
        Type of rounding to perform. Valid options are "start" or "end".
        If "start", date is rounded down to the start of the time period.
        If "end", date is rounded up to the end of the time period.
        Default is "end".

    Returns
    -------
    date_rounded : str
        The rounded date string.

    Examples
    --------
    >>> round_date_strings('2020')
    '2020-12-31 00:00:00'

    >>> round_date_strings('2020-01', round_type='start')
    '2020-01-01 00:00:00'

    >>> round_date_strings('2020-01', round_type='end')
    '2020-01-31 00:00:00'
    """

    # Determine precision of input date string
    date_segments = len(date.split("-"))

    # If provided date has no "-", treat it as having year precision
    if date_segments == 1 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + YearBegin(0))
    elif date_segments == 1 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + YearEnd(0))

    # If provided date has one "-", treat it as having month precision
    elif date_segments == 2 and round_type == "start":
        date_rounded = str(pd.to_datetime(date) + MonthBegin(0))
    elif date_segments == 2 and round_type == "end":
        date_rounded = str(pd.to_datetime(date) + MonthEnd(0))

    # If more than one "-", then return date as-is
    elif date_segments > 2:
        date_rounded = date

    return date_rounded


def export_intertidal_rasters(
    ds,
    prefix="data/interim/testing",
    int_bands=None,
    int_nodata=-999,
    int_dtype=np.int16,
    float_dtype=np.float32,
    overwrite=True,
):
    """
    Export outputs of the DEA Intertidal workflow to COG GeoTIFF files.

    If a band contains "elevation" in the name it is exported as a
    float32 data type. Otherwise, the band is exported as an integer16
    data type, after filling NaN with the nodata value and setting the
    nodata attribute on the layer.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the bands to be exported.
    prefix : str, optional
        A string that will be used as a prefix for the output file
        names (default is "testing").
    int_bands : tuple or list, optional
        A list of bands to export as integer datatype. If None, will use
        the following list of bands: ("exposure", "extents",
        "offset_hightide", "offset_lowtide", "spread")
    int_nodata : int, optional
        An integer that represents nodata values for integer bands
        (default is -999).
    int_dtype : string or numpy data type, optional
        The data type to use for integer layers (default is
        np.int16).
    float_dtype : string or numpy data type, optional
        The data type to use for floating point layers (default is
        np.float32).
    overwrite : bool, optional
        A boolean value that determines whether or not to overwrite
        existing files (default is True).

    Returns
    -------
    None
    """

    # Use default list of bands to convert to integers if none provided
    if int_bands is None:
        int_bands = (
            # Primary layers
            "exposure",
            "extents",
            "oa_offset_hightide",
            "oa_offset_lowtide",
            "oa_spread",
            # Debug/auxiliary layers
            "misclassified_px_count",
            "intertidal_candidates",
        )

    for band in ds:
        # Export specific bands as integer16 data types by first filling
        # NaN with nodata value before converting to int, then setting
        # nodata attribute on layer
        if band in int_bands:
            band_da = ds[band].fillna(int_nodata).astype(int_dtype)
            band_da.attrs["nodata"] = int_nodata

        # Export other bands as float32 data types
        else:
            band_da = ds[band].astype(float_dtype)

        # Export band to file
        band_da.odc.write_cog(fname=f"{prefix}_{band}.tif", overwrite=overwrite)


def intertidal_hillshade(
    elevation,
    extents,
    azdeg=315,
    altdeg=45,
    dyx=10,
    vert_exag=100,
    **shade_kwargs,
):
    """
    Create a hillshade array for an intertidal zone given an elevation
    array and an extents array.

    Parameters
    ----------
    elevation : str or xr.DataArray
        An xr.DataArray or a path to the elevation raster file.
    extents : str or xr.DataArray
        xr.DataArray or a path to the extents raster file.
    azdeg : float, optional
        The azimuth angle of the light source, in degrees. Default is 315.
    altdeg : float, optional
        The altitude angle of the light source, in degrees. Default is 45.
    dyx : float, optional
        The distance between pixels in the x and y directions, in meters.
        Default is 10.
    vert_exag : float, optional
        The vertical exaggeration of the hillshade. Default is 100.
    **shade_kwargs : optional
        Additional keyword arguments to pass to
        `matplotlib.colors.LightSource.shade()`.

    Returns
    -------
    xr.DataArray
        The hillshade array for the intertidal zone.
    """

    from matplotlib.colors import LightSource, Normalize
    import matplotlib.pyplot as plt
    import xarray as xr

    # Read data
    if isinstance(elevation, str):
        elevation = xr.open_rasterio(elevation).squeeze("band")
    if isinstance(extents, str):
        extents = xr.open_rasterio(extents).squeeze("band")

    # Fill upper and bottom of intertidal zone with min and max heights
    # so that hillshade can be applied across the entire raster
    # elevation_filled = xr.where(extents == 0, elevation.min(), elevation)
    # elevation_filled = xr.where(extents == 2, elevation.max(), elevation_filled)
    elevation_filled = xr.where(extents == 0, elevation.max(), elevation)
    elevation_filled = xr.where(extents == 2, elevation.min(), elevation_filled)
    elevation_filled = xr.where(extents == 3, elevation.max(), elevation_filled)
    elevation_filled = xr.where(extents == 4, elevation.max(), elevation_filled)

    from scipy.ndimage import gaussian_filter

    input_data = gaussian_filter(elevation_filled, sigma=1)

    # Create hillshade based on elevation data
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hillshade = ls.shade(
        input_data,
        cmap=plt.cm.viridis,
        blend_mode=lambda x, y: x * y,
        vert_exag=vert_exag,
        dx=dyx,
        dy=dyx,
        **shade_kwargs,
    )

    # Mask out non-intertidal pixels
    hillshade = np.where(
        np.expand_dims(extents.values == 1, axis=-1), hillshade, np.nan
    )

    # Create a new xarray data array from the numpy array
    hillshaded_da = xr.DataArray(
        hillshade,
        dims=["y", "x", "variables"],
        coords={
            "y": elevation.y,
            "x": elevation.x,
            "variables": ["red", "green", "blue", "alpha"],
        },
    )

    return hillshaded_da


def pixel_ebb_flow(satellite_ds, tide_m, offset_min=15):
    """
    Computes whether each pixel in a satellite dataset represents ebb or
    flow tide conditions.

    This function compares the original modelled tide heights from the
    moment of satellite data acquisition with new tide hieghts modelled
    after after shifting the original timesteps forward by `offset_min`
    minutes. If a pixel's tide height is lower at the `tide_m_offset`
    time than at the original time, it is considered to be flowing
    (rising) over time. Otherwise, it is considered to be ebbing
    (falling) over time.

    Parameters
    ----------
    tide_m : xarray.DataArray
        An array containing modelled tide heights for each pixel and
        timestep in a satellite dataset.
    offset_min : int, optional
        The time offset in minutes to use for the comparison (default
        is 15).

    Returns
    -------
    ebb_flow_da : xarray.DataArray
        A new DataArray with the same dimensions as `tide_m` containing
        a boolean value for each pixel indicating whether the tide is
        ebbing (False) or flowing (True).
    tide_m_offset : xarray.DataArray
        An array containing modelled tides offset by `offset_min`
        minutes in time, for reference.
    """

    # Offset times in original array by X minutes forward in time
    times_offset = tide_m.time + np.timedelta64(offset_min, "m")

    # Model tides into every pixel in the three-dimensional (x, y, time)
    # input array
    tide_m_offset, _ = pixel_tides(tide_m, times=times_offset, resample=True)

    # Restore original times so both arrays can be combined using xarray
    tide_m_offset["time"] = tide_m["time"]

    # For each pixel, test whether original tides were lower than
    # tides offset to X minutes later. If they were lower, the tide was
    # "flowing" (rising) over time, if they were higher the tide was
    # "ebbing" (falling) over time
    ebb_flow_da = (tide_m < tide_m_offset).rename("ebb_flow")

    return ebb_flow_da, tide_m_offset


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


# def load_config(config_path: str) -> dict:
#     """
#     Loads a YAML config file and returns data as a nested dictionary.

#     config_path can be a path or URL to a web accessible YAML file
#     """
#     with fsspec.open(config_path, mode="r") as f:
#         config = yaml.safe_load(f)
#     return config
