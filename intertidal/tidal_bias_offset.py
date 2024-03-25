import xarray as xr
import numpy as np

from dea_tools.spatial import subpixel_contours, points_on_line


def bias_offset(tide_m, tide_cq, lat_hat=True, lot_hot=None):
    """
    Calculate the pixel-based sensor-observed spread and high/low
    offsets in tide heights compared to the full modelled tide range.
    Optionally, also return the highest and lowest astronomical and
    sensor-observed tides for each pixel.

    Parameters
    ----------
    tide_m : xr.DataArray
        An xarray.DataArray representing sensor observed tide heights
        for each pixel. Should have 'time', 'x' and 'y' in its
        dimensions.
    tide_cq : xr.DataArray
        An xarray.DataArray representing modelled tidal heights for
        each pixel. Should have 'quantile', 'x' and 'y' in its
        dimensions.
    lat_hat : bool, optional
        Lowest/highest astronomical tides. This work considers the
        modelled tides to be equivalent to the astronomical tides.
        Default is True.
    lot_hot : bool, optional
        Lowest/highest sensor-observed tides. Default is None.

    Returns
    -------
    Depending on the values of `lat_hat` and `lot_hot`, returns a tuple
    with some or all of the following as xarray.DataArrays:
        * `lat`: The lowest astronomical tide.
        * `hat`: The highest astronomical tide.
        * `lot`: The lowest sensor-observed tide.
        * `hot`: The highest sensor-observed tide.
        * `spread`: The spread of the observed tide heights as a
        percentage of the modelled tide heights.
        * `offset_lowtide`: The low tide offset measures the offset of the
        sensor-observed lowest tide from the minimum modelled tide.
        * `offset_hightide`: The high tide measures the offset of the
        sensor-observed highest tide from the maximum modelled tide.
    """

    # Set the maximum and minimum values per pixel for the observed and
    # modelled datasets
    max_obs = tide_m.max(dim="time")
    min_obs = tide_m.min(dim="time")
    max_mod = tide_cq.max(dim="quantile")
    min_mod = tide_cq.min(dim="quantile")

    # Set the maximum range in the modelled and observed tide heights
    mod_range = max_mod - min_mod
    obs_range = max_obs - min_obs

    # Calculate the spread of the observed tide heights as a percentage
    # of the modelled tide heights
    spread = obs_range / mod_range * 100

    # Calculate the high and low tide offset of the observed tide
    # heights as a percentage of the modelled highest and lowest tides.
    offset_hightide = (abs(max_mod - max_obs)) / mod_range * 100
    offset_lowtide = (abs(min_mod - min_obs)) / mod_range * 100
    
    # Add the lowest and highest astronomical tides
    if lat_hat:
        lat = min_mod
        hat = max_mod

    # Add the lowest and highest sensor-observed tides
    if lot_hot:
        lot = min_obs
        hot = max_obs

    if lat_hat:
        if lot_hot:
            return lat, hat, lot, hot, spread, offset_lowtide, offset_hightide
        else:
            return lat, hat, spread, offset_lowtide, offset_hightide
    elif lot_hot:
        return lot, hot, spread, offset_lowtide, offset_hightide
    else:
        return spread, offset_lowtide, offset_hightide


# def tidal_offset_tidelines(extents, offset_hightide, offset_lowtide, distance=500):
#     """
#     This function extracts high and low tidelines from a rasterised
#     'sometimes wet' layer in the extents input xr.DataArray,
#     calculates the tidal offsets at each point on the lines, and returns
#     the offset values in separate `geopandas.GeoDataFrame` objects.

#     Parameters
#     ----------
#     extents : xarray.DataArray
#         An xarray.DataArray containing binary shoreline information,
#         depicting always, sometimes and never wet pixels.
#     offset_hightide: xarray.DataArray
#         An xarray.DataArray containing the percentage high-tide offset of the
#         satellite observed tide heights from the modelled heights.
#     offset_lowtide: xarray.DataArray
#         An xarray.DataArray containing the percentage low-tide offset of the
#         satellite observed tide heights from the modelled heights.
#     distance : integer or float, optional
#         A number giving the interval at which to generate points along
#         the line feature. Defaults to 500, which will generate a point
#         at every 500 metres along the line.

#     Returns
#     -------
#     tuple
#         A tuple of two `geopandas.GeoDataFrame` objects containing the
#         high and low tidelines with their respective tidal offsets and
#         a `geopandas.GeoDataFrame` containing the multilinestring tidelines.
#     """
#     ## Create a three class extents dataset: tidal wet/intertidal/dry
#     extents = extents.where((extents == 0) | (extents == 1) | (extents == 2), np.nan)

#     # Extract the high/low tide boundaries
#     tidelines_gdf = subpixel_contours(da=extents, z_values=[1.5, 0.5])

#     # Translate the high/Low tidelines into point data at regular intervals
#     lowtideline = points_on_line(tidelines_gdf, 0, distance=distance)
#     hightideline = points_on_line(tidelines_gdf, 1, distance=distance)

#     # Extract the point coordinates into xarray for the hightideline dataset
#     x_indexer_high = xr.DataArray(hightideline.centroid.x, dims=["point"])
#     y_indexer_high = xr.DataArray(hightideline.centroid.y, dims=["point"])

#     # Extract the point coordinates into xarray for the lowtideline dataset
#     x_indexer_low = xr.DataArray(lowtideline.centroid.x, dims=["point"])
#     y_indexer_low = xr.DataArray(lowtideline.centroid.y, dims=["point"])

#     # Extract the high or low tide offset at each point in the high and low tidelines respectively
#     # From https://stackoverflow.com/questions/67425567/extract-values-from-xarray-dataset-using-geopandas-multilinestring
#     highlineoffset = offset_hightide.sel(
#         x=x_indexer_high, y=y_indexer_high, method="nearest"
#     )
#     lowlineoffset = offset_lowtide.sel(
#         x=x_indexer_low, y=y_indexer_low, method="nearest"
#     )

#     # Replace the offset values per point into the master dataframes
#     hightideline["offset_hightide"] = highlineoffset
#     lowtideline["offset_lowtide"] = lowlineoffset

#     ## Consider adding the points to the tidelines
#     ## https://gis.stackexchange.com/questions/448788/merging-points-to-linestrings-using-geopandas

#     return hightideline, lowtideline, tidelines_gdf
