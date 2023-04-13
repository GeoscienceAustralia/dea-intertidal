import numpy as np
import xarray as xr
import geopandas as gpd
from dea_tools.spatial import subpixel_contours, points_on_line


def tidal_offset_tidelines (extents, 
                            offset_hightide,
                            offset_lowtide,
                            distance=10):
    '''
    This function extracts high and low tidelines from a rasterised 
    'sometimes wet' layer in the extents input xr.DataArray,
    calculates the tidal offsets at each point on the lines, and returns
    the offset values in separate `geopandas.GeoDataFrame` objects.

    Parameters
    ----------
    extents : xarray.DataArray
        An xarray.DataArray containing binary shoreline information,
        depicting always, sometimes and never wet pixels.
    offset_hightide: xarray.DataArray
        An xarray.DataArray containing the percentage high-tide offset of the 
        satellite observed tide heights from the modelled heights.
    offset_lowtide: xarray.DataArray
        An xarray.DataArray containing the percentage low-tide offset of the 
        satellite observed tide heights from the modelled heights.
    distance : integer or float, optional
        A number giving the interval at which to generate points along
        the line feature. Defaults to 10, which will generate a point
        at every 10 metres along the line.

    Returns
    -------
    tuple
        A tuple of two `geopandas.GeoDataFrame` objects containing the
        high and low tidelines with their respective tidal offsets and
        a `geopandas.GeoDataFrame` containing the multilinestring tidelines.   
    '''
    # Extract the high/low tide boundaries
    tidelines_gdf = subpixel_contours(da=extents, z_values=[0.5, 1.5])
    
    # Translate the high/Low tidelines into point data at regular intervals
    lowtideline = points_on_line(tidelines_gdf, 0, distance=distance)
    hightideline = points_on_line(tidelines_gdf, 1, distance=distance)

    # Extract the point coordinates into xarray for the hightideline dataset
    x_indexer_high = xr.DataArray(hightideline.centroid.x, dims=['point'])
    y_indexer_high = xr.DataArray(hightideline.centroid.y, dims=['point'])
    
    # Extract the point coordinates into xarray for the lowtideline dataset
    x_indexer_low = xr.DataArray(lowtideline.centroid.x, dims=['point'])
    y_indexer_low = xr.DataArray(lowtideline.centroid.y, dims=['point'])

    # Extract the high or low tide offset at each point in the high and low tidelines respectively
    # From https://stackoverflow.com/questions/67425567/extract-values-from-xarray-dataset-using-geopandas-multilinestring
    highlineoffset = offset_hightide.sel(x=x_indexer_high, y=y_indexer_high, method='nearest')
    lowlineoffset = offset_lowtide.sel(x=x_indexer_low, y=y_indexer_low, method='nearest')

    # Replace the offset values per point into the master dataframes
    hightideline['offset_hightide'] = highlineoffset
    lowtideline['offset_lowtide'] = lowlineoffset
    
    ## Consider adding the points to the tidelines
    ## https://gis.stackexchange.com/questions/448788/merging-points-to-linestrings-using-geopandas
    
    return hightideline, lowtideline, tidelines_gdf
    