import numpy as np
import xarray as xr
import geopandas as gpd
from dea_tools.spatial import subpixel_contours

## From https://github.com/GeoscienceAustralia/dea-coastlines/blob/stable/coastlines/vector.py#L707-## L748. The points_on_line func can be removed once a PR to have it added to dea_tools.spatial is
## approved and added to the master branch of dea_notebooks. CP 17/03/2023

def points_on_line(gdf, index, distance=30):
    """
    Generates evenly-spaced point features along a specific line feature
    in a `geopandas.GeoDataFrame`.
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        A `geopandas.GeoDataFrame` containing line features with an
        index and CRS.
    index : string or int
        An value giving the index of the line to generate points along
    distance : integer or float, optional
        A number giving the interval at which to generate points along
        the line feature. Defaults to 30, which will generate a point
        at every 30 metres along the line.
    Returns:
    --------
    points_gdf : geopandas.GeoDataFrame
        A `geopandas.GeoDataFrame` containing point features at every
        `distance` along the selected line.
    """

    # Select individual line to generate points along
    line_feature = gdf.loc[[index]].geometry

    # If multiple features are returned, take unary union
    if line_feature.shape[0] > 0:
        line_feature = line_feature.unary_union
    else:
        line_feature = line_feature.iloc[0]

    # Generate points along line and convert to geopandas.GeoDataFrame
    points_line = [
        line_feature.interpolate(i)
        for i in range(0, int(line_feature.length), distance)
    ]
    points_gdf = gpd.GeoDataFrame(geometry=points_line, crs=gdf.crs)

    return points_gdf

def tidal_offset_tidelines (ds, distance = 10):
    '''
    This function extracts high and low tidelines from a `xarray.Dataset`,
    calculates the tidal offsets at each point on the lines, and returns
    the offset values in separate `geopandas.GeoDataFrame` objects.

    Parameters
    ----------
    ds : xarray.Dataset
        A `xarray.Dataset` containing 'ht_offset' and 'lt_offset' DataArrays
        and an 'Extents' DataArray containing binary shoreline information.
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
    ## Extract the high/low tide boundaries
    tidelines_gdf = subpixel_contours(da=ds['Extents'], z_values=[0.5,1.5])
    
    ## Translate the high/Low tidelines into point data at regular intervals
    lowtideline = points_on_line(tidelines_gdf, 0, distance=distance)
    hightideline = points_on_line(tidelines_gdf, 1, distance=distance)

    ## Extract the point coordinates into xarray for the hightideline dataset
    x_indexer_high = xr.DataArray(hightideline.centroid.x, dims=['point'])
    y_indexer_high = xr.DataArray(hightideline.centroid.y, dims=['point'])
    
    ## Extract the point coordinates into xarray for the lowtideline dataset
    x_indexer_low = xr.DataArray(lowtideline.centroid.x, dims=['point'])
    y_indexer_low = xr.DataArray(lowtideline.centroid.y, dims=['point'])

    ## Extract the high or low tide offset at each point in the high and low tidelines respectively
    ## From https://stackoverflow.com/questions/67425567/extract-values-from-xarray-dataset-using-geopandas-multilinestring
    highlineoffset = ds.ht_offset.sel(x=x_indexer_high, y=y_indexer_high, method='nearest')
    lowlineoffset = ds.lt_offset.sel(x=x_indexer_low, y=y_indexer_low, method='nearest')

    ## Replace the offset values per point into the master dataframes
    hightideline['ht_offset'] = highlineoffset
    lowtideline['lt_offset'] = lowlineoffset
    
    ## Consider adding the points to the tidelines
    ## https://gis.stackexchange.com/questions/448788/merging-points-to-linestrings-using-geopandas
    
    return hightideline, lowtideline, tidelines_gdf
    