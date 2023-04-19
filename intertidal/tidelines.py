from dea_tools.spatial import subpixel_contours #xr_vectorize

## From https://github.com/GeoscienceAustralia/dea-coastlines/blob/stable/coastlines/vector.py#L707-L748
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

def tidal_offset_tidelines (ds):
    '''
    
    '''
    ## High/Low tideline extraction
    tidelines_gdf = subpixel_contours(da=ds['Extents'], z_values=[0.5,1.5])

    lowtideline = points_on_line(tidelines_gdf, 0, distance=10)
    hightideline = points_on_line(tidelines_gdf, 1, distance=10)

    ## From https://stackoverflow.com/questions/67425567/extract-values-from-xarray-dataset-using-geopandas-multilinestring

    x_indexer_high = xr.DataArray(hightideline.centroid.x, dims=['point'])
    y_indexer_high = xr.DataArray(hightideline.centroid.y, dims=['point'])

    x_indexer_low = xr.DataArray(lowtideline.centroid.x, dims=['point'])
    y_indexer_low = xr.DataArray(lowtideline.centroid.y, dims=['point'])

    highlineoffset = ds.ht_offset.sel(x=x_indexer_high, y=y_indexer_high, method='nearest')
    lowlineoffset = ds.lt_offset.sel(x=x_indexer_low, y=y_indexer_low, method='nearest')

    hightideline['ht_offset'] = highlineoffset
    lowtideline['lt_offset'] = lowlineoffset
    
    return hightideline, lowtideline
    