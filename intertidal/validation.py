import numpy as np
from odc.algo import mask_cleanup


def eval_metrics(x, y, round=3, all_regress=False):
    """
    Calculate a set of common statistical metrics
    based on two input actual and predicted vectors.

    These include:
        - Pearson correlation
        - Root Mean Squared Error
        - Mean Absolute Error
        - R-squared
        - Bias
        - Linear regression parameters (slope,
          p-value, intercept, standard error)

    Parameters
    ----------
    x : numpy.array
        An array providing "actual" variable values
    y : numpy.array
        An array providing "predicted" variable values
    round : int
        Number of decimal places to round each metric
        to. Defaults to 3
    all_regress : bool
        Whether to return linear regression p-value,
        intercept and standard error (in addition to
        only regression slope). Defaults to False

    Returns
    -------
    A pandas.Series containing calculated metrics
    """

    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from math import sqrt
    from scipy import stats

    # Create dataframe to drop na
    xy_df = pd.DataFrame({"x": x, "y": y}).dropna()

    # Compute linear regression
    lin_reg = stats.linregress(x=xy_df.x, y=xy_df.y)

    # Calculate statistics
    stats_dict = {
        "Correlation": xy_df.corr().iloc[0, 1],
        "RMSE": sqrt(mean_squared_error(xy_df.x, xy_df.y)),
        "MAE": mean_absolute_error(xy_df.x, xy_df.y),
        "R-squared": lin_reg.rvalue**2,
        "Bias": (xy_df.y - xy_df.x).mean(),
        "Regression slope": lin_reg.slope,
    }

    # Additional regression params
    if all_regress:
        stats_dict.update(
            {
                "Regression p-value": lin_reg.pvalue,
                "Regression intercept": lin_reg.intercept,
                "Regression standard error": lin_reg.stderr,
            }
        )

    # Return as
    return pd.Series(stats_dict).round(round)


def map_raster(
    ds,
    dst_nodata=np.nan,
    cmap="viridis",
    vmin=None,
    vmax=None,
    display_map=True,
    return_map=False,
):
    """
    Plot raster data over an interactive map.
    """
    import folium
    import odc.geo.xr

    # Turn dataset and visualisation params into list if not already
    ds = [ds] if not isinstance(ds, list) else ds
    cmap = [cmap] if not isinstance(cmap, list) else cmap
    vmin = [vmin] if not isinstance(vmin, list) else vmin
    vmax = [vmax] if not isinstance(vmax, list) else vmax

    # Multiply out visualisation params to length of ds
    cmap = cmap * len(ds) if len(cmap) == 1 else cmap
    vmin = vmin * len(ds) if len(vmin) == 1 else vmin
    vmax = vmax * len(ds) if len(vmax) == 1 else vmax

    # Create folium map
    m = folium.Map(control=True)

    # Add satellite imagery basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Loop through each item in list
    for i, ds_i in enumerate(ds):
        # Reproject data to EPSG:3857 and add to map
        layer = ds_i.odc.reproject("epsg:3857", dst_nodata=dst_nodata).odc.add_to(
            m, cmap=cmap[i], opacity=1.0, vmin=vmin[i], vmax=vmax[i]
        )

        # Use name from dataset if available, otherwise "layer 1")
        if ds_i.name is not None:
            layer.layer_name = ds_i.name
        else:
            layer.layer_name = f"layer {i + 1}"

    # Add a layer control
    folium.LayerControl().add_to(m)

    # Snap map to bounds of final dataset
    bounds = ds_i.odc.map_bounds()
    m.fit_bounds(bounds)

    # Return map if requested
    if return_map:
        return m

    # Display map if requested
    if display_map:
        display(m)


def preprocess_validation(modelled_ds, validation_ds, clean_slope=True):
    # Remove zero slope areas
    if clean_slope:
        import xrspatial.slope
        
        # Calculate slope then identify invalid flat areas that are
        # highly likely to be ocean. Buffer these by 1 pixel so we
        # remove any pixels partially obscured by ocean.
        validation_slope = xrspatial.slope(agg=validation_ds)
        validation_flat = mask_cleanup(
            validation_slope == 0, mask_filters=[("dilation", 1)]
        )
        validation_ds = validation_ds.where(~validation_flat)

    # Analyse only pixels that contain valid data in both
    modelled_nodata = modelled_ds.isnull()
    validation_nodata = validation_ds.isnull()

    # Export 1D modelled and validation data for valid data area
    invalid_data = modelled_nodata | validation_nodata
    validation_z = validation_ds.values[~invalid_data.values]
    modelled_z = modelled_ds.values[~invalid_data.values]

    return validation_z, modelled_z
