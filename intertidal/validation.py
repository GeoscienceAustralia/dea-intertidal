import numpy as np


def eval_metrics(x, y, round=3, all_regress=False):
    """
    Calculate a set of common statistical metrics
    based on two input actual and predicted vectors.

    These include:
        - Pearson correlation
        - Root Mean Squared Error
        - Mean Absolute Error
        - R-squared
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
        "R-squared": r2_score(xy_df.x, xy_df.y),
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
    vmin=None,
    vmax=None,
    display_map=True,
    return_map=False,
    backend="folium",
):
    """
    Plot raster data over an interactive map.
    """
    
    if backend == "folium":

        # Create folium map
        import folium

        m = folium.Map()

    elif backend == "ipyleaflet":

        # Create ipyleaflet map
        import ipyleaflet

        m = ipyleaflet.Map()

    # Reproject data to epsg:3857 and add to map
    ds.odc.reproject("epsg:3857", dst_nodata=np.nan).odc.add_to(
        m, opacity=1.0, vmin=vmin, vmax=vmax
    )

    # Snap map to data bounds
    m.fit_bounds(ds.odc.map_bounds())

    # Return map if requested
    if return_map:
        return m

    # Display map if requested
    if display_map:
        display(m)
