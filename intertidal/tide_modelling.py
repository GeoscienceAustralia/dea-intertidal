import xarray as xr
import geopandas as gpd

from dea_tools.coastal import pixel_tides, _pixel_tides_resample
from dea_tools.spatial import interpolate_2d


def pixel_tides_ensemble(
    satellite_ds,
    directory,
    ancillary_points,
    top_n=3,
    models=None,
    interp_method="nearest",
    times=None,
    calculate_quantiles=None,
    cutoff=None,
    **pixel_tides_kwargs,
):
    """
    Generate an ensemble tide model, choosing the best three tide models
    for any coastal location using ancillary point data (e.g. altimetry
    observations or NDWI correlations along the coastline).

    This function generates an ensemble of tidal height predictions for
    each pixel in a satellite dataset. Firstly, tides from multiple tide
    models are modelled into a low resolution grid using `pixel_tides`.
    Ancillary point data is then loaded and interpolated to the same
    grid to serve as weightings. These weightings are used to retain
    only the top three tidal models, and remaining top models are
    combined into a single ensemble output for each time/x/y.
    The resulting ensemble tides are then resampled and reprojected to
    match the high-resolution satellite data.

    Parameters:
    -----------
    satellite_ds : xarray.Dataset
        Three-dimensional dataset containing satellite-derived
        information (x by y by time).
    directory : str
        Directory containing tidal model data; see `pixel_tides`.
    ancillary_points : str
        Path to a file containing point correlations for different tidal
        models.
    times : tuple or None, optional
        Tuple containing start and end time of time range to be used for
        tide model in the format of "YYYY-MM-DD".
    top_n : integer, optional
        The number of top models to use in the ensemble calculation.
        Default is 3, which will calculate a median of the top 3 models.
    models : list or None, optional
        An optional list of tide models to use for the ensemble model.
        Default is None, which will use "FES2014", "FES2012", "EOT20",
        "TPXO8-atlas-v1", "TPXO9-atlas-v5", "HAMTIDE11", "GOT4.10".
    interp_method : str, optional
        Interpolation method used to interpolate correlations onto the
        low-resolution tide grid. Default is "nearest".
    **pixel_tides_kwargs
        Optional keyword arguments to provide to the `pixel_tides` function.

    Returns:
    --------
    tides_highres : xarray.Dataset
        High-resolution ensemble tidal heights dataset.
    weights_ds : xarray.Dataset
        Dataset containing weights for each tidal model used in the ensemble.
    """
    # Use default models if none provided
    if models is None:
        models = [
            "FES2014",
            "FES2012",
            "TPXO8-atlas-v1",
            "TPXO9-atlas-v5",
            "EOT20",
            "HAMTIDE11",
            "GOT4.10",
        ]

    tide_lowres = pixel_tides(
        satellite_ds,
        resample=False,
        calculate_quantiles=calculate_quantiles,
        times=times,
        model=models,
        directory=directory,
        cutoff=cutoff,
        **pixel_tides_kwargs,
    )

    # Load ancillary points from file, reproject to match satellite
    # data, and drop empty points
    print("Generating ensemble tide model from point inputs")
    corr_gdf = (
        gpd.read_file(ancillary_points)[models + ["geometry"]]
        .to_crs(satellite_ds.odc.crs)
        .dropna()
    )

    # Loop through each model, interpolating correlations into
    # low-res tide grid
    out_list = []

    for model in models:
        out = interpolate_2d(
            tide_lowres,
            x_coords=corr_gdf.geometry.x,
            y_coords=corr_gdf.geometry.y,
            z_coords=corr_gdf[model],
            method=interp_method,
        ).expand_dims({"tide_model": [model]})

        out_list.append(out)

    # Combine along tide model dimension into a single xarray.Dataset
    weights_ds = xr.concat(out_list, dim="tide_model")

    # Mask out all but the top N models, then take median of remaining
    # to produce a single ensemble output for each time/x/y
    tide_lowres_ensemble = tide_lowres.where(
        (weights_ds.rank(dim="tide_model") > (len(models) - top_n))
    ).median("tide_model")

    # Resample/reproject ensemble tides to match high-res satellite data
    tides_highres, tides_lowres = _pixel_tides_resample(
        tides_lowres=tide_lowres_ensemble,
        ds=satellite_ds,
    )

    return tides_highres, weights_ds
