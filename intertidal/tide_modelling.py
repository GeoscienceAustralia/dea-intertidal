import xarray as xr
import geopandas as gpd

from dea_tools.coastal import pixel_tides, _pixel_tides_resample
from dea_tools.spatial import xr_interpolate


def pixel_tides_ensemble(
    ds,
    ancillary_points,
    model="ensemble",
    top_n=3,
    interp_method="idw",
    reduce_method="mean",
    ancillary_valid_perc=0.02,
    **pixel_tides_kwargs,
):
    """
    Generate an ensemble tide model, combining the top local tide models
    for any coastal location using ancillary point data (e.g. altimetry
    observations or NDWI correlations along the coastline).

    This function generates an ensemble of tidal height predictions for
    each pixel in a satellite dataset. Firstly, tides from multiple tide
    models are modelled into a low resolution grid using `pixel_tides`.
    Ancillary point data is then loaded and interpolated to the same
    grid to serve as weightings. These weightings are used to retain
    only the top N tidal models, and remaining top models are reduced/
    combined into a single ensemble output for each time/x/y.
    The resulting ensemble tides are then resampled and reprojected to
    match the high-resolution satellite data.

    Parameters:
    -----------
    ds : xarray.Dataset
        A dataset whose geobox (`ds.odc.geobox`) will be used to define
        the spatial extent of the low resolution tide modelling grid.
    ancillary_points : str
        Path to a file containing point correlations for different tidal
        models.
    model : list or None, optional
        The default of "ensemble" will combine results from "FES2014",
        "FES2012", "EOT20", "TPXO8-atlas-v1", "TPXO9-atlas-v5", "GOT4.10"
        and "HAMTIDE11" into a single locally optimised ensemble model.
        All other options will skip the ensemble model step and
        run `pixel_tides` directly instead.
    top_n : integer, optional
        The number of top models to use in the ensemble calculation.
        Default is 3, which will reduce values from the top 3 models.
    interp_method : str, optional
        Interpolation method used to interpolate correlations onto the
        low-resolution tide grid. Default is "nearest".
    reduce_method : str, optional
        Method used to reduce values from the `top_n` tide models into
        a single enemble output. Defaults to "mean", supports "median".
    ancillary_valid_perc : float, optional
        The minimum valid percent used to filter input point correlations.
        Defaults to 0.02.
    **pixel_tides_kwargs
        All other optional keyword arguments to provide to the underlying
        `pixel_tides` function.

    Returns:
    --------
    tides_highres_ensemble : xarray.Dataset
        High-resolution ensemble tidal heights dataset.
    tides_lowres_ensemble : xarray.Dataset
        Low-resolution ensemble tidal heights dataset.
    weights_ds : xarray.Dataset
        Dataset containing weights for each tidal model used in the ensemble.
    """

    # Run pixel tides directly if "ensemble" is not specified
    if (model != "ensemble") & ("ensemble" not in model):
        return pixel_tides(
            ds,
            model=model,
            **pixel_tides_kwargs,
        )

    # Otherwise, run pixel tides on all models in preperation for
    # ensemble tide modelling
    else:
        print("Running ensemble tide modelling")
        # Extract the `resample` param if it exists so we can run
        # `pixel_tides` with `resample=False`, and then resample later
        resample_param = pixel_tides_kwargs.pop("resample", True)

        # Run `pixel_tides` on all tide models and return low-res output
        ensemble_models = [
            "FES2014",
            "FES2012",
            "TPXO8-atlas-v1",
            "TPXO9-atlas-v5",
            "EOT20",
            "HAMTIDE11",
            "GOT4.10",
        ]
        tides_lowres = pixel_tides(
            ds,
            resample=False,
            model=ensemble_models,
            **pixel_tides_kwargs,
        )

    # Load ancillary points from file, filter by minimum valid data perc
    # and drop any empty points/unnecessary columns
    print("Generating ensemble tide model from point inputs")
    corr_gdf = (
        gpd.read_file(ancillary_points)
        .query(f"valid_perc > {ancillary_valid_perc}")
        .dropna()[ensemble_models + ["geometry"]]
    )

    # Spatially interpolate each tide model
    print(f"Interpolating model weights using '{interp_method}' interpolation")
    weights_ds = xr_interpolate(
        tides_lowres, gdf=corr_gdf, columns=ensemble_models, method=interp_method
    ).to_array(dim="tide_model")

    # Print models in order of correlation
    print(
        weights_ds.drop("spatial_ref")
        .mean(dim=["x", "y"])
        .to_dataframe("weights")
        .sort_values("weights", ascending=False)
    )

    # Mask out all but the top N models
    tides_top_n = tides_lowres.where(
        (weights_ds.rank(dim="tide_model") > (len(ensemble_models) - top_n))
    )

    # Reduce remaining models to produce a single ensemble output
    # for each time/x/y
    if reduce_method == "median":
        print("Reducing multiple models into single ensemble model using 'median'")
        tides_lowres_ensemble = tides_top_n.median("tide_model")
    elif reduce_method == "mean":
        print("Reducing multiple models into single ensemble model using 'mean'")
        tides_lowres_ensemble = tides_top_n.mean("tide_model")

    # Optionally resample/reproject ensemble tides to match high-res
    # satellite data
    if resample_param:
        print("Reprojecting ensemble tides into original array")
        tides_highres_ensemble, _ = _pixel_tides_resample(
            tides_lowres=tides_lowres_ensemble, ds=ds
        )
        return tides_highres_ensemble, tides_lowres_ensemble

    else:
        print("Returning low resolution ensemble tide array")
        return tides_lowres_ensemble
