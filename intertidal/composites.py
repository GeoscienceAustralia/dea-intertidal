import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import click

import datacube
import odc.geo.xr
from odc.algo import (
    int_geomedian,
    enum_to_bool,
    erase_bad,
    keep_good_only,
)
from datacube.utils.aws import configure_s3_access

from dea_tools.coastal import pixel_tides
from dea_tools.dask import create_local_dask_cluster

from intertidal.utils import configure_logging
from intertidal.elevation import load_data


def intertidal_composites(
    satellite_ds,
    threshold_lowtide=0.2,
    threshold_hightide=0.8,
    max_iters=10000,
    tide_model="FES2014",
    tide_model_dir="/var/share/tide_models",
    study_area=None,
    log=None,
):
    """
    Calculates Geometric Median composites of the coastal zone at low
    and high tide using satellite imagery and tidal modeling.

    This function uses tools from `odc.algo` to keep data in its
    original integer datatype until the last possible moment, improving
    the efficiency of Dask processing.

    Based on the method described in:

    Sagar, S., Phillips, C., Bala, B., Roberts, D., & Lymburner, L.
    (2018). Generating Continental Scale Pixel-Based Surface Reflectance
    Composites in Coastal Regions with the Use of a Multi-Resolution
    Tidal Model. Remote Sensing, 10, 480. https://doi.org/10.3390/rs10030480

    Parameters
    ----------
    satellite_ds : xarray.Dataset
        A satellite data time series containing spectral bands.
    threshold_lowtide : float, optional
        Quantile used to identify low tide observations, by default 0.2.
    threshold_hightide : float, optional
        Quantile used to identify high tide observations, by default 0.8.
    max_iters : int, optional
        Value to pass to the 'max_iters' param of `int_geomedian`. This
        can be set to a low value (e.g. 10) to increase the processing
        speed of test runs.
    tide_model : str, optional
        The tide model or a list of models used to model tides, as
        supported by the `pyTMD` Python package. Options include:
        - "FES2014" (default; pre-configured on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `dea_tools.coastal.model_tides`.
    study_area : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    ds_lowtide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the lowest X quantile tide heights for each pixel.
    ds_hightide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the highest X quantile tide values for each pixel.
    """

    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use study area name for logs if it exists
    if study_area is not None:
        log_prefix = f"Study area {study_area}: "
    else:
        log_prefix = ""

    # Model tides into every pixel in the three-dimensional (x by y by
    # time) satellite dataset
    log.info(f"Study area {study_area}: Modelling tide heights for each pixel")
    tides_highres, tides_lowres = pixel_tides(
        satellite_ds,
        resample=True,
        model=tide_model,
        directory=tide_model_dir,
        cutoff=np.inf,
    )

    # Convert tides to a Dask array so we can use it in `keep_good_only`
    tides_highres_dask = tides_highres.chunk(chunks=satellite_ds.nbart_red.data.chunks)

    # Calculate low and high tide height thresholds using quantile of
    # all tide observations.
    log.info(f"Study area {study_area}: Calculate low and high tide height thresholds")
    threshhold_ds = (
        tides_lowres.quantile(q=[threshold_lowtide, threshold_hightide], dim="time")
        .odc.assign_crs(satellite_ds.odc.geobox.crs)
        .odc.reproject(satellite_ds.odc.geobox, resampling="bilinear")
        .drop("quantile")
    )

    # Apply threshold to keep only pixels with tides less or greater than
    # than tide height threshold
    log.info(f"Study area {study_area}: Masking to low and high tide observations")
    low_mask = tides_highres_dask <= threshhold_ds.isel(quantile=0)
    high_mask = tides_highres_dask >= threshhold_ds.isel(quantile=-1)

    # Mask out pixels outside of selected tides. Drop fully empty scenes
    # to speed up geomedian
    ds_low = keep_good_only(x=satellite_ds, where=low_mask).sel(
        time=low_mask.any(dim=["x", "y"])
    )
    ds_high = keep_good_only(x=satellite_ds, where=high_mask).sel(
        time=high_mask.any(dim=["x", "y"])
    )

    # Calculate low and high tide geomedians
    log.info(f"Study area {study_area}: Calculating geomedians")
    num_threads = os.cpu_count() - 2
    ds_lowtide = int_geomedian(ds=ds_low, maxiters=max_iters, num_threads=num_threads)
    ds_hightide = int_geomedian(ds=ds_high, maxiters=max_iters, num_threads=num_threads)

    return ds_lowtide, ds_hightide


@click.command()
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a GridSpec tile ID (e.g. in the form "
    "'x143y56') to run the analysis on.",
)
@click.option(
    "--start_date",
    type=str,
    default="2020",
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. ",
)
@click.option(
    "--end_date",
    type=str,
    default="2022",
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. ",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="The spatial resolution in metres used to load satellite "
    "data and produce intertidal outputs. Defaults to 10 metre "
    "Sentinel-2 resolution.",
)
@click.option(
    "--threshold_lowtide",
    type=float,
    default=0.2,
    help="The quantile used to identify low tide observations. " "Defaults to 0.2.",
)
@click.option(
    "--threshold_hightide",
    type=float,
    default=0.8,
    help="The quantile used to identify high tide observations. " "Defaults to 0.8.",
)
@click.option(
    "--tide_model",
    type=str,
    multiple=True,
    default=["FES2014"],
    help="The tide model used to model tides, as supported by the "
    "`pyTMD` Python package. Options include 'FES2014' (default), "
    "'TPXO8-atlas' and 'TPXO9-atlas-v5'. This parameter can be "
    "repeated to request multiple models, e.g.: "
    "`--tide_model FES2014 --tide_model TPXO9-atlas-v5`.",
)
@click.option(
    "--tide_model_dir",
    type=str,
    default="/var/share/tide_models",
    help="The directory containing tide model data files. Defaults to "
    "'/var/share/tide_models'; for more information about the required "
    "directory structure, refer to `dea_tools.coastal.model_tides`.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    type=bool,
    default=True,
    help="Whether to use sign AWS requests for S3 access",
)
def intertidal_composites_cli(
    study_area,
    start_date,
    end_date,
    resolution,
    threshold_lowtide,
    threshold_hightide,
    tide_model,
    tide_model_dir,
    aws_unsigned,
):
    log = configure_logging(
        f"Study area {study_area}: Generating Intertidal composites"
    )

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    # Create output folder. If it doesn't exist, create it
    output_dir = f"data/interim/{study_area}/{start_date}-{end_date}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        log.info(f"Study area {study_area}: Loading satellite data")

        # Connect to datacube to access data
        dc = datacube.Datacube(app="Intertidal_composites_CLI")

        # Lazily load Sentinel-2 satellite data
        satellite_ds = load_data(
            dc=dc,
            study_area=study_area,
            time_range=(start_date, end_date),
            resolution=resolution,
            crs="EPSG:3577",
            include_s2=True,
            include_ls=False,
            filter_gqa=False,
            ndwi=False,
        )

        # Calculate high and low tide geomedian composites
        log.info(f"Study area {study_area}: Running geomedians")
        ds_lowtide, ds_hightide = intertidal_composites(
            satellite_ds=satellite_ds,
            threshold_lowtide=threshold_lowtide,
            threshold_hightide=threshold_hightide,
            max_iters=10,
            study_area=study_area,
            log=log,
        )

        # Create local dask cluster to improve data load time
        client = create_local_dask_cluster(return_client=True)

        # Process and load low and high tide composites using Dask
        log.info(f"Study area {study_area}: Processing low tide composite")
        ds_lowtide.load()
        log.info(f"Study area {study_area}: Processing high tide composite")
        ds_hightide.load()

        # Close dask client
        client.close()

        # Export layers as GeoTIFFs
        log.info(f"Study area {study_area}: Exporting outputs GeoTIFFs to {output_dir}")

        prefix = f"{output_dir}/{study_area}_{start_date}_{end_date}"
        ds_lowtide.to_array().odc.write_cog(
            f"{prefix}_composite_lowtide_{int(threshold_lowtide * 100)}.tif",
            overwrite=True,
        )
        ds_hightide.to_array().odc.write_cog(
            f"{prefix}_composite_hightide_{int(threshold_hightide * 100)}.tif",
            overwrite=True,
        )

        # Export as images
        prefix = f"data/figures/{study_area}_{start_date}_{end_date}"
        ds_lowtide.odc.to_rgba(
            bands=["nbart_red", "nbart_green", "nbart_blue"], vmin=100, vmax=2500
        ).plot.imshow().figure.savefig(
            f"{prefix}_composite_lowtide_{int(threshold_lowtide * 100)}_rgb.png"
        )
        ds_hightide.odc.to_rgba(
            bands=["nbart_red", "nbart_green", "nbart_blue"], vmin=100, vmax=2500
        ).plot.imshow().figure.savefig(
            f"{prefix}_composite_hightide_{int(threshold_hightide * 100)}_rgb.png"
        )

        # Workflow completed
        log.info(
            f"Study area {study_area}: Completed DEA Intertidal composites workflow"
        )

    except Exception as e:
        log.exception(f"Study area {study_area}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    intertidal_composites_cli()
