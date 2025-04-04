import os
import sys
import numpy as np
import click
import xarray
import datacube
import odc.geo.xr
from odc.geo.geom import BoundingBox
from odc.algo import (
    int_geomedian,
    keep_good_only,
)
from datacube.utils.aws import configure_s3_access
from eo_tides.eo import pixel_tides

# from dea_tools.coastal import pixel_tides
from dea_tools.dask import create_local_dask_cluster

from intertidal.utils import configure_logging
from intertidal.io import (
    load_data,
    prepare_for_export,
    tidal_metadata,
    export_dataset_metadata,
)


# Function to rename the bands
def rename_bands(ds, old_string, new_string):
    # Create a new dataset with renamed bands
    ds_renamed = ds.rename(
        {band: band.replace(old_string, new_string) for band in ds.data_vars}
    )
    return ds_renamed


def tidal_composites(
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
        - "TPXO9-atlas-v5"
        - "TPXO8-atlas"
        - "EOT20"
        - "HAMTIDE11"
        - "GOT4.10"
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

    # Model tides into for spatial extent and timesteps in satellite data
    log.info(f"Study area {study_area}: Modelling tide heights")

    tides_highres = pixel_tides(
        data=satellite_ds,
        model=tide_model,
        resample=True,
        directory=tide_model_dir,
    )

    # Mask tide data with invalid obs in the the satellite data
    tides_highres = tides_highres.where(satellite_ds.nbart_red > -999)

    threshold_ds = tides_highres.quantile(
        [threshold_lowtide, threshold_hightide], dim="time"
    ).drop("quantile")

    # create a mask for selecting satellite obs below the low tide treshold
    low_mask = tides_highres <= threshold_ds.isel(quantile=0)

    # create a mask for selecting satellite obs above the high tide treshold
    high_mask = tides_highres >= threshold_ds.isel(quantile=-1)

    # Mask out pixels outside of selected tides. Drop fully empty scenes
    # to speed up geomedian
    ds_low = keep_good_only(x=satellite_ds, where=low_mask).sel(
        time=low_mask.any(dim=["x", "y"])
    )
    # export low_mask.count and high_mask count
    ds_high = keep_good_only(x=satellite_ds, where=high_mask).sel(
        time=high_mask.any(dim=["x", "y"])
    )

    # Calculate low and high tide geomedians
    log.info(f"Study area {study_area}: Calculating geomedians")
    num_threads = os.cpu_count() - 2

    ds_lowtide = int_geomedian(ds=ds_low, maxiters=max_iters, num_threads=num_threads)
    ds_hightide = int_geomedian(ds=ds_high, maxiters=max_iters, num_threads=num_threads)

    ds_lowtide["low_count_clear"] = ds_low.nbart_red.where(
        ds_low.nbart_red > -999
    ).count(dim=["time"])

    # this didn't work
    # ds_lowtide['low_count_clear2'] =  keep_good_only(x=satellite_ds, where=low_mask).nbart_red.count()

    ds_lowtide["low_threshold"] = threshold_ds.isel(quantile=0)

    ds_hightide["high_threshold"] = threshold_ds.isel(quantile=1)

    # this didn't work
    # ds_hightide['high_count_clear2'] = keep_good_only(x=satellite_ds, where=high_mask).nbart_red.count()

    ds_hightide["high_count_clear"] = ds_high.nbart_red.where(
        ds_high.nbart_red > -999
    ).count(dim=["time"])

    return ds_lowtide, ds_hightide


@click.command()
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a GridSpec tile ID (e.g. in the form "
    "'x123y123') to run the analysis on.",
)
@click.option(
    "--start_date",
    type=str,
    required=True,
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Intertidal, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--end_date",
    type=str,
    required=True,
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Intertidal, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--label_date",
    type=str,
    required=True,
    help="The date used to label output arrays, and to use as the date "
    "assigned to the dataset when indexed into Datacube.",
)
@click.option(
    "--output_version",
    type=str,
    required=True,
    help="The version number to use for output files and metadata (e.g. " "'0.0.1').",
)
@click.option(
    "--output_dir",
    type=str,
    default="data/processed/",
    help="The directory/location to output data and metadata; supports "
    "both local disk and S3 locations. Defaults to 'data/processed/'.",
)
@click.option(
    "--product_maturity",
    type=str,
    default="provisional",
    help="Product maturity metadata to use for the output dataset. "
    "Defaults to 'provisional', can also be 'stable'.",
)
@click.option(
    "--dataset_maturity",
    type=str,
    default="final",
    help="Dataset maturity metadata to use for the output dataset. "
    "Defaults to 'final', can also be 'interim'.",
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
    "--correct_seasonality/--no-correct_seasonality",
    is_flag=True,
    default=False,
    help="If True, remove any seasonal signal from the tide height data "
    "by subtracting monthly mean tide height from each value prior to "
    "correlation calculations. This can reduce false tide correlations "
    "in regions where tide heights correlate with seasonal changes in "
    "surface water. Note that seasonally corrected tides are only used "
    "to identify potentially tide influenced pixels - not for elevation "
    "modelling itself.",
)
@click.option(
    "--mask_sunglint",
    type=int,
    default=0,
    help="Whether to mask out pixels that are likely to be "
    "affected by sunglint using glint angles. Low glint angles "
    "(e.g. < 20) often correspond with sunglint. Defaults to None; "
    "set to e.g. '20' to mask out all pixels with a glint angle of "
    "less than 20.",
)
@click.option(
    "--max_iters",
    type=int,
    default=10000,
    help="Value to pass to the 'max_iters' param of `int_geomedian`. This "
    "can be set to a low value (e.g. 10) to increase the processing "
    "speed of test runs.",
)
@click.option(
    "--tide_model",
    type=str,
    multiple=True,
    default=["FES2014"],
    help="The model used for tide modelling, as supported by the "
    "`pyTMD` Python package. Options include 'FES2014' (default), "
    "'TPXO9-atlas-v5', 'TPXO8-atlas-v1', 'EOT20', 'HAMTIDE11', 'GOT4.10'. ",
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
@click.option(
    "--include_coastal_aerosol/--no-include_coastal_aerosol",
    type=bool,
    default=True,
    help="Whether to include the coastal aerosol band",
)
@click.option(
    "--overwrite/--no-overwrite",
    type=bool,
    default=True,
    help="Whether to include the coastal aerosol band",
)
def tidal_composites_cli(
    study_area,
    start_date,
    end_date,
    label_date,
    output_version,
    output_dir,
    product_maturity,
    dataset_maturity,
    resolution,
    threshold_lowtide,
    threshold_hightide,
    correct_seasonality,
    mask_sunglint,
    max_iters,
    tide_model,
    tide_model_dir,
    aws_unsigned,
    include_coastal_aerosol,
    overwrite,
):

    filename = f"{output_dir}/ga_s2_tidal_composites_cyear_3/{output_version.replace('.','-')}/{study_area[:4]}/{study_area[4:]}/{label_date}--P1Y/ga_s2_tidal_composites_cyear_3_{study_area}_{label_date}--P1Y_final.stac-item.json"

    if mask_sunglint < 1:
        mask_sunglint = None

    process_tile = True
    if overwrite:
        process_tile = True
    else:
        if os.path.exists(filename):
            process_tile = False

    # Create a unique run ID for analysis based on input params and use
    # for logs

    input_params = locals()
    run_id = f"[{output_version}] [{label_date}] [{study_area}]"
    log = configure_logging(run_id)

    # Record params in logs
    log.info(f"{run_id}: Using parameters {input_params}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    if process_tile:
        # Create output folder. If it doesn't exist, create it
        # output_dir = f"data/interim/{study_area}/{start_date}-{end_date}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            log.info(f"{run_id}: Loading satellite data")

            # Create local dask cluster to improve data load time
            client = create_local_dask_cluster(return_client=True)

            # Connect to datacube to load data
            dc = datacube.Datacube(app="Composites_CLI")

            # Use a custom polygon if in testing mode
            if study_area == "testing":
                log.info(f"{run_id}: Running in testing mode using custom study area")
                geom = BoundingBox(
                    467510, -1665790, 468260, -1664840, crs="EPSG:3577"
                ).polygon
            else:
                geom = None

            # Load satellite data and dataset IDs for metadata
            satellite_ds, dss_s2, dss_ls = load_data(
                dc=dc,
                study_area=study_area,
                geom=geom,
                time_range=(start_date, end_date),
                resolution=resolution,
                crs="EPSG:3577",
                include_s2=True,
                include_ls=False,
                filter_gqa=True,
                ndwi=False,
                mask_sunglint=mask_sunglint,
                include_coastal_aerosol=include_coastal_aerosol,
                max_cloudcover=90,
                skip_broken_datasets=True,
                dataset_maturity="final",
                dtype="int16",
            )
            satellite_ds.load()

            # Calculate high and low tide geomedian composites
            log.info(
                f"{run_id}: Study area {study_area}: Running Intertidal composites"
            )
            ds_lowtide, ds_hightide = tidal_composites(
                satellite_ds=satellite_ds,
                threshold_lowtide=threshold_lowtide,
                threshold_hightide=threshold_hightide,
                max_iters=max_iters,
                tide_model=tide_model,
                tide_model_dir=tide_model_dir,
                study_area=study_area,
                log=log,
            )

            # Process and load low and high tide composites using Dask
            log.info(f"Study area {study_area}: Processing low tide composite")
            ds_lowtide.load()
            log.info(f"Study area {study_area}: Processing high tide composite")
            ds_hightide.load()

            ds_hightide = rename_bands(ds_hightide, "nbart", "high")
            ds_hightide = odc.geo.xr.assign_crs(ds_hightide, satellite_ds.odc.crs)

            ds_lowtide = rename_bands(ds_lowtide, "nbart", "low")
            ds_lowtide = odc.geo.xr.assign_crs(ds_lowtide, satellite_ds.odc.crs)

            # Concatenate
            ds_hltc = xarray.merge([ds_lowtide, ds_hightide])

            ds_hltc["count_clear"] = satellite_ds.nbart_red.where(
                satellite_ds.nbart_red > -999
            ).count(dim=["time"])

            custom_dtypes = {
                "count_clear": (np.int16, -999),
                "low_coastal_aerosol": (np.int16, -999),
                "low_blue": (np.int16, -999),
                "low_green": (np.int16, -999),
                "low_red": (np.int16, -999),
                "low_red_edge_1": (np.int16, -999),
                "low_red_edge_2": (np.int16, -999),
                "low_red_edge_3": (np.int16, -999),
                "low_nir_1": (np.int16, -999),
                "low_nir_2": (np.int16, -999),
                "low_swir_2": (np.int16, -999),
                "low_swir_3": (np.int16, -999),
                "low_threshold": (np.float32, np.nan),
                "low_count_clear": (np.int16, -999),
                "high_coastal_aerosol": (np.int16, -999),
                "high_blue": (np.int16, -999),
                "high_green": (np.int16, -999),
                "high_red": (np.int16, -999),
                "high_red_edge_1": (np.int16, -999),
                "high_red_edge_2": (np.int16, -999),
                "high_red_edge_3": (np.int16, -999),
                "high_nir_1": (np.int16, -999),
                "high_nir_2": (np.int16, -999),
                "high_swir_2": (np.int16, -999),
                "high_swir_3": (np.int16, -999),
                "high_threshold": (np.float32, np.nan),
                "high_count_clear": (np.int16, -999),
            }

            ds_prepared = prepare_for_export(
                ds_hltc,
                custom_dtypes=custom_dtypes,
                log=log,
            )  # sets correct dtypes and nodata

            # Export data and metadata
            export_dataset_metadata(
                ds_prepared,
                year=label_date,
                study_area=study_area,
                output_location=output_dir,
                ls_lineage=dss_ls,
                s2_lineage=dss_s2,
                dataset_version=output_version,
                product_family="composites",
                odc_product="ga_s2_tidal_composites_cyear_3",
                thumbnail_bands=["low_red", "low_green", "low_blue"],
                product_maturity=product_maturity,
                dataset_maturity=dataset_maturity,
                run_id=run_id,
                log=log,
            )

            # Close dask client
            client.close()

            log.info(
                f"Study area {study_area}: Completed DEA Intertidal Composites workflow"
            )

        except Exception as e:
            log.exception(f"{run_id}: Failed to run process with error {e}")
            sys.exit(1)
    else:
        log.info(f"Study area {study_area}: Skipping as overwrite ==False")


if __name__ == "__main__":
    tidal_composites_cli()
