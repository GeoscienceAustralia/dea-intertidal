import os
import sys

import click
import numpy as np
import odc.geo.xr
import xarray as xr
from dea_tools.dask import create_local_dask_cluster
from eo_tides.eo import pixel_tides
from odc.algo import (
    int_geomedian,
    keep_good_only,
)
from odc.geo.geom import BoundingBox

from intertidal.io import (
    export_dataset_metadata,
    load_data,
    prepare_for_export,
    tidal_metadata,
)
from intertidal.utils import configure_logging


# Function to rename the bands
def rename_bands(ds, old_string, new_string):
    # Create a new dataset with renamed bands
    ds_renamed = ds.rename({band: band.replace(old_string, new_string) for band in ds.data_vars})
    return ds_renamed


def tidal_thresholds(
    tides_highres,
    threshold_lowtide=0.15,
    threshold_hightide=0.85,
    min_obs=0,
):
    # Calculate per-pixel integer rankings for each tide height
    rank_n = tides_highres.rank(dim="time")

    # Calculate pixel-based low and high ranking thresholds from
    # max ranking. Max ranking needs to be rounded up to the nearest
    # integer using "ceil" as xarray will give multiple observation
    # an average rank (e.g. 50.5) value if they are both identical.
    # Additionally: to ensure we capture all matching values, Low
    # threshold needs to be rounded up ("ceil"), and high tide
    # rounded down ("floor").
    rank_max = np.ceil(rank_n.max(dim="time"))
    rank_thresh_low = np.ceil(rank_max * threshold_lowtide)
    rank_thresh_high = np.floor(rank_max * threshold_hightide)

    # Update thresholds to ensure minimum number of valid observations
    if min_obs > 0:
        rank_thresh_low = np.maximum(rank_thresh_low, min_obs)
        rank_thresh_high = np.minimum(rank_thresh_high, rank_max - min_obs)

    # Calculate tide thresholds by masking tides by ranking threshold
    tide_thresh_low = tides_highres.where(rank_n <= rank_thresh_low).max(dim="time")
    tide_thresh_high = tides_highres.where(rank_n >= rank_thresh_high).min(dim="time")

    return tide_thresh_low, tide_thresh_high


def filter_granules(dataset):
    """Return False for any Sentinel-2 dataset with a MGRS
    granule region code in the list of bad region codes.
    """
    drop_list = [
        "50HKG",
        "50HNF",
        "51LWD",
        "51LXE",
        "51LZF",
        "52LBL",
        "52LCL",
        "52LDK",
        "53HNA",
        "53LRC",
        "54GYU",
        "54LWR",
        "54LXR",
        "54LYR",
        "55GBP",
        "55KEA",
        "55KFV",
        "55KGV",
        "55KHT",
        "55KHU",
        "56KKC",
        "56KLC",
        "56KMC",
        "56KMV",
        "56KNU",
        "54LWQ",
        "54LWP",
    ]
    return dataset.metadata.region_code not in drop_list


def tidal_composites(
    satellite_ds,
    threshold_lowtide=0.15,
    threshold_hightide=0.85,
    min_obs=0,
    eps=1e-4,
    cpus=None,
    max_iters=10000,
    tide_model="EOT20",
    tide_model_dir="/var/share/tide_models",
    run_id=None,
    log=None,
):
    """Calculates Geometric Median composites of the coastal zone at low
    and high tide using satellite imagery and tidal modeling.

    This function uses tools from `odc.algo` to keep data in its
    original integer datatype throughout the analysis to minimise
    memory usage. Modelled tide data and nodata pixels are used
    to filter satellite data to low and high tide images prior to
    loading it into memory, allowing more efficient processing.

    Pixel-based implementation of the method originally published in:

    Sagar, S., Phillips, C., Bala, B., Roberts, D., & Lymburner, L.
    (2018). Generating Continental Scale Pixel-Based Surface Reflectance
    Composites in Coastal Regions with the Use of a Multi-Resolution
    Tidal Model. Remote Sensing, 10, 480. https://doi.org/10.3390/rs10030480

    Parameters
    ----------
    satellite_ds : xarray.Dataset
        A satellite data time series containing spectral bands.
    threshold_lowtide : float, optional
        Quantile used to identify low tide observations, by default 0.15.
    threshold_hightide : float, optional
        Quantile used to identify high tide observations, by default 0.85.
    min_obs : int, optional
        Minimum number of clear observations to enforce when calculating tide
        height thresholds. Defaults to 0, which will not apply any minimum.
    eps: float, optional
        Termination criteria passed on to the geomedian algorithm.
    cpus: int, optional
        Requested number of cpus which is passed on to the geomedian function.
    max_iters : int, optional
        Maximum number of iterations done per output pixel in the
        geomedian calculation. This can be set to a low value (e.g. 10)
        to increase the processing speed of test runs.
    tide_model : str, optional
        The tide model or a list of models used to model tides, as
        supported by the `eo-tides` Python package. Options include:
        - "EOT20" (default)
        - "TPXO10-atlas-v2-nc"
        - "FES2022"
        - "FES2022_extrapolated"
        - "FES2014"
        - "FES2014_extrapolated"
        - "GOT5.6"
        - "ensemble" (experimental: combine all above into single ensemble)
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `eo-tides.utils.list_models`.
    run_id : string, optional
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

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id

    # Model tides into for spatial extent and timesteps in satellite data
    log.info(f"{run_id}: Modelling tide heights for each pixel")
    tides_highres = pixel_tides(
        data=satellite_ds,
        model=tide_model,
        resample=True,
        directory=tide_model_dir,
    )
    # Identify nodata pixels in satellite data array by loading only
    # a single band into memory
    log.info(f"{run_id}: Loading red band to identify nodata pixels")
    nodata = satellite_ds.nbart_red.nodata
    nodata_array = (satellite_ds.nbart_red != nodata).compute()

    # Mask tides to make nodata match satellite data array
    tides_highres = tides_highres.where(nodata_array)

    # Calculate low and high tide thresholds from masked tide data
    log.info(f"{run_id}: Calculating low and high tide thresholds with minimum {min_obs} observations")
    low_threshold, high_threshold = tidal_thresholds(
        tides_highres=tides_highres,
        threshold_lowtide=threshold_lowtide,
        threshold_hightide=threshold_hightide,
        min_obs=min_obs,
    )

    # Create masks for selecting satellite observations below and above the
    # low and high tide thresholds
    low_mask = tides_highres <= low_threshold
    high_mask = tides_highres >= high_threshold

    # Keep only scenes with at least 1% valid data to speed up geomedian
    low_keep = low_mask.mean(dim=["x", "y"]) >= 0.01
    high_keep = high_mask.mean(dim=["x", "y"]) >= 0.01
    ds_low = satellite_ds.sel(time=low_keep)
    ds_high = satellite_ds.sel(time=high_keep)

    # Load low and high subsets of data into memory
    log.info(f"{run_id}: Loading {len(ds_low.time)} low tide satellite images into memory")
    ds_low.load()
    log.info(f"{run_id}: Loading {len(ds_high.time)} high tide satellite images into memory")
    ds_high.load()

    # Use `keep_good_only` to set any pixels outside of the tide masks to nodata
    ds_low_masked = keep_good_only(x=ds_low, where=low_mask.sel(time=low_keep))
    ds_high_masked = keep_good_only(x=ds_high, where=high_mask.sel(time=high_keep))

    # Calculate low and high tide geomedians
    num_threads = cpus if cpus is not None else os.cpu_count() - 2
    log.info(f"{run_id}: Running low tide geomedian with {num_threads} threads")
    ds_lowtide = int_geomedian(
        ds=ds_low_masked,
        maxiters=max_iters,
        num_threads=num_threads,
        eps=eps,
    )
    log.info(f"{run_id}: Running high tide geomedian with {num_threads} threads")
    ds_hightide = int_geomedian(
        ds=ds_high_masked,
        maxiters=max_iters,
        num_threads=num_threads,
        eps=eps,
    )

    # Calculate clear count (both low and high tide clear counts
    # are identical, so we can just use one)
    log.info(f"{run_id}: Calculating clear counts")
    ds_lowtide["qa_count_clear"] = (ds_low_masked.nbart_red != nodata).sum(dim="time").astype("int16")

    # Add low and high tide thresholds to the output datasets
    ds_lowtide["qa_low_threshold"] = low_threshold
    ds_hightide["qa_high_threshold"] = high_threshold

    return ds_lowtide, ds_hightide


@click.command()
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a GridSpec tile ID (e.g. in the form 'x123y123') to run the analysis on.",
)
@click.option(
    "--start_date",
    type=str,
    required=True,
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Tidal Composites, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--end_date",
    type=str,
    required=True,
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Tidal Composites, this is set to provide a three year window "
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
    help="The version number to use for output files and metadata (e.g. '0.0.1').",
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
    help="Product maturity metadata to use for the output dataset. Defaults to 'provisional', can also be 'stable'.",
)
@click.option(
    "--dataset_maturity",
    type=str,
    default="final",
    help="Dataset maturity metadata to use for the output dataset. Defaults to 'final', can also be 'interim'.",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="The spatial resolution in metres used to load satellite "
    "data and produce tidal composite outputs. Defaults to 10 metre "
    "Sentinel-2 resolution.",
)
@click.option(
    "--threshold_lowtide",
    type=float,
    default=0.15,
    help="The quantile used to identify low tide observations. Defaults to 0.15.",
)
@click.option(
    "--threshold_hightide",
    type=float,
    default=0.85,
    help="The quantile used to identify high tide observations. Defaults to 0.85.",
)
@click.option(
    "--min_obs",
    type=int,
    default=0,
    help="Minimum number of clear observations to enforce when calculating tide "
    "height thresholds. Defaults to 0, which will not apply any minimum.",
)
@click.option(
    "--mask_sunglint",
    type=int,
    default=20,
    help="Whether to mask out pixels that are likely to be "
    "affected by sunglint using glint angles. Low glint angles "
    "(e.g. < 20) often correspond with sunglint. Defaults to 20, "
    "which will mask all pixels with a glint angle of less than 20.",
)
@click.option(
    "--gqa_filter/--no-gqa_filter",
    type=bool,
    default=True,
    help="Whether to filter scenes when loading data based on gqa values. Defaults to True",
)
@click.option(
    "--include_coastal_aerosol/--no-include_coastal_aerosol",
    type=bool,
    default=True,
    help="Whether to include the coastal aerosol band. Defaults to True",
)
@click.option(
    "--eps",
    type=float,
    default=1e-4,
    help="Termination criteria passed on to the geomedian algorithm.",
)
@click.option(
    "--cpus",
    type=int,
    default=None,
    help="Requested number of CPUs which is passed on to the geomedian function.",
)
@click.option(
    "--max_iters",
    type=int,
    default=1000,
    help="Maximum number of iterations done per output pixel in the "
    "geomedian calculation. This can be set to a low value (e.g. 10) "
    "to increase the processing speed of test runs.",
)
@click.option(
    "--tide_model",
    type=str,
    multiple=True,
    default=["EOT20"],
    help="The model used for tide modelling, as supported by the "
    "`eo-tides` Python package. Options include 'EOT20' (default), "
    "'TPXO10-atlas-v2-nc', 'FES2022', 'FES2014', 'GOT5.6', 'ensemble'.",
)
@click.option(
    "--tide_model_dir",
    type=str,
    default="/var/share/tide_models",
    help="The directory containing tide model data files. Defaults to "
    "'/var/share/tide_models'; for more information about the required "
    "directory structure, refer to `eo-tides.utils.list_models`.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    type=bool,
    default=True,
    help="Whether to use sign AWS requests for S3 access",
)
@click.option(
    "--overwrite/--no-overwrite",
    type=bool,
    default=True,
    help="Whether to overwrite tile data if it already exists.",
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
    min_obs,
    mask_sunglint,
    gqa_filter,
    include_coastal_aerosol,
    eps,
    cpus,
    max_iters,
    tide_model,
    tide_model_dir,
    aws_unsigned,
    overwrite,
):
    # Attempt to import datacube and raise an error if not available
    try:
        import datacube
        from datacube.utils.aws import configure_s3_access
    except ImportError as e:
        msg = (
            "The DEA Tidal Composites CLI is configured for Australian applications, and "
            "requires `datacube`. Please install DEA Intertidal with the "
            "`[datacube]` extra, e.g.: `pip install dea-intertidal[datacube]`"
        )
        raise ImportError(msg) from e

    # Create sample filename to test if data exists on file system
    filename = f"{output_dir}ga_s2_tidal_composites_cyear_3/{output_version.replace('.', '-')}/{study_area[:4]}/{study_area[4:]}/{label_date}--P1Y/ga_s2_tidal_composites_cyear_3_{study_area}_{label_date}--P1Y_final.stac-item.json"

    process_tile = True
    if overwrite:
        process_tile = True
    elif os.path.exists(filename):
        process_tile = False

    # Create a unique run ID based on input params and use for logs
    input_params = locals()
    run_id = f"[{output_version}] [{label_date}] [{study_area}]"
    log = configure_logging(run_id)

    # Record params in logs
    log.info(f"{run_id}: Using parameters {input_params}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    if process_tile:
        try:
            # Create local dask cluster to improve data load time
            client = create_local_dask_cluster(return_client=True)

            # Connect to datacube to load data
            dc = datacube.Datacube(app="Composites_CLI")

            # Use a custom polygon if in testing mode
            if study_area == "testing":
                log.info(f"{run_id}: Running in testing mode using custom study area")
                geom = BoundingBox(467510, -1665790, 468260, -1664840, crs="EPSG:3577").polygon
            else:
                geom = None

            # Load satellite data and dataset IDs for metadata
            # Use `filter_granules` predicate function to drop list of
            # custom Sentinel-2 MGRS granules with poor data coverage
            satellite_ds, dss_s2, _ = load_data(
                dc=dc,
                study_area=study_area,
                geom=geom,
                time_range=(start_date, end_date),
                resolution=resolution,
                crs="EPSG:3577",
                include_s2=True,
                include_ls=False,
                filter_gqa=gqa_filter,
                ndwi=False,
                mask_sunglint=mask_sunglint,
                include_coastal_aerosol=include_coastal_aerosol,
                max_cloudcover=90,
                skip_broken_datasets=True,
                dataset_maturity="final",
                dtype="int16",
                dataset_predicate=filter_granules,
            )
            log.info(f"{run_id}: Found {len(satellite_ds.time)} satellite data timesteps")

            # Fail early if not enough observations
            if len(satellite_ds.time) < 50:
                raise Exception("Insufficient satellite data available to process composites; skipping.")

            # Calculate high and low tide geomedian composites
            log.info(f"{run_id}: Running DEA Tidal Composites workflow")
            ds_lowtide, ds_hightide = tidal_composites(
                satellite_ds=satellite_ds,
                threshold_lowtide=threshold_lowtide,
                threshold_hightide=threshold_hightide,
                min_obs=min_obs,
                eps=eps,
                cpus=cpus,
                max_iters=max_iters,
                tide_model=tide_model,
                tide_model_dir=tide_model_dir,
                run_id=run_id,
                log=log,
            )

            # Rename low and high tide bands to add "low"/"high" prefix in place of "nbart"
            ds_hightide = rename_bands(ds_hightide, "nbart", "high")
            ds_lowtide = rename_bands(ds_lowtide, "nbart", "low")

            # Concatenate into a single output dataset
            ds_tidalcomposites = xr.merge([ds_lowtide, ds_hightide])

            # Ensure spatial information is still attached
            ds_tidalcomposites = odc.geo.xr.assign_crs(ds_tidalcomposites, satellite_ds.odc.crs)

            custom_dtypes = {
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
                "qa_low_threshold": (np.float32, np.nan),
                "qa_high_threshold": (np.float32, np.nan),
                "qa_count_clear": (np.int16, -999),
            }

            # Sets correct dtypes and nodata
            ds_prepared = prepare_for_export(
                ds_tidalcomposites,
                custom_dtypes=custom_dtypes,
                log=log,
            )

            # Calculate additional tile-level tidal metadata and graph.
            metadata_dict, tide_graph_fig = tidal_metadata(
                product_family="tidal_composites",
                threshold_lowtide=threshold_lowtide,
                threshold_hightide=threshold_hightide,
                data=satellite_ds,
                modelled_freq="30min",
                model=tide_model,
                directory=tide_model_dir,
            )

            # Export data and metadata
            export_dataset_metadata(
                ds_prepared,
                year=label_date,
                study_area=study_area,
                output_location=output_dir,
                s2_lineage=dss_s2,
                dataset_version=output_version,
                product_family="tidal_composites",
                odc_product="ga_s2_tidal_composites_cyear_3",
                thumbnail_bands=["low_red", "low_green", "low_blue"],
                tide_graph_fig=tide_graph_fig,
                additional_metadata=metadata_dict,
                product_maturity=product_maturity,
                dataset_maturity=dataset_maturity,
                run_id=run_id,
                log=log,
            )

            # Close dask client
            client.close()
            log.info(f"{run_id}: Completed DEA Tidal Composites workflow")

        except Exception as e:
            log.exception(f"{run_id}: Failed to run process with error {e}")
            sys.exit(1)
    else:
        log.info(f"{run_id}: Skipping as overwrite==False")


if __name__ == "__main__":
    tidal_composites_cli()
