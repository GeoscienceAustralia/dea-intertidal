import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import click

import datacube
import odc.geo.xr
from odc.algo import xr_geomedian, xr_quantile
from datacube.utils.geometry import Geometry
from datacube.utils.aws import configure_s3_access

from dea_tools.coastal import pixel_tides
from dea_tools.dask import create_local_dask_cluster

from intertidal.utils import (
    load_config,
    configure_logging,
    round_date_strings,
    export_intertidal_rasters,
)
from intertidal.elevation import load_data


def intertidal_composites(
    study_area,
    start_date="2020",
    end_date="2022",
    resolution=10,
    threshold_method="percent",
    threshold_lowtide=0.2,
    threshold_hightide=0.8,
    crs="EPSG:3577",
    include_s2=True,
    include_ls=False,
    filter_gqa=False,
    config_path="configs/dea_intertidal_config.yaml",
    log=None,
):
    """
    Calculates Geometric Median composites of the coastal zone at low 
    and high tide using satellite imagery and tidal modeling.
    
    Based on the method described in:
    
    Sagar, S., Phillips, C., Bala, B., Roberts, D., & Lymburner, L. 
    (2018). Generating Continental Scale Pixel-Based Surface Reflectance
    Composites in Coastal Regions with the Use of a Multi-Resolution
    Tidal Model. Remote Sensing, 10, 480. 
    https://doi.org/10.3390/rs10030480

    Parameters
    ----------
    study_area : int or str or Geometry
        Study area polygon represented as either the ID of a tile grid
        cell, or a Geometry object.
    start_date : str, optional
        Start date of data to load (inclusive), by default '2020'. Can
        be any string supported by datacube (e.g. '2020-01-01')
    end_date : str, optional
        End date of data to load (inclusive), by default '2022'. Can
        be any string supported by datacube (e.g. '2022-12-31')
    resolution : int, optional
        Pixel size in meters, by default 10.
    threshold_method : str, optional
        The method used to identify the lower and upper tide height
        thresholds used to create low and high tide composites. Options
        are "percent" which will use an absolute percentage of the total
        tide range, or "percentile" which will take a percentile of all
        tide observations.
    threshold_lowtide : float, optional
        The percent or percentile used to identify low tide
        observations, by default 0.2.
    threshold_hightide : float, optional
        The percent or percentile used to identify high tide
        observations, by default 0.8.
    crs : str, optional
        Coordinate reference system used to load data, by default
        "EPSG:3577".
    include_s2 : bool, optional
        Whether to include Sentinel-2 data, by default True.
    include_ls : bool, optional
        Whether to include Landsat data, by default False.
    filter_gqa : bool, optional
        Whether to apply the GQA filter to the dataset, by default False.
    config_path : str, optional
        Path to the configuration file, by default
        "configs/dea_intertidal_config.yaml".
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    ds_lowtide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the lowest X percent or percentile tide heights for each
        pixel in the study area.
    ds_hightide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the highest X percent or percentile tide values for each
        pixel in the study area.
    """

    if log is None:
        log = configure_logging()

    # Create local dask cluster to improve data load time
    client = create_local_dask_cluster(return_client=True)

    # Connect to datacube
    dc = datacube.Datacube(app="Intertidal_composites")

    # Load analysis params from config file
    config = load_config(config_path)

    # Load study area from tile grid if passed a string
    if isinstance(study_area, (int, str)):
        # Load study area
        gridcell_gdf = (
            gpd.read_file(config["Input files"]["grid_path"])
            .to_crs(epsg=4326)
            .set_index("id")
        )
        gridcell_gdf.index = gridcell_gdf.index.astype(str)
        gridcell_gdf = gridcell_gdf.loc[[str(study_area)]]

        # Create geom as input for dc.load
        geom = Geometry(geom=gridcell_gdf.iloc[0].geometry, crs="EPSG:4326")
        log.info(f"Study area {study_area}: Loaded study area grid")

    # Otherwise, use supplied geom
    else:
        geom = study_area
        study_area = "testing"
        log.info(f"Study area {study_area}: Loaded custom study area")

    # Load data
    log.info(f"Study area {study_area}: Loading satellite data")
    satellite_ds = load_data(
        dc=dc,
        geom=geom,
        time_range=(start_date, end_date),
        resolution=resolution,
        crs=crs,
        s2_prod="s2_nbart_norm" if include_s2 else None,
        ls_prod="ls_nbart_norm" if include_ls else None,
        config_path=config["Virtual product"]["virtual_product_path"],
        filter_gqa=filter_gqa,
    )
    satellite_ds.persist()

    # Model tides into every pixel in the three-dimensional (x by y by time) satellite dataset
    log.info(f"Study area {study_area}: Modelling tide heights for each pixel")
    tide_m, _ = pixel_tides(satellite_ds, resample=True)

    # Add tide heights to satellite data array
    log.info(f"Study area {study_area}: Add tide heights to satellite data array")
    satellite_ds["tide_m"] = tide_m

    # Calculate a threshold for low and high tide composite using either
    # an absolute percent of the total tide range, or a percentile of
    # of all tide observations.
    log.info(f"Study area {study_area}: Calculate low and high tide thresholds")
    if threshold_method == "percent":
        # Calculate max, min and full range of tide
        tide_max = satellite_ds.tide_m.max(dim="time")
        tide_min = satellite_ds.tide_m.min(dim="time")
        tide_range = tide_max - tide_min

        # Use tide range to calculate thresholds
        min_thresh = tide_min + (tide_range * threshold_lowtide)
        max_thresh = tide_min + (tide_range * threshold_hightide)

    elif threshold_method == "percentile":
        # Calculate min and max thresholds using percentiles of tides
        tide_q = xr_quantile(
            src=satellite_ds[["tide_m"]],
            quantiles=[threshold_lowtide, threshold_hightide],
            nodata=np.nan,
        )
        min_thresh = tide_q.sel(quantile=threshold_lowtide).tide_m
        max_thresh = tide_q.sel(quantile=threshold_hightide).tide_m

    # Select low and high tide obs
    log.info(f"Study area {study_area}: Masking to low and high tide observations")
    ds_low = satellite_ds.where(satellite_ds.tide_m <= min_thresh)
    ds_high = satellite_ds.where(satellite_ds.tide_m >= max_thresh)

    # Drop fully empty scenes to speed up geomedian
    ds_low = ds_low.sel(time=ds_low.tide_m.isnull().mean(dim=["x", "y"]) < 1).drop(
        "tide_m"
    )
    ds_high = ds_high.sel(time=ds_high.tide_m.isnull().mean(dim=["x", "y"]) < 1).drop(
        "tide_m"
    )

    # Compute geomedian
    log.info(f"Study area {study_area}: Calculate geomedians")
    ds_lowtide = xr_geomedian(ds=ds_low)
    ds_hightide = xr_geomedian(ds=ds_high)

    # Load data and close dask client
    ds_lowtide.load()
    ds_hightide.load()

    client.close()

    return ds_lowtide, ds_hightide


@click.command()
@click.option(
    "--config_path",
    type=str,
    required=True,
    help="Path to the YAML config file defining inputs to "
    "use for this analysis. These are typically located in "
    "the `dea-intertidal/configs/` directory.",
)
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a unique ID of an analysis "
    "gridcell that will be used to run the analysis. This "
    'should match a row in the "id" column of the provided '
    "analysis gridcell vector file.",
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
    help="The percent or percentile used to identify low tide "
    "observations. Defaults to 0.2.",
)
@click.option(
    "--threshold_hightide",
    type=float,
    default=0.8,
    help="The percent or percentile used to identify high tide "
    "observations. Defaults to 0.8.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    type=bool,
    default=True,
    help="Whether to use sign AWS requests for S3 access",
)
def intertidal_composites_cli(
    config_path,
    study_area,
    start_date,
    end_date,
    resolution,
    threshold_lowtide,
    threshold_hightide,
    aws_unsigned,
):
    log = configure_logging(
        f"Study area {study_area}: Generating Intertidal composites"
    )

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    try:
        # Calculate high and low tide geomedian composites
        ds_lowtide, ds_hightide = intertidal_composites(
            study_area,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            threshold_method="percent",
            threshold_lowtide=threshold_lowtide,
            threshold_hightide=threshold_hightide,
            crs="EPSG:3577",
            include_s2=True,
            include_ls=False,
            filter_gqa=False,
            config_path=config_path,
            log=log,
        )

        # Export layers as GeoTIFFs
        log.info(f"Study area {study_area}: Exporting outputs to GeoTIFFs")

        prefix = f"data/interim/{study_area}_{start_date}_{end_date}"
        ds_lowtide.to_array().odc.write_cog(
            f"{prefix}_composite_{int(threshold_lowtide * 100)}_s2.tif", overwrite=True
        )
        ds_hightide.to_array().odc.write_cog(
            f"{prefix}_composite_{int(threshold_hightide * 100)}_s2.tif", overwrite=True
        )

        # Export as images
        prefix = f"data/figures/{study_area}_{start_date}_{end_date}"
        ds_lowtide.odc.to_rgba(vmin=0.0, vmax=0.3).plot.imshow().figure.savefig(
            f"{prefix}_composite_{int(threshold_lowtide * 100)}_s2_rgb.png"
        )
        ds_hightide.odc.to_rgba(vmin=0.0, vmax=0.3).plot.imshow().figure.savefig(
            f"{prefix}_composite_{int(threshold_hightide * 100)}_s2_rgb.png"
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
