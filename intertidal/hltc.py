import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import click

import datacube
import odc.geo.xr
from odc.algo import xr_geomedian
from odc.algo import mask_cleanup
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


def load_data(
    dc,
    geom,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:3577",
    s2_prod="s2_nbart_norm",
    ls_prod="ls_nbart_norm",
    config_path="configs/dea_virtual_product_landsat_s2.yaml",
    filter_gqa=True,
):
    """
    Load cloud-masked Landsat and Sentinel-2 data for a given
    spatial and temporal extent.

    Parameters
    ----------
    dc : Datacube
        A datacube instance connected to a database.
    geom : Geometry object from datacube.utils.geometry
        A geometry object from `datacube.utils.geometry` that defines
        the spatial extent of interest.
    time_range : tuple, optional
        A tuple containing the start and end date for the time range of
        interest, in the format (start_date, end_date). The default is
        ("2019", "2021").
    resolution : int or float, optional
        The spatial resolution (in metres) to load data at. The default
        is 10.
    crs : str, optional
        The coordinate reference system (CRS) to project data into. The
        default is Australian Albers "EPSG:3577".
    s2_prod : str, optional
        The name of the virtual product to use for Sentinel-2 data. The
        default is "s2_nbart_norm".
    ls_prod : str, optional
        The name of the virtual product to use for Landsat data. The
        default is "s2_nbart_norm".
    config_path : str, optional
        The path to the virtual product configuration file. The default is
        "configs/dea_virtual_product_landsat_s2.yaml".
    filter_gqa : bool, optional
        Whether or not to filter Sentinel-2 data using the GQA filter.
        The default is True.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Landsat and Sentinel-2
        data with cloud masking applied.
    """

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry

    # Load in virtual product catalogue
    catalog = catalog_from_file(config_path)

    # Create the 'query' dictionary object
    query_params = {
        "geopolygon": geom,
        "time": time_range,
        "resolution": (-resolution, resolution),
        "output_crs": crs,
        "dask_chunks": {"time": 1, "x": 2048, "y": 2048},
        "resampling": {
            "*": "cubic",
            "oa_nbart_contiguity": "nearest",
            "oa_fmask": "nearest",
            "oa_s2cloudless_mask": "nearest",
        },
    }

    # Optionally add GQA
    # TODO: Remove once Sentinel-2 GQA issue is resolved
    if filter_gqa:
        query_params["gqa_iterative_mean_xy"] = (0, 1)

    # Output list
    data_list = []

    # If Sentinel-2 data is requested
    if s2_prod is not None:
        # Load Sentinel-2 data
        product = catalog[s2_prod]
        s2_ds = product.load(dc, **query_params)

        # Apply cloud mask and contiguity mask
        s2_ds_masked = s2_ds.where(s2_ds.cloud_mask == 1 & s2_ds.contiguity)
        data_list.append(s2_ds_masked)

    # If Landsat data is requested - default for hltc is no ls.
    if ls_prod is not None:
        # Load Landsat data
        product = catalog[ls_prod]
        ls_ds = product.load(dc, **query_params)

        # Clean cloud mask by applying morphological closing to all
        # valid (non cloud, shadow or nodata) pixels. This removes
        # long, narrow features like false positives over bright beaches.
        good_data_cleaned = mask_cleanup(
            mask=ls_ds.cloud_mask.isin([1, 4, 5]),
            mask_filters=[("closing", 5)],
        )

        # Dilate cloud and shadow. To ensure that nodata areas (e.g.
        # Landsat 7 SLC off stripes) are not also dilated, only dilate
        # mask pixels (e.g. values 0 in `good_data_cleaned`) that are
        # outside of the original nodata pixels (e.g. not 0 in
        # `ls_ds.cloud_mask`)
        good_data_mask = mask_cleanup(
            mask=(good_data_cleaned == 0) & (ls_ds.cloud_mask != 0),
            mask_filters=[("dilation", 5)],
        )

        # Apply cloud mask and contiguity mask
        ls_ds_masked = ls_ds.where(~good_data_mask & ls_ds.contiguity)
        data_list.append(ls_ds_masked)

    # Combine into a single ds, sort and drop no longer needed bands
    satellite_ds = (
        xr.concat(data_list, dim="time")
        .sortby("time")
        .drop(["cloud_mask", "contiguity"])
    )
    return satellite_ds


def hltc_geomedians(
    study_area,
    start_date="2020",
    end_date="2022",
    resolution=10,
    crs="EPSG:3577",
    include_s2=True,
    include_ls=False,
    filter_gqa=False,
    config_path="configs/dea_intertidal_config.yaml",
    log=None,
):
    """
    Calculates Geomedians of High tides and Low tides using satellite imagery and
    tidal modeling.

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
    crs : str, optional
        Coordinate reference system, by default "EPSG:3577".
    include_s2 : bool, optional
        Whether to include Sentinel-2 data, by default True.
    include_ls : bool, optional
        Whether to include Landsat data, by default True.
    filter_gqa : bool, optional
        Whether to apply the GQA filter to the dataset, by default False.
    config_path : str, optional
        Path to the configuration file, by default
        "configs/dea_intertidal_config.yaml".
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    ds_min_median : xarray.Dataset
        xarray.Dataset object containing a geomedian of the obs with the lowest 20% tide values for each pixel in the study area.
    ds_max_median : xarray.DataArray
        xarray.Dataset object containing a geomedian of the obs with the highest 20% tide values for each pixel in the study area.
    """

    if log is None:
        log = configure_logging()

    # Create local dask cluster to improve data load time
    client = create_local_dask_cluster(return_client=True)

    # Connect to datacube
    dc = datacube.Datacube(app="Intertidal_elevation")

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

    # Calculate max, min and full range of tide
    tide_max = satellite_ds.tide_m.max(dim="time")
    tide_min = satellite_ds.tide_m.min(dim="time")
    tide_range = tide_max - tide_min

    # Calculate a threshold for low and high tide composite
    min_thresh = tide_min + (tide_range * 0.2)
    max_thresh = tide_max - (tide_range * 0.2)

    # select low tide obs
    ds_min = satellite_ds.where(satellite_ds.tide_m <= min_thresh)

    # select high tide obs
    ds_max = satellite_ds.where(satellite_ds.tide_m >= max_thresh)

    # Drop fully cloudy scenes to speed up geomedian
    ds_min = ds_min.sel(time=ds_min.tide_m.isnull().mean(dim=["x", "y"]) < 1).drop(
        "tide_m"
    )
    ds_max = ds_max.sel(time=ds_max.tide_m.isnull().mean(dim=["x", "y"]) < 1).drop(
        "tide_m"
    )

    # Compute geomedian
    log.info(f"Calculate geomedians for {study_area}")
    ds_20_median = xr_geomedian(ds=ds_min)
    ds_80_median = xr_geomedian(ds=ds_max)

    # Load data and close dask client
    ds_20_median.load()
    ds_80_median.load()

    client.close()

    return ds_20_median, ds_80_median


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
    "--aws_unsigned/--no-aws_unsigned",
    type=bool,
    default=True,
    help="Whether to use sign AWS requests for S3 access",
)
def hltc_cli(
    config_path,
    study_area,
    start_date,
    end_date,
    resolution,
    aws_unsigned,
):
    log = configure_logging(f"Generating HLTCs for study area {study_area}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    try:
        # Calculate hltc geomedians
        ds_20, ds_80 = hltc_geomedians(
            study_area,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
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
        ds_80.to_array().odc.write_cog(prefix + "_hltc_80_s2.tif", overwrite=True)
        ds_20.to_array().odc.write_cog(prefix + "_hltc_20_s2.tif", overwrite=True)
        ds_20.odc.to_rgba(vmin=0.0, vmax=0.3).plot.imshow().figure.savefig(
            prefix + "_hltc_20_s2_rgb.png"
        )
        ds_80.odc.to_rgba(vmin=0.0, vmax=0.3).plot.imshow().figure.savefig(
            prefix + "_hltc_80_s2_rgb.png"
        )
        # Workflow completed
        log.info(f"Study area {study_area}: Completed HLTC workflow")

    except Exception as e:
        log.exception(f"Study area {study_area}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    hltc_cli()
