import json
import shutil
import warnings
import tempfile
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path
from urllib.parse import urlparse

import datacube
import odc.geo.xr
from odc.geo.geobox import GeoBox
from odc.geo.gridspec import GridSpec
from odc.geo.types import xy_
from odc.algo import (
    mask_cleanup,
    enum_to_bool,
    keep_good_only,
    erase_bad,
    to_f32,
)
from dea_tools.coastal import glint_angle
from eodatasets3 import DatasetAssembler, serialise
from eodatasets3.scripts.tostac import json_fallback
from eodatasets3.verify import PackageChecksum
from eodatasets3.stac import to_stac_item, validate_item
from datacube.utils.masking import mask_invalid_data

from intertidal.utils import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _id_to_tuple(id_str):
    """
    Converts a tile ID in form 'x123y123' to a ix, iy tuple so it
    can be passed to a GridSpec (e.g. `gs[ix, iy]`)
    """
    try:
        ix, iy = id_str.replace("x", "").split("y")
        return int(ix), int(iy)
    except ValueError:
        raise ValueError(
            "Supplied study area ID is not in the form 'x123y123'. If "
            "you meant to provide an ID matching a feature from a "
            "custom vector file, make sure you run the 'Optional: "
            "load study area from vector file' notebook cell."
        )


def extract_geobox(
    study_area=None,
    geom=None,
    resolution=10,
    crs="EPSG:3577",
    tile_width=32000,
    gridspec_origin_x=-4416000,
    gridspec_origin_y=-6912000,
):
    """
    Handles extraction of a GeoBox pixel grid from either a GridSpec
    tile ID (in the form 'x143y56'), or a provided Geometry object.

    If a tile ID string is passed to `study_area`, a GeoBox will be
    extracted based on relevant GridSpec tile. If a custom Geometry
    object is passed using `geom`, it will be converted to a GeoBox.

    (Either `study_area` or `geom` is required; `geom` will override
    `study_area` if provided).

    Parameters
    ----------
    study_area : str, optional
        Tile ID string to process. This should be the ID of a GridSpec
        analysis tile in the format "x123y123". If `geom` is provided,
        this will have no effect.
    geom : Geometry, optional
        A datacube Geometry object defining a custom spatial extent of
        interest. If `geom` is provided, this will overrule any study
        area ID passed to `study_area` and will be returned as-is.
    resolution : int, optional
        The desired resolution of the GeoBox grid, in units of the
        coordinate reference system (CRS). Defaults to 10.
    crs : str, optional
        The coordinate reference system (CRS) to use for the GeoBox.
        Defaults to "EPSG:3577".
    tile_width : int, optional
        The width of a GridSpec tile, in units of the coordinate
        reference system (CRS). Defaults to 32000 metres.
    gridspec_origin_x : int, optional
        The x-coordinate of the origin (bottom-left corner) of the
        GridSpec tile grid. Defaults to -4416000.
    gridspec_origin_y : int, optional
        The y-coordinate of the origin (bottom-left corner) of the
        GridSpec tile grid. Defaults to -6912000.

    Returns
    -------
    geobox : odc.geo.geobox.GeoBox
        A GeoBox defining the pixel grid to use to load data (defining
        the CRS, resolution, shape and extent of the study area).
    """

    # List of valid input geometry types (from `odc-geo` or `datacube-core`)
    GEOM_TYPES = (odc.geo.geom.Geometry, datacube.utils.geometry._base.Geometry)

    # Either `study_area` or `geom` must be provided
    if study_area is None and geom is None:
        raise ValueError(
            "Please provide either a study area ID (using `study_area`), "
            "or a datacube Geometry object (using `geom`)."
        )

    # If custom geom is provided, verify it is a geometry
    elif geom is not None and not isinstance(geom, GEOM_TYPES):
        raise ValueError(
            "Unsupported input type for `geom`; please provide a "
            "datacube Geometry object."
        )

    # Otherwise, extract GeoBox from geometry
    elif geom is not None and isinstance(geom, GEOM_TYPES):
        geobox = GeoBox.from_geopolygon(geom, crs=crs, resolution=resolution)

    # If no custom geom provided, load tile from GridSpec tile grid
    elif geom is None:
        # Verify that resolution fits evenly inside tile width
        if tile_width % resolution != 0:
            raise ValueError(
                "Ensure that `resolution` divides into `tile_width` evenly."
            )

        # Calculate tile pixels
        n_pixels = tile_width / resolution

        # Create GridSpec tile grid
        gs = GridSpec(
            crs=crs,
            resolution=resolution,
            tile_shape=(n_pixels, n_pixels),
            origin=xy_(gridspec_origin_x, gridspec_origin_y),
        )

        # Extract GeoBox from GridSpec
        geobox = gs[_id_to_tuple(study_area)]

    return geobox


def load_data(
    dc,
    study_area=None,
    geom=None,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:3577",
    include_s2=True,
    include_ls=True,
    filter_gqa=True,
    max_cloudcover=90,
    skip_broken_datasets=True,
    ndwi=True,
    mask_sunglint=None,
    dask_chunks=None,
    dtype="float32",
    **query,
):
    """
    Loads cloud-masked Sentinel-2 and Landsat satellite data for a given
    study area/geom and time range.

    Supports optionally converting to Normalised Difference Water Index
    and masking sunglinted pixels.

    Parameters
    ----------
    dc : datacube.Datacube()
        A datacube instance to load data from.
    study_area : str, optional
        Tile ID string to process. This should be the ID of a GridSpec
        analysis tile in the format "x123y123". If `geom` is provided,
        this will have no effect.
    geom : Geometry, optional
        A datacube Geometry object defining a custom spatial extent of
        interest. If `geom` is provided, this will overrule any study
        area ID passed to `study_area` and will be returned as-is.
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
    include_s2 : bool, optional
        Whether to load Sentinel-2 data.
    include_ls : bool, optional
        Whether to load Landsat data.
    filter_gqa : bool, optional
        Whether or not to filter Sentinel-2 data using the GQA filter.
        Defaults to True.
    max_cloudcover : float, optional
        The maximum cloud cover metadata value used to load data.
        Defaults to 90 (i.e. 90% cloud cover).
    skip_broken_datasets : bool, optional
        Whether to skip broken datasets during load. This can avoid
        temporary file access issues on S3, however introduces 
        randomness into the analysis (two identical runs may produce 
        different results due to different data failing to load).
    ndwi : bool, optional
        Whether to convert spectral bands to Normalised Difference Water
        Index values before returning them. Note that this must be set
        to True if both `include_s2` and `include_ls` are True.
    mask_sunglint : int, optional
        EXPERIMENTAL: Whether to mask out pixels that are likely to be
        affected by sunglint using glint angles. Low glint angles
        (e.g. < 20) often correspond with sunglint. Defaults to None;
        set to e.g. "20" to mask out all pixels with a glint angle of
        less than 20.
    dask_chunks : dict, optional
        Optional custom Dask chunks to load data with. Defaults to None,
        which will use '{"x": 1600, "y": 1600}'.
    dtype : str, optional
        Desired data type for output data. Valid values are "int16"
        (default) and "float32". If `ndwi=True`, then "float32" will be
        used regardless of what is set here (as nodata values must be
        set to 'NaN' before calculating NDWI).
    **query :
        Optional datacube.load keyword argument parameters used to
        query data.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Landsat or Sentinel-2
        data.
    dss_s2, dss_ls : lists or None
        Lists of ODC datasets loaded to produce `satellite_ds` (used
        to generate ODC lineage metadata for DEA Intertidal)
    """

    # Set spectral bands to load
    s2_spectral_bands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_red_edge_1",
        "nbart_red_edge_2",
        "nbart_red_edge_3",
        "nbart_nir_1",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
    ]
    ls_spectral_bands = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir",
        "nbart_swir_1",
        "nbart_swir_2",
    ]

    # Set masking bands to load
    s2_masking_bands = ["oa_s2cloudless_mask", "oa_nbart_contiguity"]
    ls_masking_bands = ["oa_fmask", "oa_nbart_contiguity"]

    # Set sunglint bands to load
    if mask_sunglint is not None:
        sunglint_bands = [
            "oa_solar_zenith",
            "oa_solar_azimuth",
            "oa_satellite_azimuth",
            "oa_satellite_view",
        ]
    else:
        sunglint_bands = []

    # Load study area, defined as a GeoBox pixel grid
    geobox = extract_geobox(
        study_area=study_area, geom=geom, resolution=resolution, crs=crs
    )

    # Set up query params
    query_params = {
        "like": geobox.compat,  # Load into the exact GeoBox pixel grid
        "time": time_range,
        **query,  # Optional additional query parameters
    }

    # Set up load params
    load_params = {
        "like": geobox.compat,
        "group_by": "solar_day",
        "dask_chunks": {"x": 1600, "y": 1600} if dask_chunks is None else dask_chunks,
        "resampling": {
            "*": "cubic",
            "oa_fmask": "nearest",
            "oa_s2cloudless_mask": "nearest",
        },
        "skip_broken_datasets": skip_broken_datasets,
    }

    # Optionally add GQA
    # TODO: Remove once Sentinel-2 GQA issue is resolved
    if filter_gqa:
        query_params["gqa_iterative_mean_xy"] = (0, 1)

    # Output data
    data_list = []

    # If Sentinel-2 data is requested
    if include_s2:
        # Find datasets to load
        dss_s2 = dc.find_datasets(
            product=["ga_s2am_ard_3", "ga_s2bm_ard_3"],
            s2cloudless_cloud=(0, max_cloudcover),
            **query_params,
        )

        # Load datasets
        ds_s2 = dc.load(
            datasets=dss_s2,
            measurements=s2_spectral_bands + s2_masking_bands + sunglint_bands,
            **load_params,
        )

        # Create cloud mask, treating nodata and clouds as bad pixels
        cloud_mask = enum_to_bool(
            mask=ds_s2.oa_s2cloudless_mask, categories=["nodata", "cloud"]
        )

        # Identify non-contiguous pixels
        noncontiguous_mask = enum_to_bool(ds_s2.oa_nbart_contiguity, categories=[False])

        # Set cloud mask and non-contiguous pixels to nodata
        combined_mask = cloud_mask | noncontiguous_mask
        ds_s2 = erase_bad(
            x=ds_s2[s2_spectral_bands + sunglint_bands], where=combined_mask
        )

        # Optionally, apply sunglint mask
        if mask_sunglint is not None:
            # Calculate glint angle
            glint_array = glint_angle(
                solar_azimuth=ds_s2.oa_solar_azimuth,
                solar_zenith=ds_s2.oa_solar_zenith,
                view_azimuth=ds_s2.oa_satellite_azimuth,
                view_zenith=ds_s2.oa_satellite_view,
            )

            # Apply glint angle threshold and set affected pixels to nodata
            glint_mask = glint_array > mask_sunglint
            ds_s2 = keep_good_only(x=ds_s2[s2_spectral_bands], where=glint_mask)

        # Optionally convert to float, setting all nodata pixels to `np.nan`
        # (required for NDWI, so will be applied even if `dtype="int16"`)
        if (dtype == "float32") or ndwi:
            ds_s2 = to_f32(ds_s2)

        # Convert to NDWI
        if ndwi:
            # Calculate NDWI
            ds_s2["ndwi"] = (ds_s2.nbart_green - ds_s2.nbart_nir_1) / (
                ds_s2.nbart_green + ds_s2.nbart_nir_1
            )
            data_list.append(ds_s2[["ndwi"]])
        else:
            data_list.append(ds_s2)

    # If Landsat data is requested
    if include_ls:
        # Find datasets to load
        dss_ls = dc.find_datasets(
            product=[
                "ga_ls5t_ard_3",
                "ga_ls7e_ard_3",
                "ga_ls8c_ard_3",
                "ga_ls9c_ard_3",
            ],
            cloud_cover=(0, max_cloudcover),
            **query_params,
        )

        # Load datasets
        ds_ls = dc.load(
            datasets=dss_ls,
            measurements=ls_spectral_bands + ls_masking_bands + sunglint_bands,
            **load_params,
        )

        # First, we identify all bad pixels: nodata, cloud and shadow.
        # We then apply morphological opening to clean up narrow false
        # positive clouds (e.g. bright sandy beaches). By including
        # nodata, we make sure that small areas of cloud next to Landsat
        # 7 SLC-off nodata gaps are not accidently removed (at the cost
        # of not being able to clean false positives next to SLC-off gaps)
        bad_data = enum_to_bool(
            ds_ls.oa_fmask, categories=["nodata", "cloud", "shadow"]
        )
        bad_data_cleaned = mask_cleanup(bad_data, mask_filters=[("opening", 5)])

        # We now dilate ONLY pixels in our cleaned bad data dask that
        # are outside of our iriginal nodata pixels. This ensures that
        # Landsat 7 SLC-off nodata stripes are not also dilated.
        nodata_mask = enum_to_bool(ds_ls.oa_fmask, categories=["nodata"])
        bad_data_mask = mask_cleanup(
            mask=bad_data_cleaned & ~nodata_mask,
            mask_filters=[("dilation", 5)],
        )

        # Identify non-contiguous pixels
        noncontiguous_mask = enum_to_bool(ds_ls.oa_nbart_contiguity, categories=[False])

        # Set cleaned bad pixels and non-contiguous pixels to nodata
        combined_mask = bad_data_mask | noncontiguous_mask
        ds_ls = erase_bad(ds_ls[ls_spectral_bands + sunglint_bands], combined_mask)

        # Optionally, apply sunglint mask
        if mask_sunglint is not None:
            # Calculate glint angle
            glint_array = glint_angle(
                solar_azimuth=ds_ls.oa_solar_azimuth,
                solar_zenith=ds_ls.oa_solar_zenith,
                view_azimuth=ds_ls.oa_satellite_azimuth,
                view_zenith=ds_ls.oa_satellite_view,
            )

            # Apply glint angle threshold and set affected pixels to nodata
            glint_mask = glint_array > mask_sunglint
            ds_ls = keep_good_only(x=ds_ls[ls_spectral_bands], where=glint_mask)

        # Optionally convert to float, setting all nodata pixels to `np.nan`
        # (required for NDWI, so will be applied even if `dtype="int16"`)
        if (dtype == "float32") or ndwi:
            ds_ls = to_f32(ds_ls)

        # Convert to NDWI
        if ndwi:
            # Calculate NDWI
            ds_ls["ndwi"] = (ds_ls.nbart_green - ds_ls.nbart_nir) / (
                ds_ls.nbart_green + ds_ls.nbart_nir
            )
            data_list.append(ds_ls[["ndwi"]])
        else:
            data_list.append(ds_ls)

    # Combine into a single ds, sort and drop no longer needed bands
    satellite_ds = xr.concat(data_list, dim="time").sortby("time")

    # Return satellite data dataset and dataset lineage information
    dss_ls = None if include_ls is False else dss_ls
    dss_s2 = None if include_s2 is False else dss_s2
    return satellite_ds, dss_s2, dss_ls


def load_topobathy(
    dc,
    satellite_ds,
    product="ga_ausbathytopo250m_2023",
    resampling="bilinear",
    mask_invalid=True,
):
    """
    Loads a topo-bathymetric DEM for the extents of the loaded satellite
    data. This is used as a coarse mask to constrain the analysis to the
    coastal zone, improving run time and reducing clear false positives.

    Parameters
    ----------
    dc : Datacube
        A Datacube instance for loading data.
    satellite_ds : ndarray
        The loaded satellite data, used to obtain the spatial extents
        of the data.
    product : str, optional
        The name of the topo-bathymetric DEM product to load from the
        datacube. Defaults to "ga_ausbathytopo250m_2023".
    resampling : str, optional
        The resampling method to use, by default "bilinear".
    mask_invalid : bool, optional
        Whether to mask invalid/nodata values in the array by setting
        them to NaN, by default True.

    Returns
    -------
    topobathy_ds : xarray.Dataset
        The loaded topo-bathymetric DEM.
    """
    topobathy_ds = dc.load(
        product=product, like=satellite_ds.odc.geobox.compat, resampling=resampling
    ).squeeze("time")

    # Mask invalid data
    if mask_invalid:
        topobathy_ds = mask_invalid_data(topobathy_ds)

    return topobathy_ds


def load_aclum(
    dc,
    satellite_ds,
    product="abares_clum_2020",
    resampling="nearest",
    mask_invalid=True,
):
    """
    Loads an ABARES derived land use classification of Australia
    for the extents of the loaded satellite data. The 'intensive urban'
    land use class is used as a coarse mask to clean up intertidal
    extents classifications in urban areas.

    Parameters
    ----------
    dc : Datacube
        A Datacube instance for loading data.
    satellite_ds : ndarray
        The loaded satellite data, used to obtain the spatial extents
        of the data.
    product : str, optional
        The name of the ABARES land use dataset product to load from the
        datacube. Defaults to "abares_clum_2020".
    resampling : str, optional
        The resampling method to use, by default "nearest".
    mask_invalid : bool, optional
        Whether to mask invalid/nodata values in the array by setting
        them to NaN, by default True.

    Returns
    -------
    reclassified_aclum : xarray.Dataset
        The ABARES land use mask, summarised to include only two land
        use classes: 'intensive urban' and 'other'.
    """
    aclum_ds = dc.load(
        product=product, like=satellite_ds.odc.geobox.compat, resampling=resampling
    ).squeeze("time")

    # Mask invalid data
    if mask_invalid:
        aclum_ds = mask_invalid_data(aclum_ds)

    # Manually isolate the 'intensive urban' land use summary class, set
    # all other pixels to false. For class definitions, refer to
    # gdata1/data/land_use/ABARES_CLUM/geotiff_clum_50m1220m/Land use, 18-class summary.qml)
    reclassified_aclum = aclum_ds.alum_class.isin(
        [
            500,
            530,
            531,
            532,
            533,
            534,
            535,
            536,
            537,
            538,
            540,
            541,
            550,
            551,
            552,
            553,
            554,
            555,
            560,
            561,
            562,
            563,
            564,
            565,
            566,
            567,
            570,
            571,
            572,
            573,
            574,
            575,
        ]
    )
    return reclassified_aclum


def _is_s3(path):
    """
    Determine whether output location is on S3.
    """
    uu = urlparse(path)
    return uu.scheme == "s3"


def _write_thumbnail(da, path, max_resolution=320):
    """
    Generate and save a thumbnail image from a DEA Intertidal Elevation
    `xarray.DataArray`.

    The thumbnail is reprojected to the specified maximum resolution,
    colorized using the 'viridis' colormap, and compressed as a JPEG
    with the specified quality.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray containg DEA Intertidal Elevation data.
    path : str
        The path where the thumbnail image will be saved.
    max_resolution : int, optional
        The maximum resolution of the thumbnail image, by default 320.
    """
    jpeg_data = (
        da.odc.reproject(
            how=da.odc.geobox.zoom_to(max_resolution),
            resampling="min",
        )
        .pipe(lambda x: x.where(np.isfinite(x)))
        .odc.colorize(vmin=-2.5, vmax=1.5, cmap="viridis")
        .odc.compress("jpeg", 85, transparent=[255, 255, 255])
    )

    with open(path, "wb") as f:
        f.write(jpeg_data)


def _write_stac(
    dataset_assembler,
    destination_path,
    explorer_base_url="https://explorer.dea.ga.gov.au",
    validate=False,
):
    """
    Generate a STAC (SpatioTemporal Asset Catalog) metadata JSON file
    alongside the input ODC YAML metadata file.

    Also updates the dataset assembler object with the generated STAC
    JSON file and adds the file to its checksum.

    Parameters
    ----------
    dataset_assembler : DatasetAssembler
        An eodatasets DatasetAssembler object containing metadata for
        the dataset.
    destination_path : str
        The destination directory path where the STAC metadata JSON file
        will eventually be written or uploaded to.
    explorer_base_url : str, optional
        Base URL of the explorer, by default "https://explorer.dea.ga.gov.au".
    validate : bool, optional
        Flag indicating whether to validate the generated STAC JSON
        against official STAC metadata specifications, by default True.

    Returns
    -------
    dict
        The generated STAC metadata as a dictionary dictionary.
    """

    # Get path of input metadata file from assembler object
    input_metadata_path = (
        dataset_assembler.names.dataset_path / dataset_assembler.names.metadata_file
    )

    # Get final destination paths of output/published metadata files
    # to use in STAC metadata
    odc_dataset_metadata_url = (
        f"{destination_path}{dataset_assembler.names.metadata_file}"
    )
    stac_item_destination_url = odc_dataset_metadata_url.replace(
        "odc-metadata.yaml", "stac-item.json"
    )

    # Generate STAC
    stac = to_stac_item(
        dataset=serialise.from_path(input_metadata_path),
        stac_item_destination_url=stac_item_destination_url,
        dataset_location=destination_path,
        odc_dataset_metadata_url=odc_dataset_metadata_url,
        explorer_base_url=explorer_base_url,
    )

    # Optionally validate
    if validate:
        print("Validating STAC")
        validate_item(stac)

    # Write out STAC JSON alongside input metadata file
    output_stac_path = Path(
        str(input_metadata_path).replace("odc-metadata.yaml", "stac-item.json")
    )
    with output_stac_path.open("w") as f:
        json.dump(stac, f, default=json_fallback)

    # Add STAC as accessory (don't think this currently has any effect
    # as we have already written out our metadata to file)
    dataset_assembler.add_accessory_file("metadata:stac", output_stac_path)

    # Update checksum to include new STAC JSON file
    checksummer = PackageChecksum()
    checksum_file = (
        dataset_assembler.names.dataset_path
        / dataset_assembler._accessories["checksum:sha1"].name
    )
    checksummer.read(checksum_file)
    checksummer.add_file(output_stac_path)
    checksummer.write(checksum_file)

    return stac


def _ls_platform_instrument(year):
    """
    Indentify relevant Landsat platforms and instruments for a given
    year of DEA Intertidal analysis. Only applicable from 2015 onward.
    """
    # Platforms and intruments
    year = int(year)
    if year <= 2020:
        platform = "landsat-7,landsat-8,sentinel-2a,sentinel-2b"
        instrument = "ETM_OLI_TIRS_MSI"
    elif year in (2021, 2022):
        platform = "landsat-7,landsat-8,landsat-9,sentinel-2a,sentinel-2b"
        instrument = "ETM_OLI_TIRS_MSI"
    else:
        platform = "landsat-8,landsat-9,sentinel-2a,sentinel-2b"
        instrument = "OLI_TIRS_MSI"

    return platform, instrument


def prepare_for_export(
    ds,
    int_bands=None,
    int_nodata=-999,
    int_dtype=np.int16,
    float_dtype=np.float32,
    output_location=None,
    overwrite=True,
    log=None,
):
    """
    Prepares DEA Intertidal data for export by correctly setting nodata
    values and datatypes. Optionally supports exporting data to GeoTIFFs
    on file.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the bands to be exported.
    int_bands : tuple or list, optional
        A list of bands to export as integer datatype. If None, will use
        the following list of bands: ("exposure", "extents",
        "offset_hightide", "offset_lowtide", "spread")
    int_nodata : int, optional
        An integer that represents nodata values for integer bands
        (default is -999).
    int_dtype : string or numpy data type, optional
        The data type to use for integer layers (default is
        np.int16).
    float_dtype : string or numpy data type, optional
        The data type to use for floating point layers (default is
        np.float32).
    output_location : str
        An optional location to output the data; defaults to None which
        will not output data.
    overwrite : bool, optional
        A boolean value that determines whether or not to overwrite
        existing files (default is True).

    Returns
    -------
    ds : xarray.Dataset
        The input dataset with correctly set nodata attributes and dtypes.
    """

    def _prepare_band(
        band, int_bands, int_nodata, int_dtype, float_dtype, output_location, overwrite
    ):
        # Export specific bands as integer16 data types by first filling
        # NaN with nodata value before converting to int, then setting
        # nodata attribute on layer
        if band.name in int_bands:
            band = band.fillna(int_nodata).astype(int_dtype)
            band.attrs["nodata"] = int_nodata

        # Export other bands as float32 data types
        else:
            band = band.astype(float_dtype)

        # Export band to file
        if output_location is not None:
            band.odc.write_cog(
                fname=f"{output_location}/{band.name}.tif", overwrite=overwrite
            )

        return band

    # Use default list of bands to convert to integers if none provided
    if int_bands is None:
        int_bands = (
            # Primary layers
            "exposure",
            "extents",
            # Tide attribute layers
            "ta_spread",
            "ta_offset_high",
            "ta_offset_low",
            # QA layers
            "qa_ndwi_freq",
        )

    # Apply to each array in the input `ds`
    return ds.apply(
        lambda x: _prepare_band(
            x, int_bands, int_nodata, int_dtype, float_dtype, output_location, overwrite
        )
    )


def export_dataset_metadata(
    ds,
    year,
    study_area,
    output_location,
    ls_lineage=None,
    s2_lineage=None,
    ancillary_lineage=None,
    dataset_version="0.0.1",
    product_maturity="provisional",
    dataset_maturity="final",
    debug=True,
    run_id=None,
    log=None,
):
    """
    Exports a DEA Intertidal product dataset package including thumbnail
    and processed STAC and ODC metadata for indexing.

    Parameters
    ----------
    ds : xarray.Dataset
        Processed DEA Intertidal data.
    year : int
        Centre year to use for the dataset's start end end date range.
    study_area : str
        A string providing the GridSpec tile ID (e.g. in the form 'x143y56')
        to use as ODC Region code.
    output_location : str
        Location to output the data; supports both local disk and S3.
    s2_lineage : list, optional
        Any Sentinel-2 ODC datasets used for product generation to
        record as source/lineage, e.g. as produced by `dc.find_datasets()`.
        Default is None.
    ls_lineage : list, optional
        Any Landsat ODC datasets used for product generation to record
        as source/lineage, e.g. as produced by `dc.find_datasets()`.
        Default is None.
    ancillary_lineage : list, optional
        ODC datasets from all other ancillarly products used for product
        generation to record as source/lineage, e.g. as produced by
        `dc.find_datasets()`. Default is None.
    dataset_version : str, optional
        Dataset version to use for the output dataset. Default is "0.0.1".
    product_maturity : str, optional
        Product maturity to use for the output dataset. Default is
        "provisional".
    dataset_maturity : str, optional
        Dataset maturity to use for the output dataset. Default is
        "final".
    debug : bool, optional
        When true, this will write S3 outputs locally so they can be
        checked for correctness. Default is True.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.
    """
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists        
    run_id = "Processing" if run_id is None else run_id

    # Use a temporary directory to write outputs to before we either copy
    # it locally or sync to S3
    with tempfile.TemporaryDirectory() as temp_dir:
        log.info(f"{run_id}: Assembling dataset")

        # Open a DatasetAssembler object using the temporary directory
        # and the DEA Collection 3 Naming Convention for output file names
        with DatasetAssembler(
            collection_location=Path(temp_dir),
            naming_conventions="dea_c3",
        ) as dataset_assembler:
            # General product details
            dataset_assembler.product_family = "intertidal"
            dataset_assembler.producer = "ga.gov.au"

            # Platforms and intruments
            platform, instrument = _ls_platform_instrument(year)
            dataset_assembler.platform = platform
            dataset_assembler.instrument = instrument

            # Spatial and temporal information
            dataset_assembler.region_code = study_area
            dataset_assembler.datetime = f"{year}-01-01"
            dataset_assembler.datetime_range = (
                f"{year}-01-01",
                f"{year}-12-31T23:59:59.999999",
            )
            dataset_assembler.processed_now()

            # Product maturity and versioning
            dataset_assembler.product_maturity = product_maturity
            dataset_assembler.maturity = dataset_maturity
            dataset_assembler.dataset_version = dataset_version

            # Set additional properties
            dataset_assembler.properties.update(
                {
                    "odc:product": "ga_s2ls_intertidal_cyear_3",
                    "odc:file_format": "GeoTIFF",
                    "odc:collection_number": 3,
                    "eo:gsd": ds.odc.geobox.resolution.x,
                }
            )

            # Update to temporal naming convention
            time_convention = f"{year}--P1Y"
            dataset_assembler.names.time_folder = time_convention
            label_parts = dataset_assembler.names.dataset_label.split("_")
            label_parts[-2] = time_convention
            dataset_assembler.names.dataset_label = "_".join(label_parts)

            # Write measurements from xarray, extracting nodata values
            # from each input array and assigning these on the outputs
            for dataarray in ds:
                log.info(f"{run_id}: Writing array {dataarray}")
                nodata = ds[dataarray].attrs.get("nodata", None)
                dataset_assembler.write_measurements_odc_xarray(
                    ds[[dataarray]], nodata=nodata
                )

            # Add lineage
            s2_set = set(d.id for d in s2_lineage) if s2_lineage else []
            ls_set = set(d.id for d in ls_lineage) if ls_lineage else []
            ancillary_set = (
                set(d.id for d in ancillary_lineage) if ancillary_lineage else []
            )
            dataset_assembler.note_source_datasets("s2_ard", *s2_set)
            dataset_assembler.note_source_datasets("ls_ard", *ls_set)
            dataset_assembler.note_source_datasets("ancillary", *ancillary_set)

            # Add a starting thumbnail; this will be overwritten later
            dataset_assembler.write_thumbnail("elevation", "elevation", "elevation")

            # Complete the dataset
            dataset_id, metadata_path = dataset_assembler.done()
            log.info(f"{run_id}: Assembled dataset: {metadata_path}")

            # Replace the thumbnail with something nicer
            thumbnail_path = (
                dataset_assembler.names.dataset_path
                / dataset_assembler.names.thumbnail_filename()
            )
            _write_thumbnail(da=ds.elevation, path=thumbnail_path, max_resolution=320)

            # Generate final destination path
            destination_path = (
                f"{output_location.rstrip('/')}/"
                f"{dataset_assembler.names.dataset_folder}/"
            )

            # Export STAC metadata using destination path to correctly
            # populate required metadata/dataset links. This step
            # also ensures all previous data was written out correctly.
            if "dea-public-data-dev" in output_location:
                explorer_url = "https://explorer.dev.dea.ga.gov.au"
            else:
                explorer_url = "https://explorer.dea.ga.gov.au"
            _write_stac(
                dataset_assembler,
                destination_path=destination_path,
                explorer_base_url=explorer_url,
            )

            # Either sync to S3 or copy files to local destination
            if _is_s3(destination_path):
                s3_command = [
                    "aws",
                    "s3",
                    "sync",
                    "--only-show-errors",
                    "--acl bucket-owner-full-control",
                    str(dataset_assembler.names.dataset_path),
                    str(destination_path),
                ]

                if debug:
                    # Copy from tempfile to output location
                    destination_path_debug = (
                        f"{'data/processed/'.rstrip('/')}/"
                        f"{dataset_assembler.names.dataset_folder}"
                    )
                    log.info(
                        f"{run_id}: Writing debug S3 layers to: {destination_path_debug}"
                    )
                    if Path(destination_path_debug).exists():
                        shutil.rmtree(destination_path_debug)
                    shutil.copytree(
                        dataset_assembler.names.dataset_path, destination_path_debug
                    )
                    return dataset_assembler

                else:
                    log.info(f"{run_id}: Writing to S3: {destination_path}")
                    subprocess.run(" ".join(s3_command), shell=True, check=True)

            else:
                # Copy from tempfile to output location
                log.info(f"{run_id}: Writing data locally: {destination_path}")
                if Path(destination_path).exists():
                    shutil.rmtree(destination_path)
                shutil.copytree(dataset_assembler.names.dataset_path, destination_path)
