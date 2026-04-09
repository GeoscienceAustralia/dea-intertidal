"""
Mosaic Metadata Generator

Generate ODC YAML and STAC JSON metadata for DEA coastal ecosystem products.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import click
import eodatasets3.stac as eo3stac
import requests
import rioxarray
import s3fs
import xarray as xr
import yaml
from eodatasets3 import DatasetPrepare, GridSpec, serialise
from eodatasets3.validate import validate_dataset
from matplotlib.colors import ListedColormap
from shapely.geometry import box

from intertidal.io import (
    _is_s3,
    _s2_platform_instrument,
    _s2ls_platform_instrument,
    _write_thumbnail,
)


def _write_thumbnail_cem(da: xr.DataArray, path: str, max_resolution: int = 320):
    """
    Generate and save a thumbnail image from a DEA Coastal Ecosystems
    classification `xarray.DataArray`.

    The thumbnail is reprojected to the specified maximum resolution,
    colorized using a custom colormap based on QGIS style, and
    compressed as a JPEG with the specified quality.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray containing DEA Coastal Ecosystems
        classification data.
    path : str
        The path where the thumbnail image will be saved.
    max_resolution : int, optional
        The maximum resolution of the thumbnail image, by default 320.
    """

    # Define custom colormap for coastal ecosystems based on QGIS style
    # Values from: cem_manual_edits_3577_v1.qml
    colors = {
        2: "#823f4b",  # Tidal flat - burgundy/maroon
        3: "#268f52",  # Mangroves - green
        4: "#74bcfb",  # Saltmarsh - light blue
        5: "#3d4ca5",  # Seagrass - deep blue
        6: "#f2f20c",  # Saltpan - yellow
    }

    # Create colormap array (need 256 colors for standard colormap)
    # Use white for background/nodata
    color_array = ["#ffffff"] * 256  # Initialize with white
    for value, color in colors.items():
        if value < 256:
            color_array[value] = color

    # Convert hex colors to RGBA tuples (0-1 range)
    def hex_to_rgba(hex_color):
        if len(hex_color) == 9:  # #RRGGBBAA format
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            a = int(hex_color[7:9], 16) / 255.0
            return (r, g, b, a)
        else:  # #RRGGBB format
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            return (r, g, b, 1.0)

    color_list = [hex_to_rgba(c) for c in color_array]
    cmap = ListedColormap(color_list)

    # Reproject and prepare data
    reprojected = da.odc.reproject(
        how=da.odc.geobox.zoom_to(max_resolution),
        resampling="mode",  # Use mode for categorical data
    )

    # Replace NaN/nodata with 255 (will be white)
    reprojected = reprojected.fillna(255)

    jpeg_data = reprojected.odc.colorize(vmin=0, vmax=255, cmap=cmap).odc.compress(
        "jpeg", 85, transparent=[255, 255, 255]
    )  # White as transparent

    with open(path, "wb") as f:
        f.write(jpeg_data)


@dataclass
class ProductConfig:
    """Configuration for a product metadata generation."""

    year: str
    version: str
    product_name: str
    product_family: str
    study_area: str = "AU"
    freq: str = "P1Y"
    product_maturity: str = "stable"
    dataset_maturity: str = "final"
    nodata_value: int = 255
    dev: bool = True
    naming_conventions: str = "dea_c3"  # 'dea' or 'dea_c3'

    def __post_init__(self):
        """Set up derived paths and filenames."""
        # For dea_c3, don't include dataset_maturity in filenames
        if self.naming_conventions == "dea_c3":
            # Use 'mosaic' as the base identifier for continental mosaics
            if "continental" in self.study_area.lower() or self.study_area == "AU":
                # Continental mosaic: ga_s2ls_intertidal_cyear_3_mosaic_2024--P1Y
                self.title = f"{self.product_name}_mosaic_{self.year}--{self.freq}"
                self.s3_folder_base = f"{self.version}/continental_mosaics"
            else:
                # Regional mosaic: ga_s2ls_intertidal_cyear_3_QLD_2024--P1Y
                self.title = (
                    f"{self.product_name}_{self.study_area}_{self.year}--{self.freq}"
                )
                self.s3_folder_base = f"{self.version}/{self.study_area}"
        else:
            # Standard dea naming with dataset_maturity
            # Extract last two numbers from version (e.g., '1-0-0' -> '0-0', '1-1-0' -> '1-0')
            version_parts = self.version.split("-")
            if len(version_parts) >= 3:
                version_suffix = f"{version_parts[-2]}-{version_parts[-1]}"
            else:
                version_suffix = (
                    self.version
                )  # Fallback to full version if format unexpected

            # ga_s2_coastalecosystems_cyear_3_v1-0-0_AU_2022--P1Y_final
            self.title = f"{self.product_name}-{version_suffix}_{self.study_area}_{self.year}--{self.freq}_{self.dataset_maturity}"
            self.s3_folder_base = self.study_area

        # S3 paths
        s3_bucket = "dea-public-data-dev" if self.dev else "dea-public-data"
        self.s3_folder = f"s3://{s3_bucket}/derivative/{self.product_name}/{self.s3_folder_base}/{self.year}--{self.freq}/"

        # Local and S3 metadata paths
        self.thumbnail_filename = f"{self.title}-thumbnail.jpg"
        self.local_stac_path = f"{self.title}.stac-item.json"
        self.local_odc_path = f"{self.title}.odc-metadata.yaml"
        self.s3_stac_path = f"{self.s3_folder}{self.local_stac_path}"
        self.s3_odc_path = f"{self.s3_folder}{self.local_odc_path}"

        # Explorer and lineage paths
        explorer_domain = (
            "explorer.dev.dea.ga.gov.au" if self.dev else "explorer.dea.ga.gov.au"
        )
        self.explorer_base_url = f"https://{explorer_domain}/"
        self.dataset_location = self.explorer_base_url
        self.s3_thumbnail_path = f"{self.s3_folder}/{self.thumbnail_filename}"

        # Accessory files
        self.tile_lineage_filename = f"{self.title}_tile-lineage.json"
        self.metadata_processor_filename = f"{self.title}.proc-info.yaml"

        # Lineage file
        self.compiled_lineage_filename = f"metadata/{self.product_name}_v{self.version}_{self.year}--{self.freq}_lineage.json"


def get_gridspec(ds: xr.Dataset) -> GridSpec:
    """
    Extract GridSpec from an xarray Dataset with spatial information.

    Args:
        ds: xarray Dataset with rioxarray spatial metadata

    Returns:
        GridSpec object for eodatasets3
    """
    # Extract shape
    ny = ds.sizes["y"]
    nx = ds.sizes["x"]
    shape = (ny, nx)

    # Extract transform and CRS
    transform = ds.rio.transform()
    crs = ds.rio.crs

    return GridSpec(shape=shape, transform=transform, crs=crs)


def load_tif_datasets(
    s3_base: str, file_endings: List[str], verbose: bool = False
) -> Dict[str, xr.Dataset]:
    """
    Load multiple GeoTIFF files from S3 into xarray Datasets.

    Args:
        s3_base: Base S3 path (e.g., 's3://bucket/path/prefix_')
        file_endings: List of file endings to load
        verbose: Print loading progress

    Returns:
        Dictionary mapping variable names to Datasets
    """
    data_vars = {}

    for ending in file_endings:
        s3_path = s3_base + ending
        if verbose:
            print(f"Loading {ending}...")

        # Load with rioxarray
        da = rioxarray.open_rasterio(s3_path, chunks="auto")
        da = rioxarray.open_rasterio(s3_path, chunks={})

        # Create a clean variable name from the filename
        var_name = ending.replace(".tif", "").replace("-", "_")

        # If there's only one band, squeeze it out
        if "band" in da.dims and len(da.band) == 1:
            da = da.squeeze("band", drop=True)

        # Cast probability layers to uint8
        if "prob" in var_name:
            da = da.astype("uint8", keep_attrs=True)
            if verbose:
                print(f"  Cast {var_name} to uint8")

        # Store in dictionary as Dataset
        ds = xr.Dataset({var_name: da})
        data_vars[var_name] = ds

        if verbose:
            print(f"  Loaded {var_name} with shape {da.shape}, dtype {da.dtype}")

    return data_vars


def load_lineage(
    lineage_file: pathlib.Path, verbose: bool = False
) -> Dict[str, List[str]]:
    """
    Load compiled lineage from JSON file.

    Args:
        lineage_file: Path to lineage JSON file
        verbose: Print loading information

    Returns:
        Dictionary with lineage categories and their source datasets
    """
    with open(lineage_file, "r") as f:
        lineage_dict = json.load(f)

    if verbose:
        print(f"Loaded compiled lineage from: {lineage_file}")
        print(f"Keys in compiled_lineage: {list(lineage_dict.keys())}")
        print("Summary:")
        for key, values in lineage_dict.items():
            print(f"  {key}: {len(values)} unique values")

    return lineage_dict


def extract_lineage_from_tiles(
    s3_bucket: str,
    product: str,
    version: str,
    year: str,
    freq: str = "P1Y",
    naming_conventions: str = "dea_c3",
    tile_naming_conventions: str = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract lineage from tile STAC files in S3.

    Args:
        s3_bucket: S3 bucket path (e.g., 'dea-public-data-dev/derivative')
        product: Product name (e.g., 'ga_s2ls_intertidal_cyear_3')
        version: Version string (e.g., '2-1-0')
        year: Year string (e.g., '2024')
        freq: Frequency code (default: 'P1Y')
        naming_conventions: Mosaic naming convention 'dea' or 'dea_c3' (default: 'dea_c3')
        tile_naming_conventions: Tile naming convention (if different from mosaic).
                                 If None, uses same as naming_conventions.
                                 For coastal ecosystems: mosaic uses 'dea', tiles use 'dea_c3'
        verbose: Print progress

    Returns:
        Dictionary with lineage categories and their source datasets
    """
    # Use tile_naming_conventions if provided, otherwise use naming_conventions
    tile_naming = (
        tile_naming_conventions if tile_naming_conventions else naming_conventions
    )

    # Build pattern based on TILE naming convention
    if tile_naming == "dea_c3":
        # dea_c3: includes version in path
        pattern = (
            f"{s3_bucket}/{product}/{version}/x*/y*/{year}--{freq}/*.stac-item.json"
        )
    else:
        # dea: no version in path
        pattern = f"{s3_bucket}/{product}/x*/y*/{year}--{freq}/*.stac-item.json"

    if verbose:
        print(f"Searching for STAC files: {pattern}")
        print(f"Mosaic naming convention: {naming_conventions}")
        print(f"Tile naming convention: {tile_naming}")

    fs = s3fs.S3FileSystem(anon=True)
    all_files = fs.glob(pattern)

    if verbose:
        print(f"Found {len(all_files)} STAC files for {product}\n")

    # Dictionary to compile all unique lineage values
    compiled_lineage = defaultdict(set)

    if verbose:
        print("Processing STAC metadata files...")

    for i, file_path in enumerate(all_files, 1):
        try:
            # Read STAC metadata file from S3
            with fs.open(f"s3://{file_path}", "r") as f:
                stac_metadata = json.load(f)

            # Extract lineage from properties
            lineage = stac_metadata.get("properties", {}).get("odc:lineage", {})

            # Add to compiled lineage (convert lists to sets for uniqueness)
            for key, values in lineage.items():
                if isinstance(values, list):
                    compiled_lineage[key].update(values)

            if verbose and i % 100 == 0:
                print(f"  Processed {i}/{len(all_files)} files...")

        except Exception as e:
            if verbose:
                print(f"  Error processing {file_path}: {e}")
            continue

    # Convert compiled lineage sets back to sorted lists
    compiled_lineage_dict = {
        key: sorted(list(values)) for key, values in compiled_lineage.items()
    }

    if verbose:
        print("\n=== Lineage Summary ===")
        print(f"Total tiles processed: {len(all_files)}")
        for key, values in compiled_lineage_dict.items():
            print(f"  {key}: {len(values)} unique UUIDs")

    return compiled_lineage_dict


def get_product_bands(product_family: str) -> Dict[str, str]:
    """
    Get band definitions for a product family.

    Args:
        product_family: Product family name

    Returns:
        Dictionary mapping internal band names to file suffixes (without .tif extension)
    """
    if product_family == "coastalecosystems":
        return {
            "classification": "classification",
            "mangrove_prob": "mangrove-prob",
            "seagrass_prob": "seagrass-prob",
            "saltmarsh_prob": "saltmarsh-prob",
            "saltflat_prob": "saltflat-prob",
            "qa_coastal_connectivity": "qa-coastal-connectivity",
            "qa_count_clear": "qa-count-clear",
        }
    elif product_family == "intertidal":
        return {
            "exposure": "exposure",
            "elevation": "elevation",
            "extents": "extents",
            "elevation_uncertainty": "elevation-uncertainty",
            "ta_spread": "ta-spread",
            "ta_offset_high": "ta-offset-high",
            "ta_offset_low": "ta-offset-low",
            "ta_lat": "ta-lat",
            "qa_count_clear": "qa-count-clear",
            "qa_ndwi_corr": "qa-ndwi-corr",
            "ta_hot": "ta-hot",
            "qa_ndwi_freq": "qa-ndwi-freq",
            "qa_coastal_connectivity": "qa-coastal-connectivity",
            "ta_hat": "ta-hat",
            "ta_lot": "ta-lot",
        }
    elif product_family == "tidal_composites":
        return {
            "low_coastal_aerosol": "low-coastal-aerosol",
            "low_blue": "low-blue",
            "low_green": "low-green",
            "low_red": "low-red",
            "low_red_edge_1": "low-red-edge-1",
            "low_red_edge_2": "low-red-edge-2",
            "low_red_edge_3": "low-red-edge-3",
            "low_nir_1": "low-nir-1",
            "low_nir_2": "low-nir-2",
            "low_swir_2": "low-swir-2",
            "low_swir_3": "low-swir-3",
            "high_coastal_aerosol": "high-coastal-aerosol",
            "high_blue": "high-blue",
            "high_green": "high-green",
            "high_red": "high-red",
            "high_red_edge_1": "high-red-edge-1",
            "high_red_edge_2": "high-red-edge-2",
            "high_red_edge_3": "high-red-edge-3",
            "high_nir_1": "high-nir-1",
            "high_nir_2": "high-nir-2",
            "high_swir_2": "high-swir-2",
            "high_swir_3": "high-swir-3",
            "qa_low_threshold": "qa-low-threshold",
            "qa_high_threshold": "qa-high-threshold",
            "qa_count_clear": "qa-count-clear",
        }
    else:
        raise ValueError(f"Unknown product family: {product_family}")


def generate_metadata(
    config: ProductConfig,
    data_vars: Dict[str, xr.Dataset],
    lineage_dict: Dict[str, List[str]],
    output_dir: pathlib.Path,
    generate_stac: bool = True,
    verbose: bool = False,
) -> Tuple[pathlib.Path, Optional[pathlib.Path]]:
    """
    Generate ODC YAML and optionally STAC JSON metadata.

    Args:
        config: Product configuration
        data_vars: Dictionary of loaded datasets
        lineage_dict: Lineage information
        output_dir: Directory for output metadata files
        generate_stac: Whether to generate STAC metadata
        verbose: Print detailed information

    Returns:
        Tuple of (odc_path, stac_path or None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up paths
    metadata_path = output_dir / config.local_odc_path
    stac_path = output_dir / config.local_stac_path if generate_stac else None

    # Get bands for this product
    bands = get_product_bands(config.product_family)

    # Get reference dataset for bounds (use first available band)
    ref_band = next(iter(data_vars.keys()))
    ref_ds = data_vars[ref_band]
    bounds = ref_ds.rio.bounds()
    extent_geometry = box(*bounds)

    # Calculate max segment length for ~10 points per edge
    minx, miny, maxx, maxy = bounds
    width = abs(maxx - minx)
    height = abs(maxy - miny)
    max_segment_length = min(width / 10, height / 10)
    extent_geometry = extent_geometry.segmentize(max_segment_length)

    # Determine platform and instrument
    year_int = int(config.year)
    if config.product_family == "intertidal":
        platform, instrument = _s2ls_platform_instrument(year_int)
    elif config.product_family in ("tidal_composites", "coastalecosystems"):
        platform, instrument = _s2_platform_instrument(year_int)
    else:
        raise ValueError(f"Unknown product family: {config.product_family}")

    # Generate ODC YAML metadata first
    if verbose:
        print(f"Generating ODC YAML metadata: {metadata_path}")

    # Generate thumbnails using COG overviews for speed
    if config.product_family == "intertidal":
        if verbose:
            print("Generating thumbnail from elevation overview...")
        # Read a lower resolution overview of the COG
        tif_filename = f"{config.s3_folder}{config.title}_elevation.tif"
        overview_level = 6
        cog_array = rioxarray.open_rasterio(tif_filename, overview_level=overview_level)

        # Squeeze out band dimension if present
        if "band" in cog_array.dims and len(cog_array.band) == 1:
            cog_array = cog_array.squeeze("band", drop=True)

        # Write thumbnail to output directory
        thumbnail_path = output_dir / config.thumbnail_filename
        _write_thumbnail(da=cog_array,
            path=str(thumbnail_path),
            max_resolution=320,
            vmin=-2.5,
            vmax=1.5,
        )

    elif config.product_family == "coastalecosystems":
        if verbose:
            print("Generating thumbnail from classification overview...")
        # Read a lower resolution overview of the classification COG
        tif_filename = f"{config.s3_folder}{config.title}_classification.tif"
        overview_level = 6
        cog_array = rioxarray.open_rasterio(tif_filename, overview_level=overview_level)

        # Squeeze out band dimension if present
        if "band" in cog_array.dims and len(cog_array.band) == 1:
            cog_array = cog_array.squeeze("band", drop=True)

        # Write thumbnail to output directory
        thumbnail_path = output_dir / config.thumbnail_filename
        _write_thumbnail_cem(da=cog_array, path=str(thumbnail_path), max_resolution=320)

    elif config.product_family == "tidal_composites":
        if verbose:
            print("Generating RGB thumbnail from low-tide bands...")

        # Read RGB bands from overview
        rgb_arrays = []
        for band in ["low-red", "low-green", "low-blue"]:
            tif_filename = f"{config.s3_folder}{config.title}_{band}.tif"
            overview_level = 6
            cog_array = rioxarray.open_rasterio(
                tif_filename,
                overview_level=overview_level,
                mask_and_scale=True,
                default_name=band,
            )
            rgb_arrays.append(cog_array)

        # Stack RGB bands
        rgb_stacked = xr.merge(rgb_arrays).squeeze("band", drop=True)

        # Write RGB thumbnail
        thumbnail_path = output_dir / config.thumbnail_filename
        _write_thumbnail(
            da=rgb_stacked,
            path=str(thumbnail_path),
            max_resolution=320,
            vmin=0,
            vmax=2000,
        )

    with DatasetPrepare(
        collection_location=(
            "s3://dea-public-data-dev" if config.dev else "s3://dea-public-data"
        ),
        dataset_location=config.dataset_location,
        metadata_path=metadata_path,
    ) as p:
        # Basic product information
        p.product_family = config.product_family
        p.product_name = config.product_name
        p.label = config.title
        p.platform = platform
        p.instrument = instrument
        p.geometry = extent_geometry

        # Metadata properties
        p.properties["odc:file_format"] = "GeoTIFF"
        p.properties["odc:producer"] = "ga.gov.au"
        p.properties["odc:product_family"] = config.product_family
        p.collection_number = 3
        p.properties["eo:gsd"] = 10

        # Spatial and temporal information
        p.region_code = config.study_area
        p.datetime = f"{config.year}-01-01"
        p.datetime_range = (
            f"{config.year}-01-01",
            f"{config.year}-12-31T23:59:59.999999",
        )

        # Product maturity and versioning
        p.product_maturity = config.product_maturity
        p.maturity = config.dataset_maturity
        p.dataset_version = config.version.replace("-", ".")
        p.processed_now()

        # Accessory files
        p.note_thumbnail(config.thumbnail_filename, "jpg")
        p.note_accessory_file("metadata:processor", config.metadata_processor_filename)

        # Add source datasets from lineage
        for lin_type in lineage_dict.keys():
            p.note_source_datasets(lin_type, *set(lineage_dict[lin_type]))

        # Set naming conventions from config
        p.naming_conventions = config.naming_conventions

        # Add measurements
        for band_name, file_suffix in bands.items():
            if band_name not in data_vars:
                if verbose:
                    print(f"Warning: {band_name} not found in data_vars, skipping")
                continue

            if verbose:
                print(f"Adding measurement: {band_name}")

            tif_filename = f"{config.title}_{file_suffix}.tif"
            g = get_gridspec(data_vars[band_name])

            # Set appropriate nodata value
            nodata = 65535 if "connectivity" in band_name else config.nodata_value

            p.note_measurement(
                band_name,
                tif_filename,
                relative_to_dataset_location=True,
                grid=g,
                nodata=nodata,
                expand_valid_data=False,
            )

        # Write ODC YAML
        p.done()

    # Generate STAC metadata from the written YAML file
    if generate_stac:
        if verbose:
            print(f"Generating STAC metadata: {stac_path}")

        # Load the dataset from the written YAML file
        dataset_doc = serialise.from_path(metadata_path)

        # Generate STAC JSON
        stac_item = eo3stac.to_stac_item(
            dataset=dataset_doc,
            stac_item_destination_url=config.s3_stac_path,
            dataset_location=config.s3_folder,
            odc_dataset_metadata_url=config.s3_odc_path,
            explorer_base_url=config.explorer_base_url,
        )

        # Write STAC JSON
        with open(stac_path, "w") as f:
            json.dump(stac_item, f, indent=4)

    if verbose:
        print("\nGenerated metadata:")
        print(f"  ODC YAML: {metadata_path}")
        if generate_stac:
            print(f"  STAC JSON: {stac_path}")

    return metadata_path, stac_path


def validate_metadata_file(metadata_file: pathlib.Path, verbose: bool = True) -> List:
    """
    Validate an EO3 metadata file (STAC JSON or ODC YAML).

    Args:
        metadata_file: Path to metadata file
        verbose: Print validation results

    Returns:
        List of validation messages
    """
    if verbose:
        print(f"Validating: {metadata_file.name}")
        print("=" * 80)

    # Load as dictionary
    if metadata_file.suffix == ".json":
        with open(metadata_file, "r") as f:
            metadata_dict = json.load(f)
    elif metadata_file.suffix in [".yaml", ".yml"]:
        with open(metadata_file, "r") as f:
            metadata_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {metadata_file.suffix}")

    # Validate
    validation_messages = list(validate_dataset(metadata_dict))

    # Filter out false positive for STAC files
    is_stac = metadata_file.suffix == ".json"
    if is_stac:
        # STAC items don't have $schema - that's expected, not an error
        validation_messages = [
            msg
            for msg in validation_messages
            if not (msg.code == "no_schema" and msg.level.name == "error")
        ]

    if verbose:
        # Categorize messages by level
        errors = []
        warnings = []
        info = []

        for msg in validation_messages:
            if msg.level.name == "error":
                errors.append((msg.code, msg.reason))
            elif msg.level.name == "warning":
                warnings.append((msg.code, msg.reason))
            else:
                info.append((msg.code, msg.reason))

        # Print results
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for code, reason in errors:
                print(f"  {code}: {reason}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for code, reason in warnings:
                print(f"  {code}: {reason}")

        if info:
            print(f"\n💡 INFO ({len(info)}):")
            for code, reason in info:
                print(f"  {code}: {reason}")

        if not validation_messages:
            print("\n✅ All validation checks passed!")

        print("\n")

    return validation_messages


# ============================================================================
# Command-Line Interface
# ============================================================================


@click.group()
@click.version_option()
def cli():
    """
    Generate and validate metadata for DEA coastal ecosystem products.

    This tool creates ODC YAML and STAC JSON metadata files for mosaic products
    stored in S3, including coastal ecosystem classifications and probability layers.
    """
    pass


@cli.command()
@click.option("--year", required=True, help="Year of the product (e.g., 2022)")
@click.option(
    "--product",
    required=True,
    help="Product name (e.g., ga_s2_coastalecosystems_cyear_3_v1)",
)
@click.option("--version", required=True, help="Product version (e.g., 1-0-0)")
@click.option(
    "--product-family",
    default=None,
    help="Product family (auto-detected from product name if not specified: intertidal, tidal_composites, or coastalecosystems)",
)
@click.option("--study-area", default="AU", help="Study area code (default: AU)")
@click.option("--freq", default="P1Y", help="Frequency code (default: P1Y)")
@click.option(
    "--product-maturity", default="stable", help="Product maturity (default: stable)"
)
@click.option(
    "--dataset-maturity", default="final", help="Dataset maturity (default: final)"
)
@click.option(
    "--nodata-value", default=255, type=int, help="NoData value (default: 255)"
)
@click.option(
    "--naming-conventions",
    default="dea_c3",
    type=click.Choice(["dea", "dea_c3"]),
    help="Naming convention: dea (includes maturity in filename) or dea_c3 (no maturity, for continental mosaics, default: dea_c3)",
)
@click.option(
    "--dev/--prod", default=True, help="Use dev or production S3 bucket (default: dev)"
)
@click.option(
    "--lineage-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to compiled lineage JSON file",
)
@click.option(
    "--auto-lineage",
    is_flag=True,
    help="Auto-generate empty lineage (for testing/development)",
)
@click.option(
    "--extract-lineage-from-tiles/--no-extract-lineage-from-tiles",
    "extract_from_tiles",
    default=True,
    help="Extract lineage from tile STAC files in S3 (default: True)",
)
@click.option(
    "--tile-dir",
    default="dea-public-data-dev/derivative",
    help="S3 bucket/path for tiles (default: dea-public-data-dev/derivative)",
)
@click.option(
    "--output-dir",
    type=str,
    default="metadata",
    help="Output destination (S3 or local path, default: metadata)",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate generated metadata (default: True)",
)
@click.option("--verbose", is_flag=True, help="Verbose output")
def generate(
    year: str,
    product: str,
    version: str,
    product_family: Optional[str],
    study_area: str,
    freq: str,
    product_maturity: str,
    dataset_maturity: str,
    nodata_value: int,
    naming_conventions: str,
    dev: bool,
    lineage_file: Optional[pathlib.Path],
    auto_lineage: bool,
    extract_from_tiles: bool,
    tile_dir: str,
    output_dir: str,
    validate: bool,
    verbose: bool,
):
    """
    Generate ODC YAML and STAC JSON metadata for a product.
    
    Example:
    
        mosaic-metadata generate --year 2022 \\
            --product ga_s2_coastalecosystems_cyear_3_v1 \\
            --version 1-0-0
    
    Product family is auto-detected from product name, or specify explicitly:
    
        mosaic-metadata generate --year 2022 \\
            --product ga_s2_coastalecosystems_cyear_3_v1 \\
            --version 1-0-0 \\
            --product-family coastalecosystems
    """
    # Auto-detect product_family from product name if not specified
    if product_family is None:
        if "intertidal" in product.lower():
            product_family = "intertidal"
        elif "tidal_composites" in product.lower():
            product_family = "tidal_composites"
        elif "coastalecosystems" in product.lower():
            product_family = "coastalecosystems"
        else:
            click.echo(
                f"Error: Could not auto-detect product family from product name: {product}",
                err=True,
            )
            click.echo(
                "Product name must contain 'intertidal', 'tidal_composites', or 'coastalecosystems'",
                err=True,
            )
            click.echo("Or specify --product-family explicitly", err=True)
            sys.exit(1)

        if verbose:
            click.echo(f"Auto-detected product family: {product_family}")

    # Create config
    config = ProductConfig(
        year=year,
        version=version,
        product_name=product,
        product_family=product_family,
        study_area=study_area,
        freq=freq,
        product_maturity=product_maturity,
        dataset_maturity=dataset_maturity,
        nodata_value=nodata_value,
        naming_conventions=naming_conventions,
        dev=dev,
    )

    if verbose:
        click.echo("Configuration:")
        click.echo(f"  Year: {config.year}")
        click.echo(f"  Product: {config.product_name}")
        click.echo(f"  Title: {config.title}")
        click.echo(f"  S3 folder: {config.s3_folder}")
        click.echo("")

    # Determine lineage file path
    if lineage_file is None:
        lineage_file = pathlib.Path(config.compiled_lineage_filename)

    # Load or generate lineage
    if auto_lineage:
        # Auto-generate empty lineage for testing/development
        click.echo("Auto-generating empty lineage...")
        lineage_dict = {
            "s2_ard": [],
            "landsat_ard": [],
            "ancillary": [],
        }
        if verbose:
            click.echo("  Using empty lineage (no source datasets)")
    elif lineage_file and lineage_file.exists():
        # Use provided lineage file
        lineage_dict = load_lineage(lineage_file, verbose=verbose)
    elif extract_from_tiles:
        # Extract lineage from tile STAC files (default behavior)
        click.echo("Extracting lineage from tile STAC files...")

        if verbose:
            click.echo(f"  Tile directory: {tile_dir}")
            click.echo(f"  Product: {product}")
            click.echo(f"  Version: {version}")
            click.echo(f"  Year: {year}")

        try:
            # All coastal products have tiles with dea_c3 naming (with version in path)
            coastal_products = ["tidal_composites", "intertidal", "coastalecosystems"]
            tile_naming = (
                "dea_c3" if product_family in coastal_products else naming_conventions
            )

            lineage_dict = extract_lineage_from_tiles(
                s3_bucket=tile_dir,
                product=product,
                version=version,
                year=year,
                freq=freq,
                naming_conventions=naming_conventions,
                tile_naming_conventions=tile_naming,
                verbose=verbose,
            )
        except Exception as e:
            click.echo(f"Error extracting lineage from tiles: {e}", err=True)
            if verbose:
                traceback.print_exc()
            sys.exit(1)
    else:
        click.echo("Error: No lineage source specified", err=True)
        click.echo("\nOptions:", err=True)
        click.echo(
            "  1. Use --extract-lineage-from-tiles (default, already enabled)", err=True
        )
        click.echo("  2. Provide --lineage-file with path to a JSON file", err=True)
        click.echo("  3. Use --auto-lineage for empty lineage (testing only)", err=True)
        sys.exit(1)

    # Load TIF datasets from S3
    # Get band definitions
    bands = get_product_bands(config.product_family)
    file_endings = [f"{suffix}.tif" for suffix in bands.values()]

    # Construct S3 base path
    s3_base = f"{config.s3_folder}{config.title}_"

    if verbose:
        click.echo(f"Loading {len(file_endings)} TIF files from S3...")

    try:
        data_vars = load_tif_datasets(s3_base, file_endings, verbose=verbose)
    except Exception as e:
        click.echo(f"Error loading TIF files: {e}", err=True)
        sys.exit(1)

    # Generate metadata
    # Always use temporary directory, then sync to output-dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        if verbose:
            click.echo(f"Using temporary directory: {temp_path}")

        try:
            odc_path, stac_path = generate_metadata(
                config=config,
                data_vars=data_vars,
                lineage_dict=lineage_dict,
                output_dir=temp_path,
                generate_stac=True,
                verbose=verbose,
            )

            click.echo("✅ Successfully generated metadata:")
            click.echo(f"   ODC YAML: {odc_path}")
            if stac_path:
                click.echo(f"   STAC JSON: {stac_path}")

        except Exception as e:
            click.echo(f"Error generating metadata: {e}", err=True)
            if verbose:
                traceback.print_exc()
            sys.exit(1)

        # Validate if requested
        if validate:
            click.echo("\nValidating metadata...")

            # Validate ODC metadata
            try:
                validation_msgs = validate_metadata_file(odc_path, verbose=True)
                has_errors = any(msg.level.name == "error" for msg in validation_msgs)

                if has_errors:
                    click.echo("❌ ODC metadata validation failed", err=True)
                    sys.exit(1)
            except Exception as e:
                click.echo(f"Error validating ODC metadata: {e}", err=True)
                sys.exit(1)

            # Validate STAC metadata if generated
            if stac_path and stac_path.exists():
                try:
                    validation_msgs = validate_metadata_file(stac_path, verbose=True)
                except Exception as e:
                    click.echo(f"Error validating STAC metadata: {e}", err=True)
                    sys.exit(1)

        # Sync from temp directory to output-dir
        # Construct final destination path based on whether output_dir is S3 or local
        if _is_s3(output_dir):
            # For S3, use the full S3 path from config
            final_destination = config.s3_folder
        else:
            # For local filesystem, mirror the S3 structure under output_dir
            # Remove s3://bucket-name/derivative/ prefix and prepend output_dir
            # config.s3_folder format: s3://bucket/derivative/product/folder_base/year--P1Y/
            s3_path_parts = config.s3_folder.replace("s3://", "").split("/")
            # Skip bucket name and 'derivative', keep product/folder_base/year--P1Y/
            relative_path = "/".join(s3_path_parts[2:])  # Skip bucket and 'derivative'
            final_destination = f"{output_dir.rstrip('/')}/{relative_path}"

        if _is_s3(final_destination):
            # Sync to S3
            click.echo(f"\nSyncing to S3: {final_destination}")
            s3_command = [
                "aws",
                "s3",
                "sync",
                "--only-show-errors",
                "--acl",
                "bucket-owner-full-control",
                str(temp_path),
                str(final_destination),
            ]

            if verbose:
                click.echo(f"Running: {' '.join(s3_command)}")

            subprocess.run(s3_command, check=True)
            click.echo(f"✅ Successfully synced to S3: {final_destination}")
        else:
            # Copy to local destination
            click.echo(f"\nCopying to local destination: {final_destination}")
            output_path = pathlib.Path(final_destination)

            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                shutil.rmtree(output_path)

            shutil.copytree(temp_path, output_path)
            click.echo(f"✅ Successfully copied to: {final_destination}")


@cli.command()
@click.argument("metadata_files", nargs=-1, type=str)
@click.option("--verbose", is_flag=True, help="Verbose output")
def validate(metadata_files, verbose: bool):
    """
    Validate one or more metadata files (local paths or URLs).

    Example:

        mosaic-metadata validate metadata/*.yaml metadata/*.json
        mosaic-metadata validate https://example.com/metadata.yaml
    """
    if not metadata_files:
        click.echo("Error: No metadata files specified", err=True)
        sys.exit(1)

    all_valid = True

    for metadata_file in metadata_files:
        try:
            # Check if it's a URL
            parsed = urlparse(metadata_file)
            if parsed.scheme in ("http", "https"):
                # Download to temporary file
                if verbose:
                    click.echo(f"Downloading {metadata_file}...")

                response = requests.get(metadata_file)
                response.raise_for_status()

                # Determine file extension from URL
                if metadata_file.endswith(".json"):
                    suffix = ".json"
                elif metadata_file.endswith(".yaml") or metadata_file.endswith(".yml"):
                    suffix = ".yaml"
                else:
                    suffix = ""

                # Write to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False
                ) as f:
                    f.write(response.text)
                    temp_path = pathlib.Path(f.name)

                try:
                    validation_msgs = validate_metadata_file(temp_path, verbose=verbose)
                    has_errors = any(
                        msg.level.name == "error" for msg in validation_msgs
                    )

                    if has_errors:
                        all_valid = False
                finally:
                    # Clean up temp file
                    temp_path.unlink()
            else:
                # Local file
                file_path = pathlib.Path(metadata_file)
                if not file_path.exists():
                    click.echo(f"Error: File does not exist: {metadata_file}", err=True)
                    all_valid = False
                    continue

                validation_msgs = validate_metadata_file(file_path, verbose=verbose)
                has_errors = any(msg.level.name == "error" for msg in validation_msgs)

                if has_errors:
                    all_valid = False
        except Exception as e:
            click.echo(f"Error validating {metadata_file}: {e}", err=True)
            all_valid = False

    if not all_valid:
        sys.exit(1)
    else:
        click.echo("\n✅ All files validated successfully!")


@cli.command()
@click.option("--year", required=True, help="Year of the product")
@click.option("--product", required=True, help="Product name")
@click.option("--version", required=True, help="Product version (e.g., 2-1-0)")
@click.option("--study-area", default="AU", help="Study area code (default: AU)")
@click.option("--freq", default="P1Y", help="Frequency code (default: P1Y)")
@click.option(
    "--dataset-maturity", default="final", help="Dataset maturity (default: final)"
)
@click.option(
    "--naming-conventions",
    default="dea_c3",
    type=click.Choice(["dea", "dea_c3"]),
    help="Naming convention (default: dea_c3)",
)
@click.option(
    "--dev/--prod", default=True, help="Use dev or production bucket (default: dev)"
)
def show_paths(
    year: str,
    product: str,
    version: str,
    study_area: str,
    freq: str,
    dataset_maturity: str,
    naming_conventions: str,
    dev: bool,
):
    """
    Show S3 paths and filenames for a product without generating metadata.
    
    Example:
    
        mosaic-metadata show-paths --year 2022 \\
            --product ga_s2_coastalecosystems_cyear_3_v1 \\
            --version 1-0-0
    """
    # Create minimal config
    config = ProductConfig(
        year=year,
        version=version,
        product_name=product,
        product_family="coastalecosystems",
        study_area=study_area,
        freq=freq,
        dataset_maturity=dataset_maturity,
        naming_conventions=naming_conventions,
        dev=dev,
    )

    click.echo("Product Configuration:")
    click.echo("=" * 80)
    click.echo(f"Title: {config.title}")
    click.echo("\nS3 Paths:")
    click.echo(f"  Folder: {config.s3_folder}")
    click.echo(f"  STAC JSON: {config.s3_stac_path}")
    click.echo(f"  ODC YAML: {config.s3_odc_path}")
    click.echo(f"  Thumbnail: {config.s3_thumbnail_path}")
    click.echo("\nLocal Paths:")
    click.echo(f"  STAC JSON: {config.local_stac_path}")
    click.echo(f"  ODC YAML: {config.local_odc_path}")
    click.echo("\nAccessory Files:")
    click.echo(f"  Processor Info: {config.metadata_processor_filename}")
    click.echo(f"  Tile Lineage: {config.tile_lineage_filename}")
    click.echo("\nExplorer:")
    click.echo(f"  Base URL: {config.explorer_base_url}")


if __name__ == "__main__":
    cli()
