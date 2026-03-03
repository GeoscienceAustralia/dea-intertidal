import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import click
import s3fs
from datacube.utils.aws import configure_s3_access

from intertidal.utils import (
    configure_logging,
)


def _is_s3(path):
    """Determine whether output location is on S3."""
    uu = urlparse(path)
    return uu.scheme == "s3"


@click.command()
@click.option(
    "--product",
    type=str,
    required=True,
    help="The name of the product to be mosaiced, "
    "e.g. 'ga_s2_tidal_composites_cyear_3' or 'ga_s2ls_intertidal_cyear_3'.",
)
@click.option(
    "--band",
    type=str,
    required=True,
    help="The name of the band to be mosaiced, e.g. 'elevation' or 'exposure'.",
)
@click.option(
    "--year",
    type=str,
    required=True,
    help="The year of the data to be mosaiced, e.g. '2022'.",
)
@click.option(
    "--version",
    type=str,
    required=True,
    help="The version number of the product to be mosaiced, e.g. '0-0-1'.",
)
@click.option(
    "--product_dir",
    type=str,
    default="s3://dea-public-data-dev/derivative/",
    help="The directory/location to read the tile COGs from; supports both local disk and S3 locations.",
)
@click.option(
    "--output_dir",
    type=str,
    default="/gdata1/projects/coastal/intertidal/mosaics/",
    help="The directory/location to output data and metadata; supports "
    "both local disk and S3 locations. "
    "Defaults to '/gdata1/projects/coastal/intertidal/mosaics/'. "
    "The function will add on `{product}/{version}/continental_mosaics/{year}--P1Y` "
    "to the provided `ouput_dir`.",
)
@click.option(
    "--dataset_maturity",
    type=str,
    default="final",
    help="The dataset maturity of the data to be mosaiced, e.g. 'final' or 'interim'.",
)
@click.option(
    "--compress",
    type=str,
    default="DEFLATE",
    help="The compression method used for generating the COG mosaic. "
    "Passed to `gdal_translate -co COMPRESS=...`. "
    "Supports DEFLATE/ZSTD/LERC_DEFLATE/LERC_ZSTD/LZMA.",
)
@click.option(
    "--overview_resampling",
    type=str,
    default="NEAREST",
    help="The resampling method used for generating the overviews COG mosaic. "
    "Passed to `gdal_translate -co OVERVIEW_RESAMPLING=...`. "
    "Supports NEAREST, BILINEAR, CUBIC, CUBICSPLINE, LANCZOS, AVERAGE, RMS, MODE",
)
@click.option(
    "--level",
    type=int,
    default=9,
    help="The compression level to use. A lower number will result "
    "in faster compression but less efficient compression rate. "
    "For DEFLATE/LZMA, 9 is the slowest/higher compression rate. "
    "For ZSTD, 22 is the slowest/higher compression rate.",
)
@click.option(
    "--overview_count",
    type=int,
    default=8,
    help="The number of COG overviews to generate. Higher nunbers "
    "will improve data streaming and performance (particularly at "
    "continental scale), but take longer to generate.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    is_flag=True,
    default=True,
    help="Whether to sign AWS requests for S3 access. Defaults to "
    "True; can be set to False by passing `--no-aws_unsigned`.",
)
def make_mosaic_cli(
    product,
    band,
    year,
    version,
    product_dir,
    output_dir,
    dataset_maturity,
    compress,
    overview_resampling,
    level,
    overview_count,
    aws_unsigned,
):
    # Set up logs
    input_params = locals()
    run_id = f"[{product}] [{version}] [{year}] [{band}]"
    log = configure_logging(run_id)
    log.info(f"{run_id}: Using parameters {input_params}")

    # Determine if input data is located on S3
    s3 = _is_s3(product_dir)

    # Clean string to prepare for analysis
    product_dir = product_dir.replace("s3://", "")
    product_dir = product_dir.rstrip("/")
    product_dir = f"{product_dir}/{product}/{version}"
    log.info(f"{run_id}: Using input data product directory: {product_dir}")

    # Determine output directory
    output_dir = output_dir.rstrip("/")
    output_dir = f"{output_dir}/{product}/{version}/continental_mosaics/{year}--P1Y"

    # Depending on whether input data is on S3, create a list of data to mosaic
    if s3:
        # Determine what files are available on S3
        log.info(f"{run_id}: Identifying input data from S3 bucket")
        fs = s3fs.S3FileSystem(anon=True)
        configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)
        cogs = fs.glob(f"{product_dir}/**/**/{year}--P1Y/{product}_*{year}--P1Y_{dataset_maturity}_{band}.tif")
    else:
        # Determine what files are available on the local file system
        log.info(f"{run_id}: Identifying input data from local file system")
        cogs = glob.glob(
            f"{product_dir}/**/{year}--P1Y/{product}_*{year}--P1Y_{dataset_maturity}_{band}.tif",
            recursive=True,
        )

    log.info(f"{run_id}: Number of COGs to mosaic: {len(cogs)}")
    if len(cogs) > 0:
        # Create a temporary directory to house files before syncing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_location = Path(temp_dir)
            log.info(f"{run_id}: Writing data to temporary folder: {temp_location}")

            # Output paths for intermediate files
            file_list_name = os.path.join(temp_location, f"{product}_mosaic_{year}_{band}.txt")
            vrt_name = os.path.join(temp_location, f"{product}_mosaic_{year}_{band}.vrt")
            output_name = os.path.join(temp_location, f"{product}_mosaic_{year}_{band}.tif")

            # Final output location
            output_file_path = os.path.join(output_dir, f"{product}_mosaic_{year}_{band}.tif")
            log.info(f"{run_id}: Output file path: {output_file_path}")

            # Write list of files to a temporary text file, so it can be
            # used as an input to `gdalbuildvrt`
            with open(file_list_name, "w") as f:
                for cog in cogs:
                    # Replace S3 bucket name in file paths to a /vsicurl/ path
                    # for `gdalbuildvrt` compatibility
                    cog = cog.replace(
                        "dea-public-data-dev/",
                        "/vsicurl/https://dea-public-data-dev.s3-ap-southeast-2.amazonaws.com/",
                    )
                    cog = cog.replace(
                        "dea-public-data/",
                        "/vsicurl/https://data.dea.ga.gov.au/",
                    )
                    f.write(f"{cog}\n")

            # Build virtual raster (VRT)
            log.info(f"{run_id}: Building virtual raster (VRT)")
            subprocess.run(
                ["gdalbuildvrt", vrt_name, "-input_file_list", file_list_name],
                check=True,
            )

            # Convert VRT to Cloud Optimized GeoTIFF (COG)
            log.info(f"{run_id}: Converting VRT to COG mosaic")
            subprocess.run(
                [
                    "gdal_translate",
                    vrt_name,
                    output_name,
                    "-co",
                    "NUM_THREADS=ALL_CPUS",  # Parallelisation
                    "-of",
                    "COG",  # Output format
                    "-co",
                    "BIGTIFF=YES",  # Allow large TIFFs
                    "-co",
                    "BLOCKSIZE=1024",  # Tiling
                    "-co",
                    "OVERVIEWS=IGNORE_EXISTING",  # Force overview regen
                    "-co",
                    f"OVERVIEW_RESAMPLING={overview_resampling}",  # Resampling for overviews
                    "-co",
                    f"OVERVIEW_COUNT={overview_count}",  # Number of overviews
                    "-co",
                    f"COMPRESS={compress}",  # Compression
                    "-co",
                    f"LEVEL={level}",  # Compression level
                    "-co",
                    "PREDICTOR=YES",  # Compression predictor
                ],
                check=True,
            )

            # Copy output to S3
            if _is_s3(output_file_path):
                log.info(f"{run_id}: Writing COG to S3: {output_file_path}")
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--only-show-errors",
                        "--acl",
                        "bucket-owner-full-control",
                        str(output_name),
                        str(output_file_path),
                    ],
                    check=True,
                )

            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Copy from tempfile to output location
                log.info(f"{run_id}: Writing data locally: {output_file_path}")
                shutil.copy(output_name, output_file_path)

    else:
        log.info(f"{run_id}: No COG inputs found")


if __name__ == "__main__":
    make_mosaic_cli()
