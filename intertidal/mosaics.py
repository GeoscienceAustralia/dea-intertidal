import s3fs
import shutil
import os
import glob
import sys
from pathlib import Path
import tempfile
import subprocess
import pandas as pd
import geopandas as gpd
from pathlib import Path
from urllib.parse import urlparse
import click

from datacube.utils.aws import configure_s3_access
from intertidal.utils import (
    configure_logging,
    round_date_strings,
)
from pathlib import Path
import os

def _is_s3(path):
    """
    Determine whether output location is on S3.
    """
    uu = urlparse(path)
    return uu.scheme == "s3"


@click.command()
@click.option(
    "--product",
    type=str,
    required=True,
    help="The name of the product being mosaiced "
    "eg ga_s2_tidal_composites_cyear_3 or ga_s2ls_intertidal_cyear_3. ",
)
@click.option(
    "--band",
    type=str,
    required=True,
    help="The name of the band to be mosaiced. eg elevation or exposure ",
)
@click.option(
    "--year",
    type=str,
    required=True,
    help="The year of the band to be mosaiced eg 2022 ",
)
@click.option(
    "--version",
    type=str,
    required=True,
    help="The version number of the product to be mosaiced (e.g. " "'0-0-1').",
)
@click.option(
    "--product_dir",
    type=str,
    default="s3://dea-public-data-dev/derivative/",
    help="The directory/location to read the tile cogs from; supports "
    "both local disk and S3 locations. "
    "Defaults to 's3://dea-public-data-dev/derivative/'.",
)
@click.option(
    "--output_dir",
    type=str,
    default="/gdata1/projects/coastal/intertidal/mosaics/",
    help="The directory/location to output data and metadata; supports "
    "both local disk and S3 locations. Defaults to '/gdata1/projects/coastal/intertidal/mosaics/'. "
    "The function will add on {product}/{version}/continental_mosaics/{year}--P1Y to the provided ouput_dir ",
)
@click.option(
    "--dataset_maturity",
    type=str,
    default="final",
    help="Dataset maturity metadata to used in the name of the input datasets and output dataset. "
    "Defaults to 'final', can also be 'interim'.",
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
    aws_unsigned,
):
    input_params = locals()
    run_id = f"[{version}] [{year}] [{band}]"
    log = configure_logging(run_id)

    # Record params in logs
    log.info(f"{run_id}: Using parameters {input_params}")

    s3 = _is_s3(product_dir)
    product_dir = product_dir.replace("s3://", "")
    product_dir = product_dir.rstrip("/")
    product_dir = f"{product_dir}/{product}/{version}"

    log.info(f"{run_id}: Using product_dir {product_dir}")
    output_dir = output_dir.rstrip("/")
    output_dir = f"{output_dir}/{product}/{version}/continental_mosaics/{year}--P1Y"

    run_id = f"[{product}] [{band}] [{year}]"
    log = configure_logging(run_id)

    fs = s3fs.S3FileSystem(anon=True)
    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)
    print(
        f"{product_dir}/**/{year}--P1Y/{product}_*{year}--P1Y_{dataset_maturity}_{band}.tif"
    )

    if s3:
        cogs = fs.glob(
            f"{product_dir}/**/**/{year}--P1Y/{product}_*{year}--P1Y_{dataset_maturity}_{band}.tif"
        )
    else:
        cogs = glob.glob(
            f"{product_dir}/**/{year}--P1Y/{product}_*{year}--P1Y_{dataset_maturity}_{band}.tif",
            recursive=True,
        )

    log.info(f"{run_id}: number of cogs to mosaic {len(cogs)}")
    if len(cogs) > 0:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_location = Path(temp_dir)
            log.info(f"{run_id}: writing to {temp_location}")
            file_list_name = os.path.join(
                temp_location, f"{product}_{year}_{band}.txt"
            )
            vrt_name = os.path.join(
                temp_location, f"{product}_{year}_{band}.vrt"
            )
            output_name = os.path.join(
                temp_location, f"{product}_{year}_{band}.tif"
            )
            output_file_path = os.path.join(
                output_dir, f"{product}_{year}_{band}.tif"
            )
            log.info(f"{run_id}: Generating file {output_name}")
            log.info(f"{run_id}: output_file_path {output_file_path}")

            with open(file_list_name, "w") as f:
                for cog in cogs:
                    cog = cog.replace(
                        "dea-public-data-dev/",
                        "/vsicurl/https://dea-public-data-dev.s3-ap-southeast-2.amazonaws.com/",
                    )
                    cog = cog.replace(
                        "dea-public-data/",
                        "/vsicurl/https://data.dea.ga.gov.au/",
                    )
                    f.write(f"{cog}\n")
            os.system(f"gdalbuildvrt {vrt_name} -input_file_list {file_list_name}")
            os.system(
                f"gdal_translate {vrt_name} {output_name} -co NUM_THREADS=ALL_CPUS -of COG -co BIGTIFF=YES -co COMPRESS=DEFLATE -co LEVEL=9 -co PREDICTOR=YES -co BLOCKSIZE=1024"
            )
            # Either sync to S3 or copy files to local destination
            if _is_s3(output_file_path):
                s3_command = [
                    "aws",
                    "s3",
                    "cp",
                    "--only-show-errors",
                    "--acl bucket-owner-full-control",
                    str(output_name),
                    str(output_file_path),
                ]

                log.info(f"{run_id}: Writing to S3: {output_file_path}")
                subprocess.run(" ".join(s3_command), shell=True, check=True)

            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Copy from tempfile to output location
                log.info(f"{run_id}: Writing data locally: {output_file_path}")
                shutil.copy(output_name, output_file_path)

    else:
        log.info(f"No rasters found for {band} in {year}")


if __name__ == "__main__":
    make_mosaic_cli()
