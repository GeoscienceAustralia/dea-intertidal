#!/usr/bin/env python3

import os
from setuptools import find_packages, setup

# What packages are required for this module to be executed?
REQUIRED = [
    "aiohttp",
    "awscli",
    "botocore",
    "Bottleneck",
    "click",
    "dask",
    "datacube[s3,performance]",
    "dea_tools>=0.3.6",
    "eodatasets3",
    "eo-tides>=0.6.3",
    "hdstats",
    "geopandas",
    "matplotlib",
    "mdutils",
    "numpy",
    "odc-algo",
    "odc-geo",
    "odc-ui",
    "pandas",
    "pyogrio",
    "pyproj",
    "pystac",
    "pytest",
    "pytest-dependency",
    "pytest-cov",
    "pytz",
    "rasterio",
    "rioxarray",
    "s3fs",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "sunriset",
    "shapely",
    "tqdm",
    "xarray",
]

# Optional dependencies
EXTRAS = {
    "postgres": ["psycopg2"],
}

# Package metadata
NAME = "dea_intertidal"
DESCRIPTION = "Tools for running Digital Earth Australia Intertidal"
URL = "https://github.com/GeoscienceAustralia/dea-intertidal"
EMAIL = "earth.observation@ga.gov.au"
AUTHOR = "Geoscience Australia"
REQUIRES_PYTHON = ">=3.10.0"

# Setup kwargs
setup_kwargs = {
    "name": NAME,
    "description": DESCRIPTION,
    "long_description": DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "author": AUTHOR,
    "author_email": EMAIL,
    "python_requires": REQUIRES_PYTHON,
    "url": URL,
    "install_requires": REQUIRED,
    "extras_require": EXTRAS,
    "packages": find_packages(),
    "include_package_data": True,
    "license": "Apache License 2.0",
    "entry_points": {
        "console_scripts": [
            "dea-intertidal = intertidal.elevation:intertidal_cli",
            "dea-tidal-composites = intertidal.composites:tidal_composites_cli",
            "dea-mosaics = intertidal.mosaics:make_mosaic_cli",
        ]
    },
}

setup(**setup_kwargs)
