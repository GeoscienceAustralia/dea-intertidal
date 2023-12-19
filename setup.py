#!/usr/bin/env python3

import os
from setuptools import find_packages, setup

# Where are we?
IS_SANDBOX = "sandbox" in os.getenv("JUPYTER_IMAGE", default="")

tests_require = [
    "pytest",
    "pytest-dependency",
    "pytest-cov",
]

extras = {
    "test": tests_require,
}

# What packages are required for this module to be executed?
REQUIRED = [
    "aiohttp",
    "affine",
    "botocore",
    "click",
    "datacube",
    "dea_tools",
    "fiona",
    "geopandas",
    "matplotlib",
    "numpy",
    "odc-geo", 
    "odc-ui",
    "odc-algo",
    "pandas",
    "pygeos",
    "pyproj",
    "pyTMD>=2.0.0",
    "pytz",
    "rasterio",
    "setuptools-scm",
    "seaborn",
    "sunriset",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "shapely",
    "tqdm",
    "xarray",
    "xskillscore",
]

# Package metadata
NAME = "dea_intertidal"
DESCRIPTION = "Tools for running Digital Earth Australia Intertidal"
URL = "https://github.com/GeoscienceAustralia/dea-intertidal"
EMAIL = "earth.observation@ga.gov.au"
AUTHOR = "Geoscience Australia"
REQUIRES_PYTHON = ">=3.8.0"

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
    "install_requires": REQUIRED if not IS_SANDBOX else [],
    "tests_require": tests_require,
    "extras_require": extras,
    "packages": find_packages(),
    "include_package_data": True,
    "license": "Apache License 2.0",
    "entry_points": {
        "console_scripts": [
            "dea-intertidal = intertidal.elevation:intertidal_cli",
            "dea-intertidal-hltc = intertidal.hltc:hltc_cli",
        ]
    },
}

setup(**setup_kwargs)
