![Digital Earth Australia Intertidal](https://raw.githubusercontent.com/GeoscienceAustralia/dea-notebooks/refs/heads/develop/Supplementary_data/dea_logo_wide.jpg)

# Digital Earth Australia Intertidal

[![PyPI](https://img.shields.io/pypi/v/dea-intertidal)](https://pypi.org/project/dea-intertidal/)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.ecss.2019.03.006-0e7fbf.svg)](https://doi.org/10.1016/j.ecss.2019.03.006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/GeoscienceAustralia/dea-intertidal/branch/main/graph/badge.svg?token=7HXSIPGT5I)](https://codecov.io/gh/GeoscienceAustralia/dea-intertidal)
[![example workflow](https://github.com/GeoscienceAustralia/dea-intertidal/actions/workflows/dea-intertidal-image.yml/badge.svg)](https://github.com/GeoscienceAustralia/dea-intertidal/actions/workflows/dea-intertidal-image.yml)

### Remote sensing tools for high-resolution mapping of the intertidal zone 🌊🛰️ 

**DEA Intertidal** combines satellite Earth observation data with advanced tide modelling to produce high-resolution maps of intertidal elevation, extents and exposure. These mapping datasets support applications from coastal hazard assessment to sediment dynamics, habitat mapping, and migratory species research.

In Australia, the package underpins the [DEA Intertidal product suite](https://knowledge.dea.ga.gov.au/data/product/dea-intertidal/), which provides continental-scale 10 m datasets of Australia's exposed intertidal zone from 2016 onwards.

**Key functionality:**

* 🛰️ **Global applicability** – integrate open satellite data with global tidal models
* ⛰️ **Elevation modelling** – pixel-based 3D intertidal elevation with quantified uncertainty
* ⏱️ **Exposure analysis** – spatio-temporal patterns of inundation and exposure
* 🗺️ **Extents classification** – categorical mapping of land, intertidal, inland waters, and ocean
* 🌊 **Tidal metrics** – per-pixel tidal ranges, offsets, and satellite sampling biases
  
---

## ⚙️ Installation

You can install `dea-intertidal` from PyPI with `pip` (https://pypi.org/project/dea-intertidal/). 
By default `dea-intertidal` will be installed with minimal dependencies which excludes `datacube`:

```console
pip install dea-intertidal
```

To install with additional `datacube` dependencies:

```console
pip install dea-intertidal[datacube]
```

Functions can then be imported in Python:

```python
from intertidal.elevation import elevation
```

## 🚀 Getting started

We recommend running the [**Getting started with DEA Intertidal**](notebooks/Getting_started_with_DEA_Intertidal.ipynb) Jupyter Notebook for an example of how to run a simple DEA Intertidal analysis.

This notebook loads data from Microsoft Planetary Computer using using Spatio-Temporal Asset Catalogue (STAC) metadata, and is suitable for any coastal location globally.

## 📁 Repository structure

The DEA Intertidal Github repository contains the following important sections:
* [`intertidal`](intertidal/): The DEA Intertidal Python package, containing modules required for loading data, tide modelling, intertidal elevation, and exposure calculations
* [`notebooks`](notebooks): Jupyter Notebooks providing workflows for generating key DEA Intertidal outputs
* [`data`](data): Contains required `raw` input data files and output `interim` and `processed` outputs
* [`metadata`](metadata): Open Data Cube (ODC) metadata required for indexing DEA Intertidal into an ODC datacube
* [`tests`](tests): Unit and integration tests, including automatically generated validation test results

## 🛠️ Contact
For assistance with any of the Python code or Jupyter Notebooks in this repository, please post a [Github issue](https://github.com/GeoscienceAustralia/dea-intertidal/issues). For questions or more information about DEA Intertidal, email earth.observation@ga.gov.au.

## 📝 Citation 
```
Bishop-Taylor, R., Phillips, C., Newey, V., Sagar, S. (2024). Digital Earth Australia Intertidal. Geoscience Australia, Canberra. https://dx.doi.org/10.26186/149403

Bishop-Taylor, R., Sagar, S., Lymburner, L., Beaman, R.L., 2019. Between the tides: modelling the elevation of Australia's exposed intertidal zone at continental scale. Estuarine, Coastal and Shelf Science. https://doi.org/10.1016/j.ecss.2019.03.006
```
