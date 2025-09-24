![Digital Earth Australia Intertidal](https://raw.githubusercontent.com/GeoscienceAustralia/dea-notebooks/refs/heads/develop/Supplementary_data/dea_logo_wide.jpg)

# Digital Earth Australia Intertidal

[![PyPI](https://img.shields.io/pypi/v/dea-intertidal)](https://pypi.org/project/dea-intertidal/)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.ecss.2019.03.006-0e7fbf.svg)](https://doi.org/10.1016/j.ecss.2019.03.006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/GeoscienceAustralia/dea-intertidal/branch/main/graph/badge.svg?token=7HXSIPGT5I)](https://codecov.io/gh/GeoscienceAustralia/dea-intertidal)
[![example workflow](https://github.com/GeoscienceAustralia/dea-intertidal/actions/workflows/dea-intertidal-image.yml/badge.svg)](https://github.com/GeoscienceAustralia/dea-intertidal/actions/workflows/dea-intertidal-image.yml)

**License:** The code in this repository is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). Digital Earth Australia data is licensed under the [Creative Commons by Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

**Contact:** For assistance with any of the Python code or Jupyter Notebooks in this repository, please post a [Github issue](https://github.com/GeoscienceAustralia/dea-intertidal/issues). For questions or more information about DEA Intertidal, email earth.observation@ga.gov.au.

**To cite:** 
> Bishop-Taylor, R., Sagar, S., Lymburner, L., Beaman, R.L., 2019. Between the tides: modelling the elevation of Australia's exposed intertidal zone at continental scale. Estuarine, Coastal and Shelf Science. https://doi.org/10.1016/j.ecss.2019.03.006

> Sagar, S., Phillips, C., Bala, B., Roberts, D., Lymburner, L., 2018. Generating continental scale pixel-based surface reflectance composites in coastal regions with the use of a multi-resolution tidal model. Remote Sensing. 10, 480. https://doi.org/10.3390/rs10030480

> Sagar, S., Roberts, D., Bala, B., Lymburner, L., 2017. Extracting the intertidal extent and topography of the Australian coastline from a 28 year time series of Landsat observations. Remote Sensing of Environment 195, 153-169. https://doi.org/10.1016/j.rse.2017.04.009

---

The DEA Intertidal product suite maps the changing elevation, exposure and tidal characteristics of Australia's exposed intertidal zone, the complex zone that defines the interface between land and sea. It is the next generation of DEA's intertidal products that have been used across government and industry to help better characterise and understand this complex zone that defines the interface between land and sea.

Incorporating both Sentinel-2 and Landsat data, the product suite provides an annual 10 m resolution elevation product for the intertidal zone, enabling users to better monitor and understand some of the most dynamic regions of Australia's coastlines. Utilising an improved tidal modelling capability, the product suite includes a continental scale mapping of intertidal exposure over time, enabling scientists and managers to integrate the data into ecological and migratory species applications and modelling.

## Installation

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

## Repository structure

The DEA Intertidal Github repository contains the following important sections:
* `intertidal`: The DEA Intertidal Python package, containing modules required for loading data, tide modelling, intertidal elevation, and exposure calculations
* `notebooks`: Jupyter Notebooks providing workflows for generating key DEA Intertidal outputs. Importantly:
  * `notebooks/Intertidal_CLI.ipynb`: For running the entire DEA Intertidal workflow via the Command Line Interface
  * `notebooks/Intertidal_workflow.ipynb`: For running the entire DEA Intertidal workflow via interactive notebook cells
  * `notebooks/Intertidal_elevation.ipynb`: For customising and running the DEA Intertidal Elevation portion of the workflow
  * `notebooks/Intertidal_elevation_stac.ipynb`: For running DEA Intertidal Elevation on global satellite data loaded from Microsoft Planetary Computer using STAC metadata
* `data`: Contains required `raw` input data files and output `interim` and `processed` outputs
* `metadata`: Open Data Cube (ODC) metadata required for indexing DEA Intertidal into an ODC datacube
* `tests`: Unit and integration tests, including automatically generated validation test results
