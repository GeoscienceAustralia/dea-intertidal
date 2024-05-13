# DEA Intertidal Open Data Cube metadata

This directory contains metadata files required for indexing DEA Intertidal into the Open Data Cube (ODC), including:
* An [ODC Product Definition YAML](https://datacube-core.readthedocs.io/en/latest/installation/product-definitions.html) describing the DEA Intertidal product and its bands/measurements
* An [ODC Metadata Type YAML](https://datacube-core.readthedocs.io/en/latest/installation/metadata-types.html) defining custom searchable metadata fields for DEA Intertidal

Individual [ODC Dataset Documents](https://datacube-core.readthedocs.io/en/latest/installation/dataset-documents.html) and Spatiotemporal Asset Catalogue (STAC) metadata are generated during the product generation workflow using the [`intertidal.io.export_dataset_metadata`](https://github.com/GeoscienceAustralia/dea-intertidal/blob/main/intertidal/io.py#L877-L1091) function.

All three metadata files can be validated using the `eo3-validate` command from `eodatasets`:
```
!eo3-validate \
metadata/ga_s2ls_intertidal_cyear_3.odc-product.yaml \
metadata/eo3_intertidal.odc-type.yaml \
data/processed/ga_s2ls_intertidal_cyear_3/0-0-1/tes/ting/2023--P1Y/ga_s2ls_intertidal_cyear_3_testing_2023--P1Y_interim.odc-metadata.yaml \
--thorough
```
