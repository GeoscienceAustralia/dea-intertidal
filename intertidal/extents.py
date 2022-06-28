def load_data(dc, geom, time_range=("2019", "2021"), resolution=10, crs="epsg:32753"):

    from datacube.virtual import catalog_from_file
    from datacube.utils.masking import mask_invalid_data
    from datacube.utils.geometry import GeoBox, Geometry
    from odc.algo import mask_cleanup
    import xarray as xr

    # Load in virtual product catalogue and select MNDWI product
    catalog = catalog_from_file("configs/dea_virtual_product_landsat_s2.yaml")

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

    # Load Sentinel-2 data
    product = catalog["s2_nbart_ndwi"]
    s2_ds = product.load(dc, **query_params)

    # Apply cloud mask and contiguity mask
    s2_ds_masked = s2_ds.where(s2_ds.cloud_mask == 1 & s2_ds.contiguity)[["ndwi"]]

    # Load Landsat data
    product = catalog["ls_nbart_ndwi"]
    ls_ds = product.load(dc, **query_params)

    # Clean cloud mask by applying morphological closing to all
    # valid (non cloud, shadow or nodata) pixels. This removes
    # long, narrow features like false positives over bright beaches.
    a = ls_ds.cloud_mask.isin([1, 4, 5])
    good_data_cleaned = mask_cleanup(
        mask=a,
        mask_filters=[("closing", 5)],
    )

    # Dilate cloud and shadow. To ensure that nodata areas (e.g.
    # Landsat 7 SLC off stripes) are not also dilated, only dilate
    # mask pixels (e.g. values 0 in `good_data_cleaned`) that are
    # outside of the original nodata pixels (e.g. not 0 in
    # `ls_ds.cloud_mask`)
    b = (good_data_cleaned == 0) & (ls_ds.cloud_mask != 0)
    good_data_mask = mask_cleanup(
        mask=b,
        mask_filters=[("dilation", 5)],
    )

    # Apply cloud mask and contiguity mask
    ls_ds_masked = ls_ds.where(~good_data_mask & ls_ds.contiguity)[["ndwi"]]

    # Combine into a single ds
    ds = xr.concat([s2_ds_masked, ls_ds_masked], dim="time").sortby("time")
    return ds


