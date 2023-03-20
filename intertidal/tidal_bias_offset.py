import xarray as xr

def bias_offset(
                ds,
                dem,
                set_dtype = None,
                LAT_HAT=True,
                LOT_HOT=False
                ):
    """
    Calculate the pixel-based sensor-observed spread and high/low offsets in tide heights compared to
    the full modelled tide range. Optionally, also return the highest and lowest astronomical and 
    sensor-observed tides for each pixel.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray of including sensor-observed tide-heights (ds.tide_m).
    dem : xarray.Dataset
        An xarray of modelled tide heights, separated by quantiles of the full modelled tide range
        (tide_cq) as well as the NIDEM elevations (dem.tide_m). Note that this version of dem is an 
        output of the pixel_exp function in exposure.py.
    set_dtype : dtype or None, optional
        Default is None. Set a dtype other than float, e.g., int. Note that changing the dtype may
        result in loss of data/resolution in your results.
    LAT_HAT : bool, optional
        Default is True. Lowest/highest astronomical tides. This work considers the modelled tides
        to be equivalent to the astronomical tides.
    LOT_HOT : bool, optional
        Default is False. Lowest/highest sensor-observed tides.

    Returns
    -------
    xarray.Dataset
        The output dataset containing arrays of the spread, ht_offset, lt_offset and optionally,
        the highest and lowest astronomical tides and the highest and lowest sensor-observed tides.
    
    Notes
    -----
    Spread measures the sensor observed range of tide heights as a percentage of the full
    modelled tidal range.
    The high-tide offset, ht_offset, measures the offset of the sensor-observed highest tide 
    from the maximum modelled tide.
    The low-tide offset, lt_offset, measures the offset of the sensor-observed lowest tide 
    from the minimum modelled tide.   
    """
    
    # Set the maximum and minimum values per pixel for the observed and modelled datasets
    max_obs = ds.tide_m.max(dim='time')
    min_obs = ds.tide_m.min(dim='time')
    max_mod = dem.tide_cq.max(dim='quantile')
    min_mod = dem.tide_cq.min(dim='quantile')
    
    # Set the maximum range in the modelled and observed tide heights
    mod_range = max_mod - min_mod
    obs_range = max_obs - min_obs

    # Calculate the spread of the observed tide heights as a percentage of the modelled tide
    # heights
    spread = obs_range/mod_range * 100
    
    # Calculate the high and low tide offset of the observed tide heights as a percentage
    # of the modelled highest and lowest tides.
    ht_offset = (abs(max_mod - max_obs))/mod_range * 100
    lt_offset = (abs(min_mod - min_obs))/mod_range * 100
    
    # Change the dtype of the output arrays
    if set_dtype is not None:
        spread = spread.astype(set_dtype)
        ht_offset = ht_offset.astype(set_dtype)
        lt_offset = lt_offset.astype(set_dtype)
    
    ## Add the lowest and highest astronomical tides
    ## TODO: mask using intertidal extent layer instead
    if LAT_HAT is True:
        ds['LAT'] = min_mod.where(ds.tide_m.min(dim='time')> -9999)
        ds['HAT'] = max_mod.where(ds.tide_m.min(dim='time')> -9999)
    
    ## Add the lowest and highest sensor-observed tides
    ## TODO: mask using intertidal extent layer instead
    if LOT_HOT is True:
        ds['LOT'] = min_obs.where(ds.tide_m.min(dim='time')> -9999)
        ds['HOT'] = max_obs.where(ds.tide_m.min(dim='time')> -9999)
        
    # Mask out non-intertidal pixels using ds extents
    ## TODO: mask using intertidal extent layer instead
    ds['spread'] = spread.where(ds.tide_m.min(dim='time') > -9999)
    ds['ht_offset'] = ht_offset.where(ds.tide_m.min(dim='time') > -9999)
    ds['lt_offset'] = lt_offset.where(ds.tide_m.min(dim='time')> -9999)
    
    return ds