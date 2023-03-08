import xarray as xr

def bias_offset(
                ds,
                dem,
                set_dtype = None
                ):
    '''
    
    Calculate the pixel-based sensor-observed spread and offset in tide heights
    compared to the full modelled tide range.
    ds.tide_m is an xarray of sensor-observed tide-heights calculated for the imagery 
    in your workflow
    dem contains an xarray of modelled tide heights, separated by quantiles of the full
    modelled tide range (tide_cq) as well as the NIDEM elevations (tide_m). NOTE: dem
    is an output of the pixel_exp function in exposure.py
    Returns the spread, ht_offset and lt_offset.
    Spread measures the sensor observed range of tide heights as a percentage of the full
    modelled tidal range.
    The high-tide offset, ht_offset, measures the offset of the sensor-observed highest tide 
    from the maximum modelled tide.
    The low-tide offset, lt_offset, measures the offset of the sensor-observed lowest tide 
    from the minimum modelled tide.
    set_dtype: default = None. Set a dtype other than float e.g. int. Note that changing the 
    dtype may result in loss of data/resolution in your results
    
    '''
    
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
    
    # # Mask out non-intertidal pixels using NIDEM extents
    # spread = spread.where(dem.tide_m > -9999)
    # ht_offset = ht_offset.where(dem.tide_m > -9999)
    # lt_offset = lt_offset.where(dem.tide_m > -9999)
    
        # TEMP Mask out non-intertidal pixels using ds extents
    spread = spread.where(ds.tide_m.min(dim='time') > -9999)
    ht_offset = ht_offset.where(ds.tide_m.min(dim='time') > -9999)
    lt_offset = lt_offset.where(ds.tide_m.min(dim='time')> -9999)
    
    return spread, ht_offset, lt_offset