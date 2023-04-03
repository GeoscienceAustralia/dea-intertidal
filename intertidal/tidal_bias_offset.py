import xarray as xr
import numpy as np

def bias_offset(tide_m = ds.tide_m,
                tide_cq = ds.tide_cq,
                extents = ds.extents,
                lat_hat=True,
                lot_hot=None
                ):
    """
    Calculate the pixel-based sensor-observed spread and high/low offsets in tide heights compared to
    the full modelled tide range. Optionally, also return the highest and lowest astronomical and 
    sensor-observed tides for each pixel.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray of including sensor-observed tide-heights (ds.tide_m).
    dem : xarray.Array
        An xarray of `ds` containing modelled tide heights, separated by quantiles of the full modelled tide range
        (tide_cq) as well as the NIDEM elevations (dem.tide_m). Note that this version of dem is an 
        output of the pixel_exp function in exposure.py.
    set_dtype : dtype or None, optional
        Default is None. Set a dtype other than float, e.g., int. Note that changing the dtype may
        result in loss of data/resolution in your results.
    lat_hat : bool, optional
        Default is True. Lowest/highest astronomical tides. This work considers the modelled tides
        to be equivalent to the astronomical tides.
    lot_hot : bool, optional
        Default is None. Lowest/highest sensor-observed tides.

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
    max_obs = tide_m.max(dim='time')
    min_obs = tide_m.min(dim='time')
    max_mod = tide_cq.max(dim='quantile')
    min_mod = tide_cq.min(dim='quantile')
    
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
        
    ## Add the lowest and highest astronomical tides
    if lat_hat:
        lat = min_mod.where(ds.extents != 2)
        hat = max_mod.where(ds.extents != 2)
    
    ## Add the lowest and highest sensor-observed tides
    if lot_hot:
        lot = min_obs.where(ds.extents != 2)
        hot = max_obs.where(ds.extents != 2)
        
    # Mask out non-intertidal pixels using ds extents
    spread = spread.where(ds.extents != 2)
    ht_offset = ht_offset.where(ds.extents != 2)
    lt_offset = lt_offset.where(ds.extents != 2)
    
    ## Convert floats to ints
    spread = spread.astype(np.int16)
    ht_offset = ht_offset.astype(np.int16)
    lt_offset = lt_offset.astype(np.int16)
    
    if lat_hat:
        if lot_hot:
            return lat, hat, lot, hot, spread, lt_offset, ht_offset
        else:
            return lat, hat, spread, lt_offset, ht_offset
    elif lot_hot:
        return lot, hot, spread, lt_offset, ht_offset
    else:
        return spread, lt_offset, ht_offset