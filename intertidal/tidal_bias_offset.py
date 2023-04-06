import xarray as xr
import numpy as np

def bias_offset(
                tide_m,
                tide_cq,
                extents,
                lat_hat=True,
                lot_hot=None
                ):
    """
    Calculate the pixel-based sensor-observed spread and high/low offsets in tide heights compared to
    the full modelled tide range. Optionally, also return the highest and lowest astronomical and 
    sensor-observed tides for each pixel.

    Parameters
    ----------
    tide_m : xr.DataArray
        Xarray DataArray representing sensor observed tide heights for each pixel. Should have 'time', 
        'x' and 'y' in its dimensions.
    tide_cq : xr.DataArray
        Xarray DataArray representing modelled tidal heights for each pixel. Should have 'quantile', 
        'x' and 'y' in its dimensions.
    extents : xr.DataArray
        Xarray DataArray representing the always, sometimes and never wet extents of the intertidal 
        zone. Should have the same dimensions as `tide_m` and `tide_cq`.
    lat_hat : bool, optional
        Default is True. Lowest/highest astronomical tides. This work considers the modelled tides
        to be equivalent to the astronomical tides.
    lot_hot : bool, optional
        Default is None. Lowest/highest sensor-observed tides.

    Returns
    -------
    Depending on the values of `lat_hat` and `lot_hot`, returns a tuple with some or all of the 
    following as xarray.DataArrays:
        * `lat`: The lowest astronomical tide.
        * `hat`: The highest astronomical tide.
        * `lot`: The lowest sensor-observed tide.
        * `hot`: The highest sensor-observed tide.
        * `spread`: The spread of the observed tide heights as a percentage of the
          modelled tide heights.
        * `lt_offset`: The low tide offset measures the offset of the sensor-observed lowest 
          tide from the minimum modelled tide.
        * `ht_offset`: The high tide measures the offset of the sensor-observed highest tide
          from the maximum modelled tide.         
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
        lat = min_mod.where(extents != 2)
        hat = max_mod.where(extents != 2)
    
    ## Add the lowest and highest sensor-observed tides
    if lot_hot:
        lot = min_obs.where(extents != 2)
        hot = max_obs.where(extents != 2)
        
    # Mask out non-intertidal pixels using ds extents
    spread = spread.where(extents != 2)
    ht_offset = ht_offset.where(extents != 2)
    lt_offset = lt_offset.where(extents != 2)
    
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