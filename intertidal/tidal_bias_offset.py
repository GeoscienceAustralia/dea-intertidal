def bias_offset(tide_m, tide_cq, lat_hat=True, lot_hot=None):
    """Calculate the pixel-based sensor-observed spread and high/low
    offsets in tide heights compared to the full modelled tide range.
    Optionally, also return the highest and lowest astronomical and
    sensor-observed tides for each pixel.

    TODO: update to use `eo-tides.stats` functionality

    Parameters
    ----------
    tide_m : xr.DataArray
        An xarray.DataArray representing sensor observed tide heights
        for each pixel. Should have 'time', 'x' and 'y' in its
        dimensions.
    tide_cq : xr.DataArray
        An xarray.DataArray representing modelled tidal heights for
        each pixel. Should have 'quantile', 'x' and 'y' in its
        dimensions.
    lat_hat : bool, optional
        Lowest/highest astronomical tides. This work considers the
        modelled tides to be equivalent to the astronomical tides.
        Default is True.
    lot_hot : bool, optional
        Lowest/highest sensor-observed tides. Default is None.

    Returns
    -------
    Depending on the values of `lat_hat` and `lot_hot`, returns a tuple
    with some or all of the following as xarray.DataArrays:
        * `lat`: The lowest astronomical tide.
        * `hat`: The highest astronomical tide.
        * `lot`: The lowest sensor-observed tide.
        * `hot`: The highest sensor-observed tide.
        * `spread`: The spread of the observed tide heights as a
        percentage of the modelled tide heights.
        * `offset_lowtide`: The low tide offset measures the offset of the
        sensor-observed lowest tide from the minimum modelled tide.
        * `offset_hightide`: The high tide measures the offset of the
        sensor-observed highest tide from the maximum modelled tide.

    """
    # Set the maximum and minimum values per pixel for the observed and
    # modelled datasets
    max_obs = tide_m.max(dim="time")
    min_obs = tide_m.min(dim="time")
    max_mod = tide_cq.max(dim="quantile")
    min_mod = tide_cq.min(dim="quantile")

    # Set the maximum range in the modelled and observed tide heights
    mod_range = max_mod - min_mod
    obs_range = max_obs - min_obs

    # Calculate the spread of the observed tide heights as a percentage
    # of the modelled tide heights
    spread = obs_range / mod_range * 100

    # Calculate the high and low tide offset of the observed tide
    # heights as a percentage of the modelled highest and lowest tides.
    offset_hightide = (abs(max_mod - max_obs)) / mod_range * 100
    offset_lowtide = (abs(min_mod - min_obs)) / mod_range * 100

    # Add the lowest and highest astronomical tides
    if lat_hat:
        lat = min_mod
        hat = max_mod

    # Add the lowest and highest sensor-observed tides
    if lot_hot:
        lot = min_obs
        hot = max_obs

    if lat_hat:
        if lot_hot:
            return lat, hat, lot, hot, spread, offset_lowtide, offset_hightide
        return lat, hat, spread, offset_lowtide, offset_hightide
    if lot_hot:
        return lot, hot, spread, offset_lowtide, offset_hightide
    return spread, offset_lowtide, offset_hightide
