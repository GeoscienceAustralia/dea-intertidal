

def extents(freq,
            dem,
            ds
           ):
    '''
    This extents code requires the 'freq' output from the 'ds_to_flat' func and the final 'dem'.
    The always/sometimes/never_wet layers are built from the NDWI frequency layer ('freq'), 
    generated to summarise the frequency of wetness per pixel for any given time-series of the 
    analysis area of interest.
    The always/sometimes/never_wet layers are calculated from the extents of the intertidal dem.
    ds is master xr.Dataset for storing intertidal outputs
    Always_wet areas are classified as 0.
    Sometimes_wet areas are classified as 1.
    Never_wet areas are classified as 2.
    '''

    ## Find the intertidal extent by masking `freq` with the non-null areas in the dem
    int_ext = freq.where(dem.notnull()==True)

    ## Find the non-intertidal extents by masking `freq` with the null areas in the dem.
    ## Create a bool for the always wet and always dry areas by separating the NDWI frequency
    ## values through the middle. (There's probably a nicer way to do this step).
    wet_dry_ext = freq.where(dem.isnull()==True)
    wet_dry_ext = wet_dry_ext >= 0.5

    ## Find the always_wet extent by masking the non-intertidal area for freq values greater than 0.5
    wet_ext = wet_dry_ext.where(wet_dry_ext == True, drop=True) ## If issues in the merge, try dropping the 'drop=True')

    ## Find the always_dry extent by masking the non-intertidal area for freq values lower than 0.5
    dry_ext = wet_dry_ext.where(wet_dry_ext == False, drop=True)

    ## Classify all non-nan areas as ints
    wet_ext = wet_ext.where(wet_ext.isnull(), 0)
    int_ext = int_ext.where(int_ext.isnull(), 1)
    dry_ext = dry_ext.where(dry_ext.isnull(), 2)

    # Combine into a single 'Extents' layer
    int_ext = int_ext.combine_first(wet_ext)
    int_ext = int_ext.combine_first(dry_ext)

    ## Add to master dataset
    ds['Extents'] = int_ext
    
    return ds