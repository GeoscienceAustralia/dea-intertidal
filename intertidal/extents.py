import xarray as xr
import numpy as np

import datacube
from skimage.measure import label, regionprops
from skimage.morphology import (binary_erosion, disk)

def intertidal_connection(ds, ocean_da, connectivity=1, dilation=None):
    """
    
    Identifies ocean by selecting regions of water that overlap
    with ocean pixels. This region can be optionally dilated to
    ensure that the sub-pixel algorithm has pixels on either side
    of the water index threshold.
    Parameters:
    -----------
    ds : xarray.DataArray
        An array containing True for land pixels, and False for water.
        This can be obtained by thresholding a water index
        array (e.g. MNDWI < 0).
    ocean_da : xarray.DataArray
        A supplementary static dataset used to separate ocean waters
        from other inland water. The array should contain values of 1
        for high certainty ocean pixels, and 0 for all other pixels
        (land, inland water etc). For Australia, we use the  Geodata
        100K coastline dataset, rasterized as the "geodata_coast_100k"
        product on the DEA datacube.
    connectivity : integer, optional
        An integer passed to the 'connectivity' parameter of the
        `skimage.measure.label` function.
    dilation : integer, optional
        The number of pixels to dilate ocean pixels to ensure than
        adequate land pixels are included for subpixel waterline
        extraction. Defaults to None.
    Returns:
    --------
    intertidal_connection : xarray.DataArray
        An array containing the a mask consisting of identified ocean
        pixels as True.
    """

    # First, break all time array into unique, discrete regions/blobs.
    # Fill NaN with 1 so it is treated as a background pixel
    blobs = xr.apply_ufunc(label, ds.fillna(1), 1, False, connectivity)

    # For each unique region/blob, use region properties to determine
    # whether it overlaps with a water feature from `water_mask`. If
    # it does, then it is considered to be directly connected with the
    # ocean; if not, then it is an inland waterbody.
    intertidal_connection = blobs.isin(
        [i.label for i in regionprops(blobs.values, ocean_da.values) if i.max_intensity]
    )

    # Dilate mask so that we include land pixels on the inland side
    # of each shoreline to ensure contour extraction accurately
    # seperates land and water spectra
    if dilation:
        intertidal_connection = xr.apply_ufunc(binary_dilation, intertidal_connection, disk(dilation))

    return intertidal_connection


def extents(freq,
           dem,
           corr,
           ):
    '''
    Classify coastal ecosystems into broad classes based 
    on their respective patterns of wetting frequency,
    relationship to tidal inundation and proximity to
    intertidal pixels.

    Parameters:
    -----------
    dem : xarray.DataArray
        An xarray.DataArray of the final intertidal DEM, generated 
        during the intertidal.elevation workflow
    freq : xarray.DataArray
        An xarray.DataArray of the NDWI frequency layer summarising the 
        frequency of wetness per pixel for any given time-series, 
        generated during the intertidal.elevation workflow
    corr : xarray.DataArray
        An xarray.DataArray of the correlation between pixel NDWI values
        and the tide-height, generated during the intertidal.elevation workflow

    Returns:
    --------
    extents: xarray.DataArray
        A binary xarray.DataArray depicting intertidal (0), tidal-wet (1),
        nontidal-wet (2), intermittently, non-tidal wet (3) and dry (4) coastal extents.
    Notes:
    ------
    Classes are defined as follows:
    '''
    # 0: Dry
    #     Pixels with wettness `freq` < 0.05
    #     Includes intermittently dry pixels with wetness frequency < 0.5 and > 0.05
    #     and `corr` to tide > 0.1 to capture intertidal pixels buffered
    #     out by the `corr` threshold of 0.2
    # 1: Intertidal
    #     Frequency of pixel wetness (`freq`) is > 0.01 and < 0.99
    #     The correlation (`corr`) between `freq` and tide-heights is > 0.2
    # 2: Wet tidal
    #     Frequency of pixel wetness (`freq`) is > 0.95
    #     Includes intermittently wet pixels with `freq` > 0.5 and < 0.95,
    #     and `corr` to tide > 0.1. This captures intertidal pixels buffered
    #     out by the `corr` threshold of 0.2 (default)
    #     Pixels are located offshore, within 10 pixels of known ocean, as defined
    #     by the Geodata 100k coastline dataset (`ocean_da`)
    # 3: Wet nontidal
    #     Frequency of pixel wetness (`freq`) is > 0.95
    #     Includes intermittently wet pixels with `freq` > 0.5 and < 0.95,
    #     and `corr` to tide > 0.1. This captures intertidal pixels buffered
    #     out by the `corr` threshold of 0.2 (default)
    #     Pixels are located onshore, more than 10 pixels from known ocean, as defined
    #     by the Geodata 100k coastline dataset (`ocean_da`)
    # 4: Intermittently wet nontidal
    #     Pixels with wetting `freq` between 0.95 and 0.05 and
    #     `corr` of `freq` to tide is < 0.1    
    
    ## Connect to datacube to load `ocean_da`
    dc = datacube.Datacube(app='ocean_masking')
 
    ## Set the upper and lower freq thresholds
    upper, lower = 0.99, 0.01
    
    '''--------------------------------------------------------------------'''
    ## Identify broad classes based on wetness frequency and tidal correlation
    dry = freq.where((freq < lower), drop=True)
    intermittent = freq.where((freq>=lower)&(freq<=upper),np.nan)
    wet = freq.where((freq>upper),np.nan)

    ##### Separate intermittent_tidal (intertidal)
    intertidal = freq.where(
                        (freq==intermittent)
                        &(corr>=0.15),
                        drop=True
                        )

    ##### Separate intermittent_nontidal
    intermittent_nontidal = freq.where(
                        (freq==intermittent)
                        &(corr<0.15),
                        drop=False
                        )
        
    ##### Separate high and low confidence intertidal pixels
    intertidal_hc = intertidal.where(dem.notnull(),drop=True)
    intertidal_lc = intertidal.where(dem.isnull(),drop=True)
    '''--------------------------------------------------------------------'''
    ##### Classify 'wet' pixels based on connectivity to intertidal pixels (into 'wet_ocean' and 'wet_inland')

    ## Create the 'always wet + intertidal' ds to compare against 'intertidal' pixels
    ## only for intertidal connectivity
    wet_intertidal = xr.where(freq>=lower,0,1)

    ## If deep-sea masked pixels, replace Nans with 'wet' boolean (0)
    if freq.isnull().any()==True is True:
        wet_intertidal = wet_intertidal.where(freq.notnull(), 0)

    ## Create a true/false layer of intertidal pixels (1) vs everything else (0)
    # # Extract intertidal pixels (value 1) then erode these by 1 pixels to ensure we only
    # # use high certainty intertidal regions for identifying connectivity to wet
    # # pixels in our satellite imagery.
    inter = freq.where((freq>=lower)&
                          (freq<=upper)&
                          (corr>=0.2))
    ## Convert to true/false
    inter = xr.where(freq==inter,True,False)
    ## Drop Nans
    if freq.isnull().any()==True is True:
        inter = inter.where(freq.notnull(), drop=True)
    ## Erode outer edge pixels by 1 pixel to drop extrema intertidal pixels and ensure connection 
    ## to high certainty intertidal pixels (POSSIBLY UNNECCESARY due to corr definition of intertidal pixels)
    inter = xr.apply_ufunc(binary_erosion, inter == 1, disk(1))

    ## Applying intertidal_connection masking function for the first of two times
    ## This first mask identifies where wet+intertidal pixels connect to intertidal pixels
    intertidal_mask1 = intertidal_connection(wet_intertidal, inter, connectivity=1)

    ## Prepare data to test for wet pixel connection to the connected 'wet and intertidal' mask.
    ## Identify and relabel the pixels in 'freq' that are 'wet (0)' and 'other (1)'.
    wet_bool = xr.where(freq==wet,False,True)
    ## If deep-sea masked pixels, replace Nans with 'wet' boolean (0)
    if freq.isnull().any()==True is True:
        wet_bool = wet_bool.where(freq.notnull(), 0)

    ## Applying intertidal_connection masking function for the second time
    intertidal_mask2 = intertidal_connection(wet_bool, intertidal_mask1, connectivity=1)

    # ## distinguish wet inland class from wet ocean class
    wet_inland = wet_bool.where((wet_bool==0) & (intertidal_mask2 == False))#, drop=True) ## Weird artefacts when drop=True
    wet_ocean = wet_bool.where((wet_bool==0) & (intertidal_mask2 == True), drop=True)

    '''--------------------------------------------------------------------'''
    ## Classify 'intermittently wet' pixels into 'intermittently_wet_inland' and 'other-intertidal_fringe'
    ## Identify and relabel the pixels in 'freq' that are 'intermittent_nontidal wet (0)' and 'other (1)'.
    int_nt = xr.where(freq==intermittent_nontidal,False,True)
    ## If deep-sea masked pixels, replace Nans with 'wet' boolean (0)
    if freq.isnull().any()==True is True:
        int_nt = int_nt.where(freq.notnull(), 0)

    ## Applying intertidal_connection masking function to separate inland from intertidal connected pixels
    intertidal_mask = intertidal_connection(int_nt, intertidal_mask1, connectivity=1)

    # ## distinguish intermittent inland from intermittent-other (intertidal_fringe) pixels
    intermittent_inland = int_nt.where((int_nt==0) & (intertidal_mask == False))#, drop=True) ## Weird artefacts when drop=True
    intertidal_fringe = int_nt.where((int_nt==0) & (intertidal_mask == True), drop=True)
    
    ## Isolate mostly dry pixels from intertidal_fringe class
    mostly_dry = intertidal_fringe.where(freq < 0.05, drop=True)
    ## Isolate mostly wet pixels from intertidal fringe class
    mostly_wet = intertidal_fringe.where(freq >= 0.05, drop=True)
    '''--------------------------------------------------------------------'''
    ## Combine wet_ocean and intertidal_fringe pixels
    wet_ocean = wet_ocean.combine_first(mostly_wet)
    
    ## Relabel pixels
    dry = dry.where(dry.isnull(), 0)
    wet_ocean = wet_ocean.where(wet_ocean.isnull(),3)
    wet_inland = wet_inland.where(wet_inland.isnull(),2)
    intermittent_inland = intermittent_inland.where(intermittent_inland.isnull(),1)
    intertidal_hc = intertidal_hc.where(intertidal_hc.isnull(),4)
    intertidal_lc = intertidal_lc.where(intertidal_lc.isnull(),5)
    # Add intertidal_fringe pixels to wet_ocean class
    # intertidal_fringe = intertidal_fringe.where(intertidal_fringe.isnull(),6)
    mostly_dry = mostly_dry.where(mostly_dry.isnull(),6)
    

    ## Combine
    extents = dry.combine_first(wet_ocean)
    extents = extents.combine_first(wet_inland)
    extents = extents.combine_first(intertidal_hc)
    extents = extents.combine_first(intermittent_inland)
    # extents = extents.combine_first(intertidal_fringe)
    extents = extents.combine_first(intertidal_lc)
    extents = extents.combine_first(mostly_dry)

    
    
    return extents