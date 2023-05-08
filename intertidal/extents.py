import xarray as xr
import numpy as np

import datacube
from skimage.measure import label, regionprops
from skimage.morphology import (binary_erosion, disk)

def ocean_masking(ds, ocean_da, connectivity=1, dilation=None):
    """
    ## from https://github.com/GeoscienceAustralia/dea-coastlines/blob/develop/coastlines/vector.py#L188-L198
    
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
    ocean_mask : xarray.DataArray
        An array containing the a mask consisting of identified ocean
        pixels as True.
    """

    # Update `ocean_da` to mask out any pixels that are land in `ds` too
    ocean_da = ocean_da & (ds != 1)

    # First, break all time array into unique, discrete regions/blobs.
    # Fill NaN with 1 so it is treated as a background pixel
    blobs = xr.apply_ufunc(label, ds.fillna(1), 1, False, connectivity)

    # For each unique region/blob, use region properties to determine
    # whether it overlaps with a water feature from `water_mask`. If
    # it does, then it is considered to be directly connected with the
    # ocean; if not, then it is an inland waterbody.
    ocean_mask = blobs.isin(
        [i.label for i in regionprops(blobs.values, ocean_da.values) if i.max_intensity]
    )

    # Dilate mask so that we include land pixels on the inland side
    # of each shoreline to ensure contour extraction accurately
    # seperates land and water spectra
    if dilation:
        ocean_mask = xr.apply_ufunc(binary_dilation, ocean_mask, disk(dilation))

    return ocean_mask


def extents(freq,
           dem,
           corr,
           product = 'geodata_coast_100k'):
    '''
    Classify coastal ecosystems into broad classes based 
    on their respective patterns of wetting frequency,
    relationship to tidal inundation and proximity to
    the ocean.

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
    ocean_da : xarray.DataArray
        A supplementary static dataset used to separate ocean waters
        from other inland water. The array should contain values of 1
        for high certainty ocean pixels, and 0 for all other pixels
        (land, inland water etc). For Australia, we use the  Geodata
        100K coastline dataset, rasterized as the "geodata_coast_100k"
        product on the DEA datacube.

    Returns:
    --------
    extents: xarray.DataArray
        A binary xarray.DataArray depicting intertidal (0), tidal-wet (1),
        nontidal-wet (2), intermittently, non-tidal wet (3) and dry (4) coastal extents.
    Notes:
    ------
    Classes are defined as follows:
    0: Dry
        Pixels with wettness `freq` < 0.05
        Includes intermittently dry pixels with wetness frequency < 0.5 and > 0.05
        and `corr` to tide > 0.1 to capture intertidal pixels buffered
        out by the `corr` threshold of 0.2
    1: Intertidal
        Frequency of pixel wetness (`freq`) is > 0.01 and < 0.99
        The correlation (`corr`) between `freq` and tide-heights is > 0.2
    2: Wet tidal
        Frequency of pixel wetness (`freq`) is > 0.95
        Includes intermittently wet pixels with `freq` > 0.5 and < 0.95,
        and `corr` to tide > 0.1. This captures intertidal pixels buffered
        out by the `corr` threshold of 0.2 (default)
        Pixels are located offshore, within 10 pixels of known ocean, as defined
        by the Geodata 100k coastline dataset (`ocean_da`)
    3: Wet nontidal
        Frequency of pixel wetness (`freq`) is > 0.95
        Includes intermittently wet pixels with `freq` > 0.5 and < 0.95,
        and `corr` to tide > 0.1. This captures intertidal pixels buffered
        out by the `corr` threshold of 0.2 (default)
        Pixels are located onshore, more than 10 pixels from known ocean, as defined
        by the Geodata 100k coastline dataset (`ocean_da`)
    4: Intermittently wet nontidal
        Pixels with wetting `freq` between 0.95 and 0.05 and
        `corr` of `freq` to tide is < 0.1    
    '''
    
    ## Connect to datacube to load `ocean_da`
    dc = datacube.Datacube(app='ocean_masking')
    
    ## Isolate intertidal class
    intertidal = freq.where(dem.notnull())

    ## Isolate non-intertidal class
    wetdry = freq.where(dem.isnull())

    ## Separate non-intertidal areas into always wet and always dry classes
    wet = wetdry.where(wetdry >= 0.95, drop=True)
    dry = wetdry.where((wetdry <= 0.05), drop=True)

    ## Identify intermittent tidal classes 
    intermittent_tidal_wet = freq.where(
                                        (wetdry < 0.95)
                                        & (wetdry >= 0.5) 
                                        & (corr > 0.1) 
                                       )

    intermittent_tidal_dry = freq.where(
                                        (wetdry < 0.5)
                                        & (wetdry > 0.05) 
                                        & (corr > 0.1) 
                                        )

    ## Identify intermittent non-tidal class
    intermittent_nontidal = freq.where(
                                       (wetdry < 0.95)
                                       & (wetdry > 0.05) 
                                       & (corr <= 0.1)
                                       )

    ## Relabel pixels in the classes. 
    ## Add intermittent tidal wet/dry to the always wet/dry classes
    dry = dry.where(dry.isnull(), 0)
    intermittent_tidal_dry = intermittent_tidal_dry.where(intermittent_tidal_dry.isnull(), 0)
    
    intertidal = intertidal.where(intertidal.isnull(), 1)
    
    wet = wet.where(wet.isnull(), 2)
    intermittent_tidal_wet = intermittent_tidal_wet.where(intermittent_tidal_wet.isnull(), 2)
        
    intermittent_nontidal = intermittent_nontidal.where(intermittent_nontidal.isnull(), 4)

    ## Combine classes
    extents = intertidal.combine_first(wet)
    extents = extents.combine_first(dry)
    extents = extents.combine_first(intermittent_tidal_wet)
    extents = extents.combine_first(intermittent_tidal_dry)
    extents = extents.combine_first(intermittent_nontidal)

    ## Separate the onshore from offshore 'always wet' class

    ## Mask all pixels in `intext` as `always_wet` or 'other'
    landwater=extents.where(extents != 1, False)

    ## Load the Geodata 100k coastline layer to use as the seawater mask

    # Load Geodata 100K coastal layer to use to separate ocean waters from
    # other inland waters. This product has values of 0 for ocean waters,
    # and values of 1 and 2 for mainland/island pixels. We extract ocean
    # pixels (value 0), then erode these by 10 pixels to ensure we only
    # use high certainty deeper water ocean regions for identifying ocean
    # pixels in our satellite imagery. If no Geodata data exists (e.g.
    # over remote ocean waters, use an all True array to represent ocean.
    try:    
        geodata_da = dc.load(product = product,
                            like=landwater.odc.geobox.compat
                          ).land.squeeze('time')
        ocean_da = xr.apply_ufunc(binary_erosion, geodata_da == 0, disk(10))
    except AttributeError:
        ocean_da = odc.geo.xr.xr_zeros(landwater.odc.geobox) == 0
    # except ValueError:  # Temporary workaround for no geodata access for tests from https://github.com/GeoscienceAustralia/dea-coastlines/blob/develop/coastlines/vector.py
    #     ocean_da = xr.apply_ufunc(binary_erosion, all_time_20==0, disk(20))

    ## Applying ocean_masking function
    ocean_mask = ocean_masking(landwater, ocean_da)

    ## distinguish wet tidal from non-tidal pixels
    wet_nontidal = extents.where((extents==2) & (ocean_mask == False))#, drop=True) ## Weird artefacts when drop=True
    wet_tidal = extents.where((extents==2) & (ocean_mask == True), drop=True)

    ## Relabel pixels
    wet_nontidal = wet_nontidal.where(wet_nontidal.isnull(), 3)
    wet_tidal = wet_tidal.where(wet_tidal.isnull(), 2)

    ## remove `wet` pixels from int_ext to replace with the tidal and non tidal wet classes
    extents = extents.where(extents != 2, np.nan)

    ## combine wet tidal and nontidal variables back into int_ext
    extents = extents.combine_first(wet_nontidal)
    extents = extents.combine_first(wet_tidal)
    
    return extents

