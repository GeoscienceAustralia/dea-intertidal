import xarray as xr

from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk

from odc.algo import mask_cleanup
import odc.geo.xr


def intertidal_connection(water_intertidal, intertidal, connectivity=1):
    """

    Identifies areas of water pixels that are adjacent to or directly
    connected to intertidal pixels.

    Parameters:
    -----------
    water_intertidal : xarray.DataArray
        An array containing True for pixels that are either water or
        intertidal pixels.
    intertidal : xarray.DataArray
        An array containing True for intertidal pixels.
    connectivity : integer, optional
        An integer passed to the 'connectivity' parameter of the
        `skimage.measure.label` function.

    Returns:
    --------
    intertidal_connection : xarray.DataArray
        An array containing the a mask consisting of identified
        intertidally-connected pixels as True.
    """

    # First, break `water_intertidal` array into unique, discrete
    # regions/blobs.
    blobs = xr.apply_ufunc(label, water_intertidal, 0, False, connectivity)

    # For each unique region/blob, use region properties to determine
    # whether it overlaps with a feature from `intertidal`. If
    # it does, then it is considered to be adjacent or directly connected
    # to intertidal pixels
    intertidal_connection = blobs.isin(
        [
            i.label
            for i in regionprops(blobs.values, intertidal.values)
            if i.max_intensity
        ]
    )

    return intertidal_connection


def extents(
    dem,
    freq,
    corr,
    reclassified_aclum,
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.15,
):
    """
    Classify coastal ecosystems into broad classes based
    on their respective patterns of wetting frequency,
    proximity to intertidal pixels and relationship to tidal
    inundation and urban land use (to mask misclassifications).

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
        and tide height, generated during the intertidal.elevation workflow
    reclassified_aclum : str
        An xarray.DataArray containing reclassified land use data, used
        to mask out urban areas misclassified as water.

    Returns:
    --------
    extents: xarray.DataArray
        A binary xarray.DataArray depicting dry (0), inland intermittent wet (1),
        inland persistent wet (2), tidal influenced persistent wet (3),
        intertidal (low confidence, 4) and intertidal (high confidence, 5) coastal extents.

    Notes:
    ------
    Classes are defined as follows:

    0: Dry
        - Pixels with wetness `freq` < 0.01
        Includes pixels that meet the following criteria:
        - Intermittently wet pixels with wetness frequency > 0.01 and < 0.99 and
        - Un-correclated to tide (p>0.15) and either of the following:
        - Connected to the intertidal class and has wetness frequency less than 0.1 or
        - Unconnected to the intertidal class and intersects with urban use land class
    1: Inland intermittent wet
        - Pixels with wetness frequency > 0.01 and < 0.99 and
        - Un-correclated to tide (p>0.15) and
        - Unconnected to the intertidal class and
        - Does not intersect with urban use land class
    2: Inland persistent wet
        - Pixels with wettness 'freq' > 0.99 and
        - Unconnected to the intertidal class
    3: Tidal influenced persistent wet
        - Pixels with wettness 'freq' > 0.99 and
        - Connected to the intertidal class
        Includes pixels that meet the following criteria:
        - Intermittently wet pixels with wetness frequency > 0.01 and < 0.99 and
        - Un-correclated to tide (p>0.15) and
        - Connected to the intertidal class and
        - Wetness frequency >= 0.1
    4: Intertidal low confidence
        - Frequency of pixel wetness (`freq`) is > 0.01 and < 0.99 and
        - The correlation (`corr`) between `freq` and tide-heights is > 0.15 and
        - Pixels do not have a valid elevation value (meaning their rolling NDWI median
          does not cross zero. However, these are still likely to be intertidal pixels
          as their rolling median curves likely fall completely above or below NDWI=0)
    5: Intertidal high confidence
        - Frequency of pixel wetness (`freq`) is > 0.01 and < 0.99 and
        - The correlation (`corr`) between `freq` and tide-heights is > 0.15 and
        - pixels have a valid elevation value (meaning their rolling NDWI median
          crosses zero)

    """

    """--------------------------------------------------------------------"""
    ## Set the upper and lower freq thresholds
    upper, lower = max_freq, min_freq

    # Set NaN values (i.e. pixels masked out over deep water) in frequency to 1
    freq = freq.fillna(1)

    ## Identify broad classes based on wetness frequency and tidal correlation
    dry = freq < lower
    intermittent = (freq >= lower) & (freq <= upper)
    wet = freq > upper

    ##### Separate intermittent_tidal (intertidal)
    intertidal = intermittent & (corr >= min_correlation)

    ##### Separate intermittent_nontidal
    intermittent_nontidal = intermittent & (corr < min_correlation)

    ##### Separate high and low confidence intertidal pixels
    intertidal_hc = intertidal & dem.notnull()
    intertidal_lc = intertidal & dem.isnull()

    """--------------------------------------------------------------------"""
    # Clean up the urban land masking class by removing high confidence intertidal areas
    reclassified_aclum = reclassified_aclum & ~intertidal_hc

    # Erode the intensive urban land use class to remove extents-class overlaps from
    # the native 50m CLUM pixel resolution dataset
    reclassified_aclum = mask_cleanup(
        mask=reclassified_aclum, mask_filters=[("erosion", 5)]
    )

    ##### Classify 'wet' pixels based on connectivity to intertidal pixels (into 'wet_ocean' and 'wet_inland')

    # Create a true/false layer of intertidal pixels (1) vs everything else (0)
    # Extract intertidal pixels (value 1) then erode these by 1 pixels to ensure we only
    # use high certainty intertidal regions for identifying connectivity to wet
    # pixels in our satellite imagery.
    inter = intertidal_hc | intertidal_lc

    # Erode outer edge pixels by 1 pixel to drop extrema intertidal pixels and ensure connection
    # to high certainty intertidal pixels
    inter = xr.apply_ufunc(binary_erosion, inter, disk(1))

    # Applying intertidal_connection masking function for the first of two times
    # This first mask identifies where wet+intertidal (e.g. not dry) pixels
    # connect to intertidal pixels
    intertidal_mask1 = intertidal_connection(~dry, inter, connectivity=1)

    # Applying intertidal_connection masking function for the second time,
    # testing for wet pixel connection to the connected 'wet and intertidal' mask.
    intertidal_mask2 = intertidal_connection(wet, intertidal_mask1, connectivity=1)

    # Mask out areas identified as 'intensive urban use' in ABARES CLUM dataset
    intertidal_mask2 = intertidal_mask2 & ~reclassified_aclum

    # Distinguish wet inland class from wet ocean class
    wet_inland = wet & ~intertidal_mask2
    wet_ocean = wet & intertidal_mask2

    """--------------------------------------------------------------------"""
    ## Classify 'intermittently wet' pixels into 'intermittently_wet_inland' and 'other-intertidal_fringe'

    ## Applying intertidal_connection masking function to separate inland from intertidal connected pixels
    intertidal_mask = intertidal_connection(
        intermittent_nontidal, intertidal_mask1, connectivity=1
    )

    # Mask out areas identified as 'intensive urban use' in ABARES CLUM dataset
    intertidal_mask = intertidal_mask & ~reclassified_aclum

    # Distinguish intermittent inland from intermittent-other (intertidal_fringe) pixels
    intermittent_inland = intermittent_nontidal & ~intertidal_mask
    intertidal_fringe = intermittent_nontidal & intertidal_mask

    # Isolate mostly dry pixels from intertidal_fringe class
    mostly_dry = intertidal_fringe & (freq < 0.1)
    # Isolate mostly wet pixels from intertidal fringe class
    mostly_wet = intertidal_fringe & (freq >= 0.1)

    # Separate misclassified urban pixels into 'dry' class
    urban_dry = reclassified_aclum & intermittent_inland
    urban_dry1 = reclassified_aclum & intertidal_hc
    urban_dry2 = reclassified_aclum & intertidal_lc

    # Identify true classified classes
    intermittent_inland = intermittent_inland & ~urban_dry
    intertidal_hc = intertidal_hc & ~urban_dry1
    intertidal_lc = intertidal_lc & ~urban_dry2

    """--------------------------------------------------------------------"""
    # Combine wet_ocean and intertidal_fringe pixels
    wet_ocean = wet_ocean | mostly_wet

    # Combine urban_dry classes
    urban_dry = urban_dry1 | urban_dry2

    # Relabel pixels
    dry = (dry * 0).where(dry)
    wet_ocean = (wet_ocean * 3).where(wet_ocean)
    wet_inland = (wet_inland * 2).where(wet_inland)
    intermittent_inland = (intermittent_inland * 1).where(intermittent_inland)
    intertidal_hc = (intertidal_hc * 5).where(intertidal_hc)
    intertidal_lc = (intertidal_lc * 4).where(intertidal_lc)
    mostly_dry = (mostly_dry * 0).where(mostly_dry)
    urban_dry = (urban_dry * 0).where(urban_dry)

    # Combine
    extents = dry.combine_first(wet_ocean)
    extents = extents.combine_first(wet_inland)
    extents = extents.combine_first(intertidal_hc)
    extents = extents.combine_first(intermittent_inland)
    extents = extents.combine_first(intertidal_lc)
    extents = extents.combine_first(mostly_dry)
    extents = extents.combine_first(0)

    return extents
