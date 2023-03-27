import xarray as xr
import numpy as np

from dea_tools.coastal import pixel_tides


def pixel_exp(ds,
              timerange, 
              directory='~/dev_intexp/dea-notebooks/tide_models_clipped'
              ):
    """
    Calculate exposure percentage for each pixel based on tide-height differences between the NIDEM value and percentile values of the tide model for a given time range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing Digital Elevation Model (DEM) data and metadata.
    timerange : tuple
        Tuple containing start and end time of time range to be used for tide model in the format of "YYYY-MM-DD".
    directory : str, optional
        Directory path to tide models, by default '~/dev_intexp/dea-notebooks/tide_models_clipped'.

    Returns
    -------
    xarray.Dataset
        Dataset containing a new int 16 array 'exposure' representing exposure percentage time for each pixel.

    Notes
    -----
    - The tide-height percentiles range from 0 to 100, divided into 101 equally spaced values.
    - The 'diff' variable is calculated as the absolute difference between tide model percentile value and the DEM value at each pixel.
    - The 'idxmin' variable is the index of the smallest tide-height difference (i.e., maximum similarity) per pixel and is equivalent to the exposure percent.
    """
    
    ## Create a Dataset to run pixel_tides on
    ds_exposure = xr.Dataset(coords=ds.coords, attrs=ds.attrs)
    
    ## Create the tide-height percentiles from which to calculate exposure statistics
    pc_range = np.linspace(0,1,101)
    
    ## Run the pixel_tides function with the calculate_quantiles option. For each pixel, an array of tideheights is returned, corresponding to the percentiles from pc_range of the timerange-tide model that each tideheight appears in the model.
    ds['tide_cq'], _ = pixel_tides(ds_exposure, 
                                     resample=True, 
                                     directory=directory,
                                     calculate_quantiles = pc_range,
                                     times=timerange) 
    
    ## Calculate the tide-height difference between the NIDEM value and each percentile value per pixel
    diff = abs(ds.tide_cq - ds.dem)

    ## Take the percentile of the smallest tide-height difference as the exposure % per pixel
    idxmin = diff.idxmin(dim='quantile')
    
    ## convert to percentage then int and add to master ds
    idxmin = (idxmin * 100).astype(np.int16)
    
    ds['exposure'] = idxmin
    
    return ds


