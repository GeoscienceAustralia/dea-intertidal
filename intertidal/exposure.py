import xarray as xr
import numpy as np

from dea_tools.coastal import pixel_tides


def pixel_exp(dem,
              timerange, 
              directory='~/dev_intexp/dea-notebooks/tide_models_clipped'
              ):
    '''
    
    '''
    ## Create a copy of the NIDEM 'tide_m' dataset
    dem_tide_m = dem.copy(deep=True)
    ## Create a dataset from NIDEM to preserve the shape of the data
    dem = dem.to_dataset()
    # Drop the NIDEM tide heights and unneccessary coords
    dem=dem.drop(['quantile', 'variable', 'tide_m'])
    # Add a time array to enable the pixel_tides func
    dem = dem.assign_coords(time=timerange)
    #reorder the coords 
    dem=dem[['time','y','x','spatial_ref']]
    
    ## Create the tide-height percentiles from which to calculate exposure statistics
    pc_range = np.linspace(0,1,1001)
    ## Run the pixel_tides function with the calculate_quantiles option. For each pixel, an array of tideheights is returned, corresponding to the percentiles from pc_range of the timerange-tide model that each tideheight appears in the model.
    dem['tide_cq'], _ = pixel_tides(dem, 
                                     resample=True, 
                                     directory=directory,
                                     calculate_quantiles = pc_range) 
    ## Replace the pixel-based NIDEM values
    dem['tide_m'] = dem_tide_m
    
    ## Calculate the tide-height difference between the NIDEM value and each percentile value per pixel
    dem['diff'] = abs(dem.tide_cq - dem.tide_m)

    ## Take the percentile of the smallest tide-height difference as the exposure % per pixel
    dem['idxmin'] = dem['diff'].idxmin(dim='quantile')
    
    return dem


