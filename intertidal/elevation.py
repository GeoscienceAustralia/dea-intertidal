import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from dea_tools.dask import create_local_dask_cluster
from eo_tides.eo import pixel_tides
from odc.algo import xr_quantile
from odc.geo.geom import BoundingBox
from skimage.morphology import binary_dilation
from tqdm.auto import tqdm

from intertidal.exposure import exposure
from intertidal.extents import extents, load_connectivity_mask
from intertidal.io import (
    export_dataset_metadata,
    load_aclum_mask,
    load_data,
    load_topobathy_mask,
    prepare_for_export,
    tidal_metadata,
)
from intertidal.tidal_bias_offset import bias_offset
from intertidal.utils import (
    configure_logging,
    spearman_correlation,
)


def ds_to_flat(
    satellite_ds,
    ndwi_thresh=0.0,
    index="ndwi",
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.15,
    corr_method="pearson",
    apply_threshold=True,
    correct_seasonality=False,
    valid_mask=None,
):
    """Flattens a three-dimensional array (x, y, time) to a two-dimensional
    array (time, z) by selecting only pixels with positive correlations
    between water observations and tide height. This greatly improves
    processing time by ensuring only a narrow strip of pixels along the
    coastline are analysed, rather than the entire x * y array.
    The x and y dimensions are stacked into a single dimension (z)

    Parameters
    ----------
    satellite_ds : xr.Dataset
        Three-dimensional (x, y, time) xarray dataset with variable
        "tide_m" and a water index variable as provided by `index`.
    ndwi_thresh : float, optional
        Threshold for NDWI index used to identify wet or dry pixels.
        Default is 0.0.
    index : str, optional
        Name of the water index variable. Default is "ndwi".
    min_freq : float, optional
        Minimum frequency of wetness required for a pixel to be included
        in the output. Default is 0.01.
    max_freq : float, optional
        Maximum frequency of wetness required for a pixel to be included
        in the output. Default is 0.99.
    min_correlation : float, optional
        Minimum correlation between water index values and tide height
        required for a pixel to be included in the output. Default is
        0.15.
    corr_method : str, optional
        Correlation method to use. Defaults to "pearson", also supports
        "spearman".
    apply_threshold : bool, optional
        Whether to threshold the water index timeseries before calculating
        correlations, to ensure small changes in index values beneath the
        water surface are not included when calcualting correlations.
        Defaults to True.
    correct_seasonality : bool, optional
        If True, remove any seasonal signal from the tide height data
        by subtracting monthly mean tide height from each value. This
        can reduce false tide correlations in regions where tide heights
        correlate with seasonal changes in surface water. Note that
        seasonally corrected tides are only used to identify potentially
        tide influenced pixels - not for elevation modelling itself.
    valid_mask : xr.DataArray, optional
        A boolean mask used to optionally constrain the analysis area,
        with the same spatial dimensions as `satellite_ds`. For example,
        this could be a mask generated from a topo-bathy DEM, used to
        limit the analysis to likely intertidal pixels. Default is None,
        which will not apply a mask.

    Returns
    -------
    flat_ds : xr.Dataset
        Two-dimensional xarray dataset with dimensions (time, z),
        containing NDWI and tide height variables.
    freq : xr.DataArray
        Frequency of wetness for each pixel (where NDWI > `ndwi_thresh`).
    corr : xr.DataArray
        Correlation of NDWI pixel wetness with tide height.

    """
    # Calculate clear count
    clear = satellite_ds[index].notnull().sum(dim="time").rename("qa_count_clear")

    # Calculate frequency of wet per pixel, then threshold
    # to exclude always wet and always dry
    freq = (
        (satellite_ds[index] > ndwi_thresh).where(~satellite_ds[index].isnull()).mean(dim="time").rename("qa_ndwi_freq")
    )

    # Mask out pixels outside of frequency bounds
    freq_mask = (freq >= min_freq) & (freq <= max_freq)
    satellite_ds = satellite_ds.where(freq_mask)

    # If an overall valid data mask is provided, apply to the data
    if valid_mask is not None:
        satellite_ds = satellite_ds.where(valid_mask)

    # Flatten satellite and freq data by stacking "y" and "x" dims.
    # Drop any pixels that are always empty, or empty timesteps
    flat_ds = satellite_ds.stack(z=("y", "x")).dropna(dim="time", how="all").dropna(dim="z", how="all")
    freq = freq.stack(z=("y", "x"))
    clear = clear.stack(z=("y", "x"))

    # Calculate correlations between NDWI water observations and tide
    # height. By default, because we are only interested in pixels with
    # inundation patterns (e.g.transitions from dry to wet) that are driven
    # by the tide, we first convert NDWI into a boolean dry/wet layer before
    # running the correlation. This prevents small changes in NDWI beneath
    # the water surface from producing correlations with tide height.
    # This can be turned off by passing `apply_threshold=False`.
    if apply_threshold:
        wet_dry = flat_ds[index] > ndwi_thresh
    else:
        print("Using raw water index values for correlation")
        wet_dry = flat_ds[index]

    # Use either tides directly or correct to remove seasonal signal
    if correct_seasonality:
        print("Removing seasonal signal before calculating tide correlations")
        gb = flat_ds.tide_m.groupby("time.month")
        tide_array = gb - gb.mean()
    else:
        tide_array = flat_ds.tide_m

    # Calculate correlation
    if corr_method == "pearson":
        corr = xr.corr(wet_dry, tide_array, dim="time")
    elif corr_method == "spearman":
        print("Applying Spearman correlation")
        corr = spearman_correlation(x=wet_dry, y=tide_array, dim="time")

    # Keep only pixels with correlations that meet min threshold
    corr = corr.rename("qa_ndwi_corr")
    corr_mask = corr >= min_correlation
    flat_ds = flat_ds.where(corr_mask, drop=True)

    # Return pixels identified as intertidal candidates
    intertidal_candidates = corr_mask.where(corr_mask, drop=True)
    print(
        f"Reducing analysed pixels from {freq.count().item()} to "
        f"{len(intertidal_candidates.z)} ({len(intertidal_candidates.z) * 100 / freq.count().item():.2f}%)"
    )

    return flat_ds, freq, corr, clear


def rolling_tide_window(
    i,
    flat_ds,
    window_spacing,
    window_radius,
    tide_min,
    min_count=5,
    statistic="median",
):
    """Filter observations from a flattened array that fall within a
    specific tide window, and summarise these values using a given
    statistic (median, mean, or quantile).

    This is used to smooth NDWI values along the tide dimension so that
    we can more easily identify the transition from dry to wet pixels
    with increasing tide height.

    Parameters
    ----------
    i : int
        Index of the current window.
    flat_ds : xarray.Dataset
        Input dataset with tide observations (tide_m) as a dimension.
    window_spacing : float
        Provides the spacing of each rolling window interval in tide
        units (e.g. metres).
    window_radius : float
        Provides the radius/width of each rolling window in tide units
        (e.g. metres).
    tide_min : float
        Bottom edge of the rolling window in tide units (e.g. metres).
    min_count : int, optional
        The minimum number of valid datapoints required to calculate the
        rolling statistic. Outputs with less observations will be set to
        NaN. Defaults to 5.
    statistic : str, optional
        Statistic to apply on the values within each window. One of
        ["median", "mean", "quantile"]. Default is "median".

    Returns
    -------
    xarray.Dataset
        Aggregated dataset of the selected statistic and additional
        information on the window. The returned dataset includes the
        aggregated NDWI values within the window.

    """
    # Set min and max thresholds to filter dataset
    thresh_centre = tide_min + (i * window_spacing)
    thresh_min = thresh_centre - window_radius
    thresh_max = thresh_centre + window_radius

    # Filter dataset
    masked_ds = flat_ds.where((flat_ds.tide_m >= thresh_min) & (flat_ds.tide_m <= thresh_max))

    # Apply median or quantile
    if statistic == "quantile":
        ds_agg = xr_quantile(
            src=masked_ds.dropna(dim="time", how="all"),
            quantiles=[0.1, 0.5, 0.9],
            nodata=np.nan,
        )
    elif statistic == "median":
        ds_agg = masked_ds.median(dim="time")
    elif statistic == "mean":
        ds_agg = masked_ds.mean(dim="time")

    # Optionally mask out observations with less than n valid datapoints.
    if min_count:
        clear_count = masked_ds.notnull().sum(dim="time")
        ds_agg = ds_agg.where(clear_count > min_count)

    return ds_agg


def pixel_rolling_median(
    flat_ds,
    windows_n=100,
    window_prop_tide=0.15,
    window_offset=5,
    min_count=5,
    max_workers=None,
):
    """Calculate rolling medians for each pixel in an xarray.Dataset from
    low to high tide, using a set number of rolling windows (defined
    by `windows_n`) with radius determined by the proportion of the tide
    range specified by `window_prop_tide`.

    For each window, the function returns the median of all tide heights
    and NDWI index values within the window, and returns an array with a
    new "interval" dimension that summarises these values from low to
    high tide.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened two dimensional (time, z) xr.Dataset containing
        variables "ndwi" and "tide_height", as produced by the
        `ds_to_flat` function
    windows_n : int, optional
        Number of rolling windows to iterate over, by default 100
    window_prop_tide : float, optional
        Proportion of the tide range to use for each window radius,
        by default 0.15
    window_offset : int, optional
        The number of additional rolling windows to process at the
        bottom of the tidal range. This can be used to provide
        additional coverage of the lower intertidal zone by starting the
        first rolling window beneath the lowest tide, although at the
        risk of introducing noisy data due to the rolling medians
        containing fewer total satellite observations. Defaults to 5.
    min_count : int, optional
        The minimum number of cloud free observations required to
        calculate the rolling statistic. Defaults to 5; higher values
        will produce cleaner results but with potentially reduced
        intertidal coverage.
    max_workers : int, optional
        Maximum number of worker processes to use for parallel
        execution, by default 64

    Returns
    -------
    interval_ds : xarray.Dataset
        An two dimensional (interval, z) xarray.Dataset containing
        rolling medians for each pixel along intervals from low to high
        tide.

    """
    # First obtain some required statistics on the satellite-observed
    # min, max and tide range per pixel
    tide_max = flat_ds.tide_m.max(dim="time")
    tide_min = flat_ds.tide_m.min(dim="time")
    tide_range = tide_max - tide_min

    # To conduct a pixel-wise rolling median, we first need to calculate
    # some statistics on the tides observed for each individual pixel in
    # the study area. These are then used to calculate rolling windows
    # that are unique/tailored for the tidal regime of each pixel:
    #
    #     - window_radius_tide: Provides the radius/width of each
    #       rolling window in tide units (e.g. metres).
    #     - window_spacing_tide: Provides the spacing of each rolling
    #       window interval in tide units (e.g. metres)
    #
    window_radius_tide = tide_range * window_prop_tide
    window_spacing_tide = tide_range / windows_n

    # Parallelise pixel-based rolling median using `concurrent.futures`
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create rolling intervals to iterate over, starting the first
        # interval at `windows_offset` windows below the lowest tide.
        rolling_intervals = range(-window_offset, windows_n)

        # Place itervals in a iterable along with params for each call
        to_iterate = (
            rolling_intervals,
            *(
                repeat(i, len(rolling_intervals))
                for i in [
                    flat_ds,
                    window_spacing_tide,
                    window_radius_tide,
                    tide_min,
                    min_count,
                ]
            ),
        )

        # Apply func in parallel
        out_list = list(
            tqdm(
                executor.map(rolling_tide_window, *to_iterate),
                total=len(list(rolling_intervals)),
            )
        )

    # Combine to match the shape of the original dataset, then sort from
    # low to high tide
    interval_ds = xr.concat(out_list, dim="interval").sortby("interval")

    return interval_ds


def pixel_dem(
    interval_ds,
    ndwi_thresh=0.1,
    interp_intervals=200,
    smooth_radius=20,
    min_periods=5,
    debug=False,
):
    """Calculates an estimate of intertidal elevation based on satellite
    imagery and tide data. Elevation is modelled by identifying the
    tide height at which a pixel transitions from dry to wet; calculated
    here as the first/minimum tide height at which a rolling median of
    NDWI becomes characterised as water (e.g. NDWI > `ndwi_thresh`).

    This function can additionally interpolate to a higher number of
    intertidal intervals and/or apply a rolling mean to smooth data
    before the elevation extraction. This can produce a cleaner output.

    Parameters
    ----------
    interval_ds : xarray.Dataset
        A flattened 2D xarray Dataset containing the rolling median for
        each pixel from low to high tide for the given area, with
        variables 'tide_m' and 'ndwi'.
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI), above which pixels are considered water. Defaults to
        0.1, which appears to more reliably capture the transition from
        dry to wet pixels than 0.0.
    interp_intervals : int, optional
        Whether to interpolate to an increased density of intervals.
        This can be useful for reducing the impact of "terrace"-like
        artefacts across very low sloping intertidal flats where we have
        minimal satellite observations. Defaults to 200; set to None to
        deactivate.
    smooth_radius : int, optional
        A rolling mean filter can be applied to smooth data along the
        tide interval dimension. This produces smoother DEM surfaces
        than using the rolling median directly. Defaults to 20; set to
        None to deactivate.
    min_periods : int or string, optional
        Minimum number of valid datapoints required to calculate rolling
        mean if `smooth_radius` is set. Defaults to 5; "auto" will use
        `int(smooth_radius / 2.0)`; `None` will use the size of the window.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the DEM for the given area, with
        a single variable 'elevation'.

    """
    # Apply optional interval interpolation
    if interp_intervals is not None:
        print(f"Applying tidal interval interpolation to {interp_intervals} intervals")
        interval_ds = interval_ds.interp(
            coords={"interval": np.linspace(0, interval_ds.interval.max().item(), interp_intervals)},
            method="linear",
            # Required as recent versions of xarray return new coord as a variable
        ).set_coords("interval")

    # Smooth tidal intervals using a rolling mean
    if smooth_radius is not None:
        print(f"Applying rolling mean smoothing with radius {smooth_radius}")
        smoothed_ds = interval_ds.rolling(
            interval=smooth_radius,
            center=False,
            min_periods=(int(smooth_radius / 2.0) if min_periods == "auto" else min_periods),
        ).mean()
    else:
        smoothed_ds = interval_ds

    # Identify the first/minimum tide per pixel where rolling median
    # NDWI becomes water. This represents the tide height at which the
    # pixel transitions from dry to wet as it gets tidally inundated.
    tide_dry = smoothed_ds.tide_m.where(smoothed_ds.ndwi > ndwi_thresh)
    tide_thresh = tide_dry.min(dim="interval")

    # Remove any pixel where the identified tide threshold is equal to
    # the highest or lowest tide height observed in the rolling median.
    # These are pixels that are either always land or always water, and
    # therefore invalid for elevation modelling.
    tide_max = smoothed_ds.tide_m.max(dim="interval")
    tide_min = smoothed_ds.tide_m.min(dim="interval")
    always_dry = tide_thresh >= tide_max
    always_wet = tide_thresh <= tide_min
    dem_flat = tide_thresh.where(~always_wet & ~always_dry)

    # Convert to xr_dataset
    dem_ds = dem_flat.to_dataset(name="elevation")

    # If debug is True, return smoothed data as well
    if debug:
        return dem_ds, smoothed_ds

    return dem_ds


def pixel_dem_debug(
    x,
    y,
    flat_ds,
    interval_ds,
    ndwi_thresh=0.1,
    interp_intervals=200,
    smooth_radius=20,
    min_periods=5,
    certainty_method="mad",
    plot_style=None,
    plot_ylim=(-1, 1),
):
    # Unstack data back to x, y so we can select pixels by their coordinates
    flat_unstacked = flat_ds[["tide_m", "ndwi"]].unstack().sortby(["time", "x", "y"])
    interval_unstacked = interval_ds[["tide_m", "ndwi"]].unstack().sortby(["interval", "x", "y"])

    # Extract nearest pixel to x and y coords
    flat_pixel = flat_unstacked.sel(x=x, y=y, method="nearest")
    interval_pixel = interval_unstacked.sel(x=x, y=y, method="nearest")

    # # Experimental feature: support for variable threshold
    # if not isinstance(ndwi_thresh, float):
    #     ndwi_thresh = xr.DataArray(
    #         np.linspace(ndwi_thresh[0], ndwi_thresh[-1], interp_intervals),
    #         coords={"interval": interval_clean_pixel.interval},
    #     )

    # Calculate DEM
    flat_dem_pixel, interval_smoothed_pixel = pixel_dem(
        interval_pixel,
        ndwi_thresh=ndwi_thresh,
        interp_intervals=interp_intervals,
        smooth_radius=smooth_radius,
        min_periods=min_periods,
        debug=True,
    )

    # Calculate certainty
    elev_low_mad, elev_high_mad, _, _ = pixel_uncertainty(
        flat_pixel,
        flat_dem_pixel,
        ndwi_thresh,
        method=certainty_method,
    )

    # Plot
    flat_pixel_df = flat_pixel.to_dataframe().drop("spatial_ref", axis=1)
    flat_pixel_df["season"] = flat_pixel.time.dt.season
    flat_pixel_df["year"] = flat_pixel.time.dt.year

    if plot_style == "season":
        sns.scatterplot(data=flat_pixel_df, x="tide_m", y="ndwi", hue="season", s=15)
    elif plot_style == "year":
        sns.scatterplot(data=flat_pixel_df, x="tide_m", y="ndwi", hue="year", s=15)
    else:
        sns.scatterplot(data=flat_pixel_df, x="tide_m", y="ndwi", color="black", s=10)

    # Convert to dataframes and plot
    interval_pixel_df = interval_pixel.to_dataframe().drop("spatial_ref", axis=1)
    interval_smoothed_pixel_df = interval_smoothed_pixel.to_dataframe().drop("spatial_ref", axis=1)
    interval_pixel_df.plot(x="tide_m", y="ndwi", ax=plt.gca(), label="NDWI (rolling median)")
    interval_smoothed_pixel_df.plot(x="tide_m", y="ndwi", ax=plt.gca(), label="NDWI (rolling median, smoothed)")

    if not isinstance(ndwi_thresh, float):
        plt.plot(
            interval_smoothed_pixel.tide_m.sel(interval=~interval_smoothed_pixel.tide_m.isnull()),
            ndwi_thresh.sel(interval=~interval_smoothed_pixel.tide_m.isnull()),
            color="black",
            linestyle="--",
            lw=1,
            alpha=1,
        )
    else:
        plt.gca().axvspan(elev_low_mad.item(), elev_high_mad.item(), color="lightgrey", alpha=0.3)
        plt.gca().axhline(ndwi_thresh, color="black", linestyle="--", lw=1, alpha=1)

    plt.gca().axvline(flat_dem_pixel.elevation, color="black", linestyle="--", lw=1, alpha=1)
    plt.gca().set_ylim(plot_ylim)
    plt.gca().set_xlabel("Tide height (m)")
    plt.gca().set_ylabel("NDWI")

    return flat_pixel_df, interval_pixel_df, interval_smoothed_pixel_df


def pixel_uncertainty(
    flat_ds,
    flat_dem,
    ndwi_thresh=0.1,
    method="mad",
    min_misclassified=3,
    min_q=0.25,
    max_q=0.75,
):
    """Calculate one-sided uncertainty bounds around a modelled elevation
    based on observations that were misclassified by a given NDWI
    threshold.

    Uncertainty is based observations that were misclassified by the
    modelled elevation, i.e., wet observations (NDWI > threshold) at
    lower tide heights than the modelled elevation, or dry observations
    (NDWI < threshold) at higher tide heights than the modelled
    elevation.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened (2D) dataset containing dimensions "time" and "z",
        and variables "ndwi" (Normalized Difference Water Index) and
        "tide_m" (tide height) for each satellite observation.
    flat_dem : xarray.DataArray
        A 2D array containing modelled elevations per pixel, as
        generated by `intertidal.elevation.pixel_dem`.
    ndwi_thresh : float, optional
        A threshold value for NDWI, below which an observation is
        considered "dry", and above which it is considered "wet". The
        default is 0.1.
    method : string, optional
        Whether to calculate uncertainty using Median Absolute Deviation
        (MAD) of the tide heights of all misclassified points, or by
        taking upper/lower tide height quantiles of miscalssified points.
        Defaults to "mad" for Median Absolute Deviation; use "quantile"
        to use quantile calculation instead.
    min_misclassified : int, optional
        If `method == "mad"`: This sets the minimum number of misclassified
        observations required to calculate a valid MAD uncertainty. Pixels
        with fewer misclassified observations will be assigned an output
        uncertainty of 0 metres (reflecting how sucessfully the provided
        elevation and NDWI threshold divide observations into dry and wet).
    min_q, max_q : float, optional
        If `method == "quantile"`: the minimum and maximum quantiles used
        to estimate uncertainty bounds based on misclassified points.
        Defaults to interquartile range, or 0.25, 0.75. This provides a
        balance between capturing the range of uncertainty at each
        pixel, while not being overly influenced by outliers in `flat_ds`.

    Returns
    -------
    dem_flat_low, dem_flat_high, dem_flat_uncertainty : xarray.DataArray
        The lower and upper uncertainty bounds around the modelled
        elevation, and the summary uncertainty range between them
        (expressed as one-sided uncertainty).
    misclassified_sum : xarray.DataArray
        The sum of individual satellite observations misclassified by
        the modelled elevation and NDWI threshold.

    """
    # Identify observations that were misclassifed by our modelled
    # elevation: e.g. wet observations (NDWI > threshold) at lower tide
    # heights than our modelled elevation, or dry observations (NDWI <
    # threshold) at higher tide heights than our modelled elevation.
    misclassified_wet = (flat_ds.ndwi > ndwi_thresh) & (flat_ds.tide_m < flat_dem.elevation)
    misclassified_dry = (flat_ds.ndwi < ndwi_thresh) & (flat_ds.tide_m > flat_dem.elevation)
    misclassified_all = misclassified_wet | misclassified_dry
    misclassified_ds = flat_ds.where(misclassified_all)

    # Calculate sum of misclassified points
    misclassified_sum = (
        misclassified_all.sum(dim="time").rename("misclassified_px_count").where(~flat_dem.elevation.isnull())
    )

    # Calculate uncertainty by taking the Median Absolute Deviation of
    # all misclassified points.
    if method == "mad":
        # Calculate median of absolute deviations
        mad = abs(misclassified_ds.tide_m - flat_dem.elevation).median(dim="time")

        # Set any pixels with < n misclassified points to 0 MAD. This
        # avoids extreme MAD values being calculated when we have only
        # a small set of misclassified observations, as well as missing
        # data caused by being unable to calculate MAD on zero
        # misclassified observations.
        mad = mad.where(misclassified_sum >= min_misclassified, 0)

        # Calculate low and high bounds
        uncertainty_low = flat_dem.elevation - mad
        uncertainty_high = flat_dem.elevation + mad

    # Calculate interquartile tide height range of our misclassified
    # observations to obtain lower and upper uncertainty bounds around our
    # modelled elevation.
    elif method == "quantile":
        # Use xr_quantile (faster than built-in .quantile)
        misclassified_q = xr_quantile(
            src=misclassified_ds.dropna(dim="time", how="all")[["tide_m"]],
            quantiles=[min_q, max_q],
            nodata=np.nan,
        ).tide_m.fillna(flat_dem.elevation)

        # Extract low and high bounds
        uncertainty_low = misclassified_q.sel(quantile=min_q, drop=True)
        uncertainty_high = misclassified_q.sel(quantile=max_q, drop=True)

    # Clip min and max uncertainty to modelled elevation to ensure lower
    # bounds are not above modelled elevation (and vice versa)
    dem_flat_low = np.minimum(uncertainty_low, flat_dem.elevation)
    dem_flat_high = np.maximum(uncertainty_high, flat_dem.elevation)

    # Subtract low from high DEM to summarise uncertainty range
    # (and divide by two to give one-sided uncertainty)
    dem_flat_uncertainty = (dem_flat_high - dem_flat_low) / 2.0

    return (
        dem_flat_low,
        dem_flat_high,
        dem_flat_uncertainty,
        misclassified_sum,
    )


def flat_to_ds(flat_ds, template, stacked_dim="z"):
    """Convert a flattened xarray Dataset with a stacked dimension to its
    original spatial dimensions, based on a given template.

    Parameters
    ----------
    flat_ds : xarray.Dataset
        A flattened xarray.Dataset, i.e., a dataset where each y/x
        pixel is stacked into a single "z" dimension.
    template : xarray.Dataset or xarray.Dataarray
        A dataset  containing the original spatial dimensions and
        coordinates of the data, used as a template to reshape the
        flattened data back to the spatial dimensions.
    stacked_dim : str, optional
        The name of the stacked y/x dimension in the flattened dataset.
        The default is "z".

    Returns
    -------
    xarray.Dataset
        The unflattened xarray Dataset, with the same spatial dimensions
        (e.g. y/x) as the template.

    Notes
    -----
    The function unstacks the flattened dataset along the stacked
    dimension, reindexes the resulting dataset to match the coordinates
    of the template, and transposes the dimensions to match the order of
    the template's spatial y/x dimensions.

    """
    unstacked_ds = (
        # First, unstack back into y/x dimensions
        flat_ds.unstack(stacked_dim)
        # After unstacking, our output can be missing entire y/x
        # coordinates contained in `template`. To address this, we need
        # to "reindex" our unstacked data so that it has exactly the
        # same coordinates as `template`. Affected pixels will be filled
        # with np.nan
        .reindex_like(template)
        # Finally, we ensure that our spatial y/x dimensions have not
        # been rotated during the unstack. The `...` preserves any extra
        # non-spatial dimensions (like "time") if they exist
        .transpose(..., *template.odc.spatial_dims)
    )

    return unstacked_ds


def clean_edge_pixels(ds):
    """Clean intertidal elevation and uncertainty data by removing pixels
    along the upper edge of the intertidal zone, where mixed pixels/edge
    effects mean that modelled elevations are likely to be inaccurate.

    This function uses binary dilation to identify the edges of
    intertidal elevation data with greater than 0 elevation. The
    resulting mask is applied to the elevation dataset to remove upper
    intertidal edge pixels from both elevation and uncertainty datasets.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing elevation and uncertainty data.

    Returns
    -------
    xarray.Dataset
        Cleaned elevation dataset with upper intertidal edge pixels removed.

    """
    # Dilate nodata area to identify edges of intertidal elevation data
    dilated = binary_dilation(ds.elevation.isnull())

    # Identify upper intertidal pixels as those on edge of intertidal
    # with elevations greater than 0
    upper_elevation = ds.elevation > 0
    upper_intertidal_edge = dilated & upper_elevation

    # Apply mask to elevation dataset
    return ds.where(~upper_intertidal_edge)


def elevation(
    satellite_ds,
    valid_mask=None,
    ndwi_thresh=0.1,
    min_freq=0.01,
    max_freq=0.99,
    min_correlation=0.15,
    windows_n=100,
    window_prop_tide=0.15,
    correct_seasonality=False,
    max_workers=None,
    tide_model="EOT20",
    tide_model_dir="/var/share/tide_models",
    run_id=None,
    log=None,
    **model_tides_kwargs,
):
    """Generates DEA Intertidal Elevation outputs using satellite imagery
    and tidal modeling.

    Parameters
    ----------
    satellite_ds : xarray.Dataset
        A satellite data time series containing an "ndwi" water index
        variable.
    valid_mask : xr.DataArray, optional
        A boolean mask used to optionally constrain the analysis area,
        with the same spatial dimensions as `satellite_ds`. For example,
        this could be a mask generated from a topo-bathy DEM, used to
        limit the analysis to likely intertidal pixels. Default is None,
        which will not apply a mask.
    ndwi_thresh : float, optional
        A threshold value for the normalized difference water index
        (NDWI) above which pixels are considered water, by default 0.1.
    min_freq, max_freq : float, optional
        Minimum and maximum frequency of wetness required for a pixel to
        be included in the analysis, by default 0.01 and 0.99.
    min_correlation : float, optional
        Minimum correlation between water index and tide height required
        for a pixel to be included in the analysis, by default 0.15.
    windows_n : int, optional
        Number of rolling windows to iterate over in the per-pixel
        rolling median calculation, by default 100
    window_prop_tide : float, optional
        Proportion of the tide range to use for each window radius in
        the per-pixel rolling median calculation, by default 0.15
    correct_seasonality : bool, optional
        If True, remove any seasonal signal from the tide height data
        by subtracting monthly mean tide height from each value prior to
        correlation calculations. This can reduce false tide correlations
        in regions where tide heights correlate with seasonal changes in
        surface water. Note that seasonally corrected tides are only used
        to identify potentially tide influenced pixels - not for elevation
        modelling itself.
    max_workers : int, optional
        Maximum number of worker processes to use for parallel execution
        in the per-pixel rolling median calculation. Defaults to None,
        which uses built-in methods from `concurrent.futures` to
        determine workers.
    tide_model : str, optional
        The tide model or a list of models used to model tides, as
        supported by the `eo-tides` Python package. Options include:
        - "EOT20" (default)
        - "TPXO10-atlas-v2-nc"
        - "FES2022"
        - "FES2022_extrapolated"
        - "FES2014"
        - "FES2014_extrapolated"
        - "GOT5.6"
        - "ensemble" (experimental: combine all above into single ensemble)
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `eo-tides.utils.list_models`.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include `cutoff` (used to
        extrapolate modelled tides away from the coast; defaults to
        `np.inf`), `crop` (whether to crop tide model constituent files
        on-the-fly to improve performance) etc.

    Returns
    -------
    ds : xarray.Dataset
        A dataset containing intertidal elevation and
        confidence values for each pixel in the study area.
    ds_aux : xarray.Dataset
        A dataset containg auxiliary layers used for subsequent
        workflows and debugging. These include information about the
        frequency of inundation for each pixel, correlations between
        NDWI and tide height, the number of misclassified observations
        resulting from the modelled elevation value, and the intertidal
        candidate pixels passed to the elevation modelling code.
    tide_m : xarray.DataArray
        An xarray.DataArray object containing the modeled tide
        heights for each pixel in the study area.

    """
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id

    # Model tides into every pixel in the three-dimensional satellite
    # dataset (x by y by time). If `model` is "ensemble" this will model
    # tides by combining the best local tide models.
    log.info(f"{run_id}: Modelling tide heights for each pixel")
    tide_m = pixel_tides(
        data=satellite_ds,
        model=tide_model,
        directory=tide_model_dir,
        **model_tides_kwargs,
    )

    # Set tide array pixels to nodata if the satellite data array pixels
    # contain nodata. This ensures that we ignore any tide observations
    # where we don't have matching satellite imagery
    log.info(f"{run_id}: Masking nodata and adding tide heights to satellite data array")
    satellite_ds["tide_m"] = tide_m.where(~satellite_ds.to_array().isel(variable=0).isnull().drop_vars("variable"))

    # Flatten array from 3D (time, y, x) to 2D (time, z) and drop pixels
    # with no correlation with tide. This greatly improves processing
    # time by ensuring only a narrow strip of tidally influenced pixels
    # along the coast are analysed, rather than the entire study area.
    # (This step is later reversed using the `flat_to_ds` function)
    log.info(f"{run_id}: Flattening satellite data array and filtering to intertidal candidate pixels")
    if valid_mask is not None:
        log.info(f"{run_id}: Applying valid data mask to constrain study area")
    flat_ds, freq, corr, clear = ds_to_flat(
        satellite_ds,
        min_freq=min_freq,
        max_freq=max_freq,
        min_correlation=min_correlation,
        correct_seasonality=correct_seasonality,
        valid_mask=valid_mask,
    )

    # Calculate per-pixel rolling median.
    log.info(f"{run_id}: Running per-pixel rolling median")
    interval_ds = pixel_rolling_median(
        flat_ds,
        windows_n=windows_n,
        window_prop_tide=window_prop_tide,
        max_workers=max_workers,
    )

    # Model intertidal elevation
    log.info(f"{run_id}: Modelling intertidal elevation")
    flat_dem = pixel_dem(interval_ds, ndwi_thresh)

    # Model intertidal elevation uncertainty
    log.info(f"{run_id}: Modelling intertidal uncertainty")
    (
        elevation_low,
        elevation_high,
        elevation_uncertainty,
        misclassified,
    ) = pixel_uncertainty(flat_ds, flat_dem, ndwi_thresh)

    # Add uncertainty array to dataset
    # TODO: decide whether we want to also keep low and high bounds
    flat_dem["elevation_uncertainty"] = elevation_uncertainty

    # Combine QA layers with elevation layers. Using `xr.combine_by_coords`
    # is required because each of our QA layers have different lengths/
    # coordinates along the "z" dimension
    flat_combined = xr.combine_by_coords(
        [
            flat_dem,  # DEM data
            freq,  # Frequency
            corr,  # Correlation
            clear,  # Clear count
        ],
    )

    # Unstack all layers back into their original spatial dimensions
    log.info(f"{run_id}: Unflattening data back to its original spatial dimensions")
    ds = flat_to_ds(flat_combined, satellite_ds)

    # Clean upper edge of intertidal zone in elevation layers
    # (likely to be inaccurate edge pixels)
    log.info(f"{run_id}: Cleaning inaccurate upper intertidal pixels")
    elevation_bands = [d for d in ds.data_vars if "elevation" in d]
    ds[elevation_bands] = clean_edge_pixels(ds[elevation_bands])

    # Return output data and tide height array
    log.info(f"{run_id}: Successfully completed intertidal elevation modelling")
    return ds, tide_m


@click.command()
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing a GridSpec tile ID (e.g. in the form 'x123y123') to run the analysis on.",
)
@click.option(
    "--start_date",
    type=str,
    required=True,
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Intertidal, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--end_date",
    type=str,
    required=True,
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Intertidal, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--label_date",
    type=str,
    required=True,
    help="The date used to label output arrays, and to use as the date "
    "assigned to the dataset when indexed into Datacube.",
)
@click.option(
    "--output_version",
    type=str,
    required=True,
    help="The version number to use for output files and metadata (e.g. '0.0.1').",
)
@click.option(
    "--output_dir",
    type=str,
    default="data/processed/",
    help="The directory/location to output data and metadata; supports "
    "both local disk and S3 locations. Defaults to 'data/processed/'.",
)
@click.option(
    "--product_maturity",
    type=str,
    default="stable",
    help="Product maturity metadata to use for the output dataset. Defaults to 'stable', can also be 'provisional'.",
)
@click.option(
    "--dataset_maturity",
    type=str,
    default="final",
    help="Dataset maturity metadata to use for the output dataset. Defaults to 'final', can also be 'interim'.",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="The spatial resolution in metres used to load satellite "
    "data and produce intertidal outputs. Defaults to 10 metre "
    "Sentinel-2 resolution.",
)
@click.option(
    "--ndwi_thresh",
    type=float,
    default=0.1,
    help="NDWI threshold used to identify the transition from dry to "
    "wet in the intertidal elevation calculation. Defaults to 0.1, "
    "which typically captures this transition more reliably than 0.0.",
)
@click.option(
    "--min_freq",
    type=float,
    default=0.01,
    help="Minimum frequency of wetness required for a pixel to be included in the analysis, by default 0.01.",
)
@click.option(
    "--max_freq",
    type=float,
    default=0.99,
    help="Maximum frequency of wetness required for a pixel to be included in the analysis, by default 0.99.",
)
@click.option(
    "--min_correlation",
    type=float,
    default=0.15,
    help="Minimum correlation between water index and tide height "
    "required for a pixel to be included in the analysis, by default "
    "0.15.",
)
@click.option(
    "--windows_n",
    type=int,
    default=100,
    help="Number of rolling windows to iterate over in the per-pixel rolling median calculation, by default 100.",
)
@click.option(
    "--window_prop_tide",
    type=float,
    default=0.15,
    help="Proportion of the tide range to use for each window radius "
    "in the per-pixel rolling median calculation, by default 0.15.",
)
@click.option(
    "--correct_seasonality/--no-correct_seasonality",
    is_flag=True,
    default=False,
    help="If True, remove any seasonal signal from the tide height data "
    "by subtracting monthly mean tide height from each value prior to "
    "correlation calculations. This can reduce false tide correlations "
    "in regions where tide heights correlate with seasonal changes in "
    "surface water. Note that seasonally corrected tides are only used "
    "to identify potentially tide influenced pixels - not for elevation "
    "modelling itself.",
)
@click.option(
    "--tide_model",
    type=str,
    multiple=True,
    default=["EOT20"],
    help="The model used for tide modelling, as supported by the "
    "`eo-tides` Python package. Options include 'EOT20' (default), "
    "'TPXO10-atlas-v2-nc', 'FES2022', 'FES2014', 'GOT5.6', 'ensemble'.",
)
@click.option(
    "--tide_model_dir",
    type=str,
    default="/var/share/tide_models",
    help="The directory containing tide model data files. Defaults to "
    "'/var/share/tide_models'; for more information about the required "
    "directory structure, refer to `eo-tides.utils.list_models`.",
)
@click.option(
    "--modelled_freq",
    type=str,
    default="30min",
    help="The frequency at which to model tides across the entire "
    "analysis period as inputs to the exposure, LAT (lowest "
    "astronomical tide), HAT (highest astronomical tide), and "
    "spread/offset calculations. Defaults to '30min' which will "
    "generate a timestep every 30 minutes between 'start_date' and "
    "'end_date'.",
)
@click.option(
    "--exposure_offsets/--no-exposure_offsets",
    is_flag=True,
    default=True,
    help="Whether to run the Exposure and spread/offsets/tidelines "
    "steps of the Intertidal workflow. Defaults to True; can be set "
    "to False by passing `--no-exposure_offsets`.",
)
@click.option(
    "--aws_unsigned/--no-aws_unsigned",
    is_flag=True,
    default=True,
    help="Whether to sign AWS requests for S3 access. Defaults to "
    "True; can be set to False by passing `--no-aws_unsigned`.",
)
def intertidal_cli(
    study_area,
    start_date,
    end_date,
    label_date,
    output_version,
    output_dir,
    product_maturity,
    dataset_maturity,
    resolution,
    ndwi_thresh,
    min_freq,
    max_freq,
    min_correlation,
    windows_n,
    window_prop_tide,
    correct_seasonality,
    tide_model,
    tide_model_dir,
    modelled_freq,
    exposure_offsets,
    aws_unsigned,
):
    # Attempt to import datacube and raise an error if not available
    try:
        import datacube
        from datacube.utils.aws import configure_s3_access
    except ImportError as e:
        msg = (
            "The DEA Intertidal CLI is configured for Australian applications, and "
            "requires `datacube`. Please install DEA Intertidal with the "
            "`[datacube]` extra, e.g.: `pip install dea-intertidal[datacube]`"
        )
        raise ImportError(msg) from e

    # Create a unique run ID for analysis based on input params and use
    # for logs
    input_params = locals()
    run_id = f"[{output_version}] [{label_date}] [{study_area}]"
    log = configure_logging(run_id)

    # Record params in logs
    log.info(f"{run_id}: Using parameters {input_params}")

    # Configure S3
    configure_s3_access(cloud_defaults=True, aws_unsigned=aws_unsigned)

    try:
        log.info(f"{run_id}: Loading satellite data")

        # Create local dask cluster to improve data load time
        client = create_local_dask_cluster(return_client=True)

        # Connect to datacube to load data
        dc = datacube.Datacube(app="Intertidal_CLI")

        # Use a custom polygon if in testing mode
        if study_area == "testing":
            log.info(f"{run_id}: Running in testing mode using custom study area")
            geom = BoundingBox(467510, -1665790, 468260, -1664840, crs="EPSG:3577").polygon
        else:
            geom = None

        # Load satellite data and dataset IDs for metadata
        satellite_ds, dss_s2, dss_ls = load_data(
            dc=dc,
            study_area=study_area,
            geom=geom,
            time_range=(start_date, end_date),
            resolution=resolution,
            crs="EPSG:3577",
            include_s2=True,
            include_ls=True,
            filter_gqa=True,
            max_cloudcover=90,
            skip_broken_datasets=True,
            dataset_maturity="final",
        )
        log.info(f"{run_id}: Loading {len(satellite_ds.time)} satellite data timesteps")
        satellite_ds.load()

        # Load topobathy mask from GA's AusBathyTopo 250m 2023 Grid,
        # urban land use class mask from ABARES CLUM, and coastal mask
        # from least-cost connectivity analysis
        topobathy_mask = load_topobathy_mask(dc, satellite_ds.odc.geobox)
        urban_mask = load_aclum_mask(dc, satellite_ds.odc.geobox)
        coastal_mask, coastal_connectivity = load_connectivity_mask(
            dc, satellite_ds.odc.geobox, add_mangroves=True, correct_hat=True
        )

        # Also load ancillary dataset IDs to use in metadata
        # (both layers are continental continental products with only
        # a single dataset, so no need for a spatial/temporal query)
        dss_ancillary = dc.find_datasets(product=["ga_ausbathytopo250m_2023", "abares_clum_2020"])

        # Calculate elevation
        log.info(f"{run_id}: Calculating Intertidal Elevation")
        ds, tide_m = elevation(
            satellite_ds,
            valid_mask=topobathy_mask & coastal_mask,
            ndwi_thresh=ndwi_thresh,
            min_freq=min_freq,
            max_freq=max_freq,
            min_correlation=min_correlation,
            windows_n=windows_n,
            window_prop_tide=window_prop_tide,
            correct_seasonality=correct_seasonality,
            tide_model=tide_model,
            tide_model_dir=tide_model_dir,
            run_id=run_id,
            log=log,
        )

        # Calculate extents (to be included in next version)
        log.info(f"{run_id}: Calculating Intertidal Extents")
        ds["extents"] = extents(
            dem=ds.elevation,
            freq=ds.qa_ndwi_freq,
            corr=ds.qa_ndwi_corr,
            coastal_mask=coastal_mask,
            urban_mask=urban_mask,
        )

        # Add coastal connectivity output layer
        ds["qa_coastal_connectivity"] = coastal_connectivity.where(coastal_connectivity < 65535)

        if exposure_offsets:
            log.info(f"{run_id}: Calculating Intertidal Exposure")

            # Calculate exposure
            exposure_ds, modelledtides_ds = exposure(
                dem=ds.elevation,
                start_date=start_date,
                end_date=end_date,
                modelled_freq=modelled_freq,
                tide_model=tide_model,
                tide_model_dir=tide_model_dir,
            )

            # Write the unfiltered exposure output as new variable in the main dataset
            ds["exposure"] = exposure_ds["unfiltered"]

            # Translate unfiltered exposure outputs to match continental
            # product suite
            modelledtides_ds = modelledtides_ds["unfiltered"]

            # Calculate spread, offsets and HAT/LAT/LOT/HOT
            log.info(f"{run_id}: Calculating spread, offset and HAT/LAT/LOT/HOT layers")
            (
                ds["ta_lat"],
                ds["ta_hat"],
                ds["ta_lot"],
                ds["ta_hot"],
                ds["ta_spread"],
                ds["ta_offset_low"],
                ds["ta_offset_high"],
            ) = bias_offset(
                tide_m=tide_m,
                tide_cq=modelledtides_ds,
                lot_hot=True,
                lat_hat=True,
            )

        else:
            log.info(f"{run_id}: Skipping Exposure and spread/offsets calculation")

        # Prepare data for export
        ds["qa_ndwi_freq"] *= 100  # Convert frequency to %
        ds_prepared = prepare_for_export(ds)  # sets correct dtypes and nodata

        # Calculate additional tile-level tidal metadata attributes and graph
        metadata_dict, tide_graph_fig = tidal_metadata(
            product_family="intertidal",
            data=satellite_ds,
            modelled_freq=modelled_freq,
            model=tide_model,
            directory=tide_model_dir,
        )

        # Export data and metadata
        export_dataset_metadata(
            ds_prepared,
            year=label_date,
            study_area=study_area,
            output_location=output_dir,
            ls_lineage=dss_ls,
            s2_lineage=dss_s2,
            ancillary_lineage=dss_ancillary,
            dataset_version=output_version,
            product_maturity=product_maturity,
            dataset_maturity=dataset_maturity,
            tide_graph_fig=tide_graph_fig,
            additional_metadata=metadata_dict,
            run_id=run_id,
            log=log,
        )

        # Workflow completed; close Dask client
        client.close()
        log.info(f"{run_id}: Completed DEA Intertidal workflow")

    except Exception as e:
        log.exception(f"{run_id}: Failed to run process with error {e}")
        sys.exit(1)


if __name__ == "__main__":
    intertidal_cli()
