import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from eo_tides.stats import tide_stats


def bias_offset(tide_m, tide_cq, lat_hat=True, lot_hot=None):
    """
    Calculate the pixel-based sensor-observed spread and high/low
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
        else:
            return lat, hat, spread, offset_lowtide, offset_hightide
    elif lot_hot:
        return lot, hot, spread, offset_lowtide, offset_hightide
    else:
        return spread, offset_lowtide, offset_hightide


def generate_tide_graph(data, modelled_freq, model, directory):

    tide_stats(
        data=data,
        modelled_freq=modelled_freq,
        model=model,
        directory=directory,
        plain_english=False,
    )
    fig = plt.gcf()
    
    # Update line and point colours
    fig.axes[0].get_lines()[0].set_color("#90b7d8")
    fig.axes[0].get_lines()[0].set_alpha(1.0)
    fig.axes[0].get_lines()[1].set_color("black")
    fig.axes[0].get_lines()[1].set_markersize(4)
    fig.axes[0].get_lines()[1].set_markeredgecolor("none")
    
    # Set background to transparent
    fig.patch.set_facecolor("#5d646c00")
    fig.axes[0].set_facecolor("#5d646c00")
    
    # Set spines and axis labels to white
    for spine in fig.axes[0].spines.values():
        spine.set_edgecolor("#ffffff")
    fig.axes[0].tick_params(axis="both", colors="#ffffff")
    fig.axes[0].yaxis.label.set_color("#ffffff")
    
    # Update the legend
    legend = fig.axes[0].get_legend()
    legend.remove()
    fig.axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=20,
        borderaxespad=0,
        frameon=False,
        labelcolor="white",
    )
    
    fig.set_size_inches(8, 2.5)
    return fig
    # fig.savefig(output_path, bbox_inches="tight")
