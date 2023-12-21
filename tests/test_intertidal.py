import os
import pytz
import pytest
import pickle
import datetime
import rioxarray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from click.testing import CliRunner

from intertidal.elevation import intertidal_cli, elevation
from intertidal.validation import eval_metrics, map_raster, preprocess_validation
from intertidal.extents import load_reproject


@pytest.fixture()
def satellite_ds():
    """
    Loads a timeseries of satellite data from a .pickle file.
    TODO: Replace this with data loaded directly from datacube
    after adding access to prod database.
    """
    with open("tests/data/satellite_ds.pickle", "rb") as handle:
        return pickle.load(handle)


@pytest.mark.dependency()
def test_intertidal_cli():
    runner = CliRunner()
    result = runner.invoke(
        intertidal_cli,
        [
            "--study_area",
            "testing",
            "--start_date",
            "2020",
            "--end_date",
            "2022",
            "--modelled_freq",
            "3h",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_intertidal_cli"])
def test_dem_accuracy(
    val_path="tests/data/lidar_10m_tests.tif",
    mod_path="data/interim/testing/2020-2022/testing_2020_2022_elevation.tif",
    output_plot="artifacts/validation.jpg",
    output_csv="artifacts/validation.csv",
):
    """
    Compares elevation outputs of the previous CLI step against
    validation data, and calculates and evaluates a range of accuracy
    metrics.
    """
    # Load validation data
    validation_da = rioxarray.open_rasterio(val_path, masked=True).squeeze("band")

    # Load modelled elevation and uncertainty data
    modelled_da = load_reproject(
        path=mod_path,
        gbox=validation_da.odc.geobox,
        resampling="average",
    ).band_data

    # Preprocess and calculate accuracy statistics
    validation_z, modelled_z = preprocess_validation(modelled_da, validation_da, None)
    accuracy_metrics = eval_metrics(x=validation_z, y=modelled_z, round=3)

    # Assert accuracy is within tolerance
    assert accuracy_metrics.Correlation > 0.9
    assert accuracy_metrics.RMSE < 0.25
    assert accuracy_metrics.MAE < 0.2
    assert accuracy_metrics["R-squared"] > 0.8
    assert accuracy_metrics.Bias < 0.2
    assert abs(accuracy_metrics["Regression slope"] - 1) < 0.1

    #########
    # Plots #
    #########

    # Transpose and add index time and prefix name
    accuracy_df = pd.DataFrame({pd.to_datetime("now", utc=True): accuracy_metrics}).T
    accuracy_df.index.name = "time"

    # Append results to file, and re-read stats from disk to ensure we get
    # older results
    accuracy_df.to_csv(
        output_csv,
        mode="a",
        header=(not os.path.exists(output_csv)),
    )
    accuracy_df = pd.read_csv(output_csv, index_col=0, parse_dates=True)

    # Extract integration test run times and convert to local time
    times_local = accuracy_df.index.tz_convert(tz="Australia/Canberra")
    accuracy_df.index = times_local

    # Get latest stats
    corr, rmse, mae, r2, bias, slope = accuracy_df.iloc[-1]

    # Create plot and add overall title
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    latest_time = times_local[-1].strftime("%Y-%m-%d %H:%M")
    plt.suptitle(
        f"Latest DEA Intertial Elevation integration test validation ({latest_time})",
        size=14,
        fontweight="bold",
        y=1.0,
    )

    ################
    # Heatmap plot #
    ################

    lim_min, lim_max = np.percentile(
        np.concatenate([validation_z, modelled_z]), [1, 99]
    )
    lim_min -= 0.2
    lim_max += 0.2
    sns.kdeplot(
        x=validation_z,
        y=modelled_z,
        cmap="inferno",
        fill=True,
        ax=ax1,
        thresh=0,
        bw_adjust=0.4,
        levels=30,
    )

    # Add text
    ax1.annotate(
        f"Correlation: {corr:.2f}\n"
        f"R-squared: {r2:.2f}\n"
        f"RMSE: {rmse:.2f} m\n"
        f"MAE: {mae:.2f} m\n"
        f"Bias: {bias:.2f} m\n"
        f"Slope: {slope:.2f}\n",
        xy=(0.04, 0.75),
        fontsize=10,
        xycoords="axes fraction",
        color="white",
    )
    ax1.set_xlabel("Validation (m)")
    ax1.set_ylabel("Modelled (m)")
    ax1.set_title(f"Modelled vs. validation elevation")

    # Formatting
    ax1.set_facecolor("black")
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], "--", c="white")
    ax1.margins(x=0, y=0)
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)

    ###################
    # Timeseries plot #
    ###################

    # Plot all integration test accuracies and biases over time
    accuracy_df.RMSE.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df.RMSE.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df.index, min_q, max_q, alpha=0.2)

    accuracy_df.MAE.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df.MAE.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df.index, min_q, max_q, alpha=0.2)

    accuracy_df.Bias.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df.Bias.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df.index, min_q, max_q, alpha=0.2)
    ax2.set_title("Accuracy and bias across test runs")
    ax2.set_ylabel("Metres (m)")
    ax2.set_xlabel(None)

    # Write into mounted artifacts directory
    plt.savefig(output_plot, dpi=100, bbox_inches="tight")


def test_elevation(satellite_ds):
    ds, ds_aux, tide_m = elevation(
        satellite_ds,
        valid_mask=None,
        ndwi_thresh=0.1,
        min_freq=0.01,
        max_freq=0.99,
        min_correlation=0.15,
        windows_n=20,
        window_prop_tide=0.15,
        max_workers=None,
        tide_model="FES2014",
        tide_model_dir="/var/share/tide_models",
        study_area=None,
        log=None,
    )
    """
    Verify that elevation code produces expected outputs.
    """

    # Verify that ds contains correct variables
    assert "elevation" in ds.data_vars
    assert "elevation_uncertainty" in ds.data_vars

    # Verify that ds is a single layer with no time dimension
    assert "time" not in ds.dims
