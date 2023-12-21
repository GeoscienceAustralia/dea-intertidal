import pytz
import pytest
import pickle
import datetime
import rioxarray
import numpy as np
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
    accuracy_df = eval_metrics(x=validation_z, y=modelled_z, round=3)

    # Assert accuracy is within tolerance
    assert accuracy_df.Correlation > 0.9
    assert accuracy_df.RMSE < 0.25
    assert accuracy_df.MAE < 0.2
    assert accuracy_df["R-squared"] > 0.8
    assert accuracy_df.Bias < 0.2
    assert abs(accuracy_df["Regression slope"] - 1) < 0.1

    # Plot and compare - heatmap
    plt.figure(figsize=(5, 5))
    lim_min, lim_max = np.percentile(
        np.concatenate([validation_z, modelled_z]), [1, 99]
    )
    lim_min -= 0.1
    lim_max += 0.1
    sns.kdeplot(
        x=validation_z,
        y=modelled_z,
        cmap="inferno",
        fill=True,
        ax=plt.gca(),
        thresh=0,
        bw_adjust=0.4,
        levels=30,
    )
    plt.gca().set_facecolor("black")
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", c="white")
    plt.margins(x=0, y=0)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.xlabel("Validation (m)")
    plt.ylabel("Modelled (m)")

    # Add title
    current_time = datetime.datetime.now(pytz.timezone("Australia/Canberra")).strftime(
        "%Y-%m-%d %H:%M"
    )
    plt.title(f"DEA Intertidal Elevation validation\n(Last run: {current_time})")

    # Add stats annotation
    plt.gca().annotate(
        f'Correlation: {accuracy_df["Correlation"]:.2f}\n'
        f'R-squared: {accuracy_df["R-squared"]:.2f}\n'
        f'RMSE: {accuracy_df["RMSE"]:.2f} m\n'
        f'MAE: {accuracy_df["MAE"]:.2f} m\n'
        f'Bias: {accuracy_df["Bias"]:.2f} m\n'
        f'Slope: {accuracy_df["Regression slope"]:.2f}\n',
        xy=(0.04, 0.7),
        fontsize=9,
        xycoords="axes fraction",
        color="white",
    )

    # Write into mounted artifacts directory
    plt.savefig(f"artifacts/validation.jpg", dpi=150, bbox_inches="tight")


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
