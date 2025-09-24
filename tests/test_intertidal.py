import os

import eodatasets3.validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rioxarray
import seaborn as sns
import xarray as xr
from click.testing import CliRunner
from dea_tools.datahandling import load_reproject
from dea_tools.validation import eval_metrics
from mdutils import Html
from mdutils.mdutils import MdUtils

from intertidal.elevation import elevation, intertidal_cli
from intertidal.validation import preprocess_validation


@pytest.fixture
def satellite_ds():
    """Loads a pre-generated timeseries of satellite data from NetCDF.
    This is used by the `test_elevation` test below.
    """
    satellite_ds = xr.open_dataset("tests/data/satellite_ds.nc")

    # Hack to fix malformed CRS
    del satellite_ds["spatial_ref"]
    satellite_ds = satellite_ds.odc.assign_crs("EPSG:3577")

    return satellite_ds


@pytest.mark.dependency
def test_intertidal_cli():
    """This test runs the DEA Intertidal CLI
    from start to finish, and will fail if any
    error is raised.
    """
    runner = CliRunner()
    result = runner.invoke(
        intertidal_cli,
        [
            "--study_area",
            "testing",
            "--start_date",
            "2020",
            "--label_date",
            "2021",
            "--end_date",
            "2022",
            "--modelled_freq",
            "3h",
            "--output_version",
            "0.0.1",
            "--tide_model",
            "FES2014",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_intertidal_cli"])
def test_dem_accuracy(
    val_path="tests/data/lidar_10m_tests.tif",
    mod_path="data/processed/ga_s2ls_intertidal_cyear_3/0-0-1/tes/ting/2021--P1Y/ga_s2ls_intertidal_cyear_3_testing_2021--P1Y_final_elevation.tif",
    input_csv="tests/validation.csv",
    output_csv="tests/validation.csv",
    output_plot="tests/validation.jpg",
    output_md="tests/README.md",
):
    """Compares elevation outputs of the previous CLI step against
    validation data, and calculates and evaluates a range of accuracy
    metrics.
    """
    # Load validation data
    validation_da = rioxarray.open_rasterio(val_path, masked=True).squeeze("band")

    # Load modelled elevation and uncertainty data
    modelled_da = load_reproject(
        path=mod_path,
        how=validation_da.odc.geobox,
        resampling="average",
    )

    # Preprocess and calculate accuracy statistics
    validation_z, modelled_z, _ = preprocess_validation(
        validation_da, modelled_da, modelled_da, lat=-5, hat=5, clean_slope=False
    )
    accuracy_metrics = eval_metrics(x=validation_z, y=modelled_z, round=3)

    # Assert accuracy is within tolerance
    # (these are intended to only catch *major* regressions - smaller
    # changes in accuracy can be reviewed on the generated plots)
    assert accuracy_metrics.Correlation > 0.8
    assert accuracy_metrics.RMSE < 0.30
    assert accuracy_metrics.MAE < 0.25
    assert accuracy_metrics["R-squared"] > 0.7
    assert accuracy_metrics.Bias < 0.25
    assert abs(accuracy_metrics["Regression slope"] - 1) < 0.15

    #########
    # Plots #
    #########

    # Transpose and add index time and prefix name
    accuracy_df = pd.DataFrame({pd.to_datetime("now", utc=True): accuracy_metrics}).T
    accuracy_df.index.name = "time"

    # Append results to file, and re-read stats from disk to ensure we get
    # older results
    accuracy_df.to_csv(
        input_csv,
        mode="a",
        header=(not os.path.exists(input_csv)),
    )
    accuracy_df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

    # Convert dataframe to local time
    accuracy_df_local = accuracy_df.tz_convert(tz="Australia/Canberra")

    # Create plot and add overall title
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    latest_time = accuracy_df_local.index[-1].strftime("%Y-%m-%d %H:%M")
    plt.suptitle(
        f"Latest DEA Intertial Elevation integration test validation ({latest_time})",
        size=14,
        fontweight="bold",
        y=1.0,
    )

    ################
    # Heatmap plot #
    ################

    lim_min, lim_max = np.percentile(np.concatenate([validation_z, modelled_z]), [1, 99])
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

    # Add text (including latest accuracy annotations)
    corr, rmse, mae, r2, bias, slope = accuracy_df_local.iloc[-1]
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
    ax1.set_title("Modelled vs. validation elevation")

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
    accuracy_df_local.RMSE.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df_local.RMSE.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df_local.index, min_q, max_q, alpha=0.2)

    accuracy_df_local.MAE.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df_local.MAE.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df_local.index, min_q, max_q, alpha=0.2)

    accuracy_df_local.Bias.plot(ax=ax2, style=".-", legend=True)
    min_q, max_q = accuracy_df_local.Bias.quantile((0.1, 0.9)).values
    ax2.fill_between(accuracy_df_local.index, min_q, max_q, alpha=0.2)
    ax2.set_title("Accuracy and bias across test runs")
    ax2.set_ylabel("Metres (m)")
    ax2.set_xlabel(None)

    # Write output CSV
    accuracy_df.to_csv(output_csv)
    plt.savefig(output_plot, dpi=100, bbox_inches="tight")

    #############
    # Readme.md #
    #############

    # Create markdown report

    # Calculate recent change and convert to plain text
    accuracy_df_temp = accuracy_df_local.copy()
    accuracy_df_temp["Bias"] = accuracy_df_temp["Bias"].abs()
    recent_diff = accuracy_df_temp.diff(1).iloc[-1].to_frame("diff")
    recent_diff.loc["Correlation"] = -recent_diff.loc["Correlation"]  # Invert as higher corrs are good
    recent_diff.loc["R-squared"] = -recent_diff.loc["R-squared"]  # Invert as higher R2 are good
    recent_diff.loc[recent_diff["diff"] < 0, "prefix"] = ":heavy_check_mark: improved by "
    recent_diff.loc[recent_diff["diff"] == 0, "prefix"] = ":heavy_minus_sign: no change"
    recent_diff.loc[recent_diff["diff"] > 0, "prefix"] = ":heavy_exclamation_mark: worsened by "
    recent_diff["suffix"] = recent_diff["diff"].abs().round(3).replace({0: ""})
    recent_diff = recent_diff.prefix.astype(str) + recent_diff.suffix.astype(str).str[0:5]

    mdFile = MdUtils(file_name=output_md, title="Integration tests")
    mdFile.new_header(level=1, title="Latest results")
    mdFile.new_paragraph("> [!NOTE]")
    mdFile.new_line(
        "> *This readme is automatically generated by the ``test_dem_accuracy`` function within [``test_intertidal.py``](../tests/test_intertidal.py).*"
    )

    mdFile.new_paragraph(
        "This directory contains tests that are run to verify that DEA Intertidal code runs correctly. The ``test_intertidal.py`` file runs a small-scale full workflow analysis over an intertidal flat in the Gulf of Carpentaria using the DEA Intertidal [Command Line Interface (CLI) tools](../notebooks/Intertidal_CLI.ipynb), and compares these results against a LiDAR validation DEM to produce some simple accuracy metrics."
    )

    mdFile.new_paragraph(
        f"The latest integration test completed at **{latest_time}**. Compared to the previous run, it had an:"
    )
    items = [
        f"RMSE accuracy of **{accuracy_df_local.RMSE[-1]:.2f} m ( {recent_diff.RMSE})**",
        f"MAE accuracy of **{accuracy_df_local.MAE[-1]:.2f} m ( {recent_diff.MAE})**",
        f"Bias of **{accuracy_df_local.Bias[-1]:.2f} m ( {recent_diff.Bias})**",
        f"Pearson correlation of **{accuracy_df_local.Correlation[-1]:.3f} ( {recent_diff.Correlation})**",
    ]
    mdFile.new_list(items=items)
    mdFile.new_paragraph(Html.image(path="validation.jpg", size="950"))
    mdFile.create_md_file()


@pytest.mark.dependency(depends=["test_intertidal_cli"])
def test_validate_intertidal_metadata():
    """Validates output EO3 metadata against product definition and metadata type.
    This will detect issues like incorrect datatypes, band names, nodata
    or missing bands.
    """
    runner = CliRunner()
    result = runner.invoke(
        eodatasets3.validate.run,
        [
            "metadata/ga_s2ls_intertidal_cyear_3.odc-product.yaml",
            "metadata/eo3_intertidal.odc-type.yaml",
            "data/processed/ga_s2ls_intertidal_cyear_3/0-0-1/tes/ting/2021--P1Y/ga_s2ls_intertidal_cyear_3_testing_2021--P1Y_final.odc-metadata.yaml",
            "--thorough",
        ],
    )

    # Return useful exception from eodatasets if error
    if result.exit_code != 0:
        raise Exception(result.output)


def test_elevation(satellite_ds):
    ds, tide_m = elevation(
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
        run_id=None,
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
