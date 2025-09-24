import eodatasets3.validate
import pytest
from click.testing import CliRunner

from intertidal.composites import tidal_composites_cli


@pytest.mark.dependency
def test_tidal_composites_cli():
    """This test runs the DEA Tidal Composites CLI
    from start to finish, and will fail if any
    error is raised.
    """
    runner = CliRunner()
    result = runner.invoke(
        tidal_composites_cli,
        [
            "--study_area",
            "testing",
            "--start_date",
            "2020",
            "--label_date",
            "2021",
            "--end_date",
            "2022",
            "--output_version",
            "0.0.1",
            "--threshold_lowtide",
            "0.15",
            "--threshold_hightide",
            "0.85",
            "--tide_model",
            "FES2014",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_tidal_composites_cli"])
def test_validate_composites_metadata():
    """Validates output EO3 metadata against product definition and metadata type.
    This will detect issues like incorrect datatypes, band names, nodata
    or missing bands.
    """
    runner = CliRunner()
    result = runner.invoke(
        eodatasets3.validate.run,
        [
            "metadata/ga_s2_tidal_composites_cyear_3.odc-product.yaml",
            "metadata/eo3_intertidal.odc-type.yaml",
            "data/processed/ga_s2_tidal_composites_cyear_3/0-0-1/tes/ting/2021--P1Y/ga_s2_tidal_composites_cyear_3_testing_2021--P1Y_final.odc-metadata.yaml",
            "--thorough",
        ],
    )

    # Return useful exception from eodatasets if error
    if result.exit_code != 0:
        raise Exception(result.output)
