import pytest
from click.testing import CliRunner
from intertidal.elevation import intertidal_cli


@pytest.fixture()
def satellite_ds():
    with open("tests/data/satellite_ds.pickle", "rb") as handle:
        return pickle.load(handle)


def test_intertidal_cli():
    runner = CliRunner()
    result = runner.invoke(
        intertidal_cli,
        [
            "--help",
        ],
    )
    assert result.exit_code == 0
    

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
    
    assert "time" not in ds.dims
    assert "elevation" in ds.data_vars 
    assert "elevation_uncertainty" in ds.data_vars 
