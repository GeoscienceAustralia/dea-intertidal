import pytest
from click.testing import CliRunner
from intertidal.elevation import intertidal_cli, test_func


@pytest.mark.dependency()
def test_intertidal_cli():
    runner = CliRunner()
    result = runner.invoke(
        intertidal_cli,
        [
            "--help",
        ],
    )
    assert result.exit_code == 0


def test_sample_func():
    assert sample_func(1) == 1
