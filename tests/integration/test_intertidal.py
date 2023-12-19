import pytest
from click.testing import CliRunner
from intertidal.elevation import intertidal_cli

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