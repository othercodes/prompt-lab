from typer.testing import CliRunner

from promptlab import __version__
from promptlab.cli import app

runner = CliRunner()


def test_version_should_display_version_when_long_flag():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"prompt-lab {__version__}" in result.output


def test_version_should_display_version_when_short_flag():
    result = runner.invoke(app, ["-v"])

    assert result.exit_code == 0
    assert f"prompt-lab {__version__}" in result.output
