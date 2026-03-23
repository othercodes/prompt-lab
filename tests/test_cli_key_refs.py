from typer.testing import CliRunner

from promptlab.cli import app

runner = CliRunner()


def test_key_ref_should_fail_when_no_colon(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--key-ref", "INVALID_FORMAT"])

    assert result.exit_code == 1
    assert "Invalid --key-ref format" in result.output


def test_key_ref_should_fail_when_unknown_provider(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--key-ref", "gemini:MY_KEY"])

    assert result.exit_code == 1
    assert "Unknown provider" in result.output
    assert "gemini" in result.output


def test_key_ref_should_show_available_providers_when_unknown(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--key-ref", "gemini:MY_KEY"])

    assert "openai" in result.output
    assert "anthropic" in result.output


def test_key_ref_should_parse_multiple_refs(tmp_path):
    """Multiple --key-ref flags are accepted without parsing errors."""
    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--key-ref",
            "openai:MY_OPENAI",
            "--key-ref",
            "anthropic:MY_ANTHROPIC",
        ],
    )

    assert "Invalid --key-ref format" not in result.output
    assert "Unknown provider" not in result.output


def test_key_ref_should_work_when_short_flag(tmp_path):
    """The -k short flag works the same as --key-ref."""
    result = runner.invoke(app, ["run", str(tmp_path), "-k", "openai:MY_KEY"])

    assert "Invalid --key-ref format" not in result.output
    assert "Unknown provider" not in result.output
