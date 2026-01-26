from pathlib import Path

import pytest

from promptlab.infrastructure.yaml_config_loader import (
    YamlConfigLoader,
    YamlConfigLoaderError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_EXPERIMENT = FIXTURES_DIR / "sample-experiment"
SAMPLE_VARIANT = SAMPLE_EXPERIMENT / "v1"


@pytest.fixture
def loader() -> YamlConfigLoader:
    return YamlConfigLoader()


def test_load_experiment_loads_config(loader: YamlConfigLoader):
    config = loader.load_experiment(SAMPLE_EXPERIMENT)

    assert config.name == "sample-experiment"
    assert config.description == "A sample experiment for testing"
    assert "openai:gpt-4o" in config.models
    assert "anthropic:claude-sonnet-4-20250514" in config.models


def test_load_experiment_raises_when_no_file(loader: YamlConfigLoader, tmp_path: Path):
    with pytest.raises(YamlConfigLoaderError, match="experiment.md not found"):
        loader.load_experiment(tmp_path)


def test_load_experiment_raises_when_no_models(
    loader: YamlConfigLoader, tmp_path: Path
):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text("---\nname: test\n---\n")

    with pytest.raises(YamlConfigLoaderError, match="No models specified"):
        loader.load_experiment(tmp_path)


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        ("", 5),  # Default when not specified
        ("runs: 10\n", 10),  # Explicit value
    ],
)
def test_load_experiment_runs(
    loader: YamlConfigLoader, tmp_path: Path, yaml_value: str, expected: int
):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text(
        f"---\nname: test\nmodels: [openai:gpt-4o]\n{yaml_value}---\n"
    )

    config = loader.load_experiment(tmp_path)
    assert config.runs == expected


def test_load_prompt_loads_content(loader: YamlConfigLoader):
    config = loader._load_prompt(SAMPLE_VARIANT)

    assert "helpful assistant" in config.content
    assert "{{ message }}" in config.content


def test_load_prompt_raises_when_no_file(loader: YamlConfigLoader, tmp_path: Path):
    with pytest.raises(YamlConfigLoaderError, match="prompt.md not found"):
        loader._load_prompt(tmp_path)


def test_load_judge_loads_from_experiment(loader: YamlConfigLoader):
    config = loader._load_judge(SAMPLE_VARIANT, SAMPLE_EXPERIMENT)

    assert config.model == "openai:gpt-4o"
    assert config.score_range == (1, 10)
    assert "evaluating" in config.content.lower()


def test_load_judge_raises_when_no_judge(loader: YamlConfigLoader, tmp_path: Path):
    with pytest.raises(YamlConfigLoaderError, match="No judge.md found"):
        loader._load_judge(tmp_path, tmp_path)


def test_load_inputs_loads_cases(loader: YamlConfigLoader):
    inputs = loader._load_inputs(SAMPLE_VARIANT, SAMPLE_EXPERIMENT)

    assert len(inputs) == 2
    assert inputs[0].id == "greeting-1"
    assert inputs[0].data["message"] == "Hello!"
    assert inputs[1].id == "greeting-2"


@pytest.mark.parametrize("create_file", [False, True])
def test_load_inputs_returns_default_when_missing_or_empty(
    loader: YamlConfigLoader, tmp_path: Path, create_file: bool
):
    if create_file:
        (tmp_path / "inputs.yaml").write_text("")

    inputs = loader._load_inputs(tmp_path, tmp_path)

    assert len(inputs) == 1
    assert inputs[0].id == "default"
    assert inputs[0].data == {}


def test_load_inputs_with_runs_override(loader: YamlConfigLoader, tmp_path: Path):
    inputs_file = tmp_path / "inputs.yaml"
    inputs_file.write_text(
        "- id: test-1\n  message: Hello\n  runs: 5\n- id: test-2\n  message: Hi\n"
    )

    inputs = loader._load_inputs(tmp_path, tmp_path)

    assert len(inputs) == 2
    assert inputs[0].id == "test-1"
    assert inputs[0].runs == 5
    assert inputs[0].data["message"] == "Hello"
    assert inputs[1].id == "test-2"
    assert inputs[1].runs is None
    assert inputs[1].data["message"] == "Hi"


def test_load_variant_loads_complete(loader: YamlConfigLoader):
    variant = loader.load_variant(SAMPLE_VARIANT)

    assert variant.experiment.name == "sample-experiment"
    assert "{{ message }}" in variant.prompt.content
    assert len(variant.inputs) == 2
    assert variant.judge.model == "openai:gpt-4o"
    assert "openai:gpt-4o" in variant.models


def test_discover_variants_finds_all(loader: YamlConfigLoader):
    variants = loader.discover_variants(SAMPLE_EXPERIMENT)

    assert len(variants) == 2
    variant_names = [v.name for v in variants]
    assert "v1" in variant_names
    assert "v2" in variant_names


def test_discover_variants_raises_when_none(loader: YamlConfigLoader, tmp_path: Path):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text("---\nname: test\nmodels: [openai:gpt-4o]\n---\n")

    with pytest.raises(YamlConfigLoaderError, match="No variants found"):
        loader.discover_variants(tmp_path)
