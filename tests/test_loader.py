from pathlib import Path

import pytest

from promptlab.loader import (
    LoaderError,
    discover_variants,
    load_experiment,
    load_inputs,
    load_judge,
    load_prompt,
    load_variant,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_EXPERIMENT = FIXTURES_DIR / "sample-experiment"
SAMPLE_VARIANT = SAMPLE_EXPERIMENT / "v1"


def test_load_experiment_loads_config():
    config = load_experiment(SAMPLE_EXPERIMENT)

    assert config.name == "sample-experiment"
    assert config.description == "A sample experiment for testing"
    assert "openai:gpt-4o" in config.models
    assert "anthropic:claude-sonnet-4-20250514" in config.models


def test_load_experiment_raises_when_no_file(tmp_path: Path):
    with pytest.raises(LoaderError, match="experiment.md not found"):
        load_experiment(tmp_path)


def test_load_experiment_raises_when_no_models(tmp_path: Path):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text("---\nname: test\n---\n")

    with pytest.raises(LoaderError, match="No models specified"):
        load_experiment(tmp_path)


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        ("", 5),  # Default when not specified
        ("runs: 10\n", 10),  # Explicit value
    ],
)
def test_load_experiment_runs(tmp_path: Path, yaml_value: str, expected: int):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text(
        f"---\nname: test\nmodels: [openai:gpt-4o]\n{yaml_value}---\n"
    )

    config = load_experiment(tmp_path)
    assert config.runs == expected


def test_load_prompt_loads_content():
    config = load_prompt(SAMPLE_VARIANT)

    assert "helpful assistant" in config.content
    assert "{{ message }}" in config.content


def test_load_prompt_raises_when_no_file(tmp_path: Path):
    with pytest.raises(LoaderError, match="prompt.md not found"):
        load_prompt(tmp_path)


def test_load_judge_loads_from_experiment():
    config = load_judge(SAMPLE_VARIANT, SAMPLE_EXPERIMENT)

    assert config.model == "openai:gpt-4o"
    assert config.score_range == (1, 10)
    assert "evaluating" in config.content.lower()


def test_load_judge_raises_when_no_judge(tmp_path: Path):
    with pytest.raises(LoaderError, match="No judge.md found"):
        load_judge(tmp_path, tmp_path)


def test_load_inputs_loads_cases():
    inputs = load_inputs(SAMPLE_VARIANT, SAMPLE_EXPERIMENT)

    assert len(inputs) == 2
    assert inputs[0].id == "greeting-1"
    assert inputs[0].data["message"] == "Hello!"
    assert inputs[1].id == "greeting-2"


@pytest.mark.parametrize("create_file", [False, True])
def test_load_inputs_returns_default_when_missing_or_empty(
    tmp_path: Path, create_file: bool
):
    if create_file:
        (tmp_path / "inputs.yaml").write_text("")

    inputs = load_inputs(tmp_path, tmp_path)

    assert len(inputs) == 1
    assert inputs[0].id == "default"
    assert inputs[0].data == {}


def test_load_inputs_with_runs_override(tmp_path: Path):
    inputs_file = tmp_path / "inputs.yaml"
    inputs_file.write_text(
        "- id: test-1\n  message: Hello\n  runs: 5\n- id: test-2\n  message: Hi\n"
    )

    inputs = load_inputs(tmp_path, tmp_path)

    assert len(inputs) == 2
    assert inputs[0].id == "test-1"
    assert inputs[0].runs == 5
    assert inputs[0].data["message"] == "Hello"
    assert inputs[1].id == "test-2"
    assert inputs[1].runs is None
    assert inputs[1].data["message"] == "Hi"


def test_load_variant_loads_complete():
    variant = load_variant(SAMPLE_VARIANT)

    assert variant.experiment.name == "sample-experiment"
    assert "{{ message }}" in variant.prompt.content
    assert len(variant.inputs) == 2
    assert variant.judge.model == "openai:gpt-4o"
    assert "openai:gpt-4o" in variant.models


def test_discover_variants_finds_all():
    variants = discover_variants(SAMPLE_EXPERIMENT)

    assert len(variants) == 2
    variant_names = [v.name for v in variants]
    assert "v1" in variant_names
    assert "v2" in variant_names


def test_discover_variants_raises_when_none(tmp_path: Path):
    experiment_file = tmp_path / "experiment.md"
    experiment_file.write_text("---\nname: test\nmodels: [openai:gpt-4o]\n---\n")

    with pytest.raises(LoaderError, match="No variants found"):
        discover_variants(tmp_path)
