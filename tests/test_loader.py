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


def _create_experiment(
    tmp_path: Path,
    judge: str = "---\nmodel: openai:gpt-4o\n---\nEvaluate.\n",
    inputs: str | None = None,
    prompt: str = "Hello {{ message }}",
    system: str | None = None,
) -> Path:
    (tmp_path / "experiment.md").write_text(
        "---\nname: test\nmodels: [openai:gpt-4o]\n---\n"
    )
    (tmp_path / "judge.md").write_text(judge)
    if inputs:
        (tmp_path / "inputs.yaml").write_text(inputs)

    variant_dir = tmp_path / "v1"
    variant_dir.mkdir()
    (variant_dir / "prompt.md").write_text(prompt)
    if system:
        (variant_dir / "system.md").write_text(system)

    return variant_dir


# --- load_experiment ---


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
    (tmp_path / "experiment.md").write_text("---\nname: test\n---\n")

    with pytest.raises(YamlConfigLoaderError, match="No models specified"):
        loader.load_experiment(tmp_path)


@pytest.mark.parametrize(
    "yaml_value,expected",
    [
        ("", 5),
        ("runs: 10\n", 10),
    ],
)
def test_load_experiment_runs(
    loader: YamlConfigLoader, tmp_path: Path, yaml_value: str, expected: int
):
    (tmp_path / "experiment.md").write_text(
        f"---\nname: test\nmodels: [openai:gpt-4o]\n{yaml_value}---\n"
    )

    config = loader.load_experiment(tmp_path)
    assert config.runs == expected


# --- load_variant ---


def test_load_variant_loads_complete(loader: YamlConfigLoader):
    variant = loader.load_variant(SAMPLE_VARIANT)

    assert variant.experiment.name == "sample-experiment"
    assert "{{ message }}" in variant.prompt.content
    assert len(variant.inputs) == 2
    assert variant.judge.model == "openai:gpt-4o"
    assert "openai:gpt-4o" in variant.models


def test_load_variant_reads_prompt(loader: YamlConfigLoader, tmp_path: Path):
    _create_experiment(tmp_path, prompt="Be a {{ role }} assistant")

    variant = loader.load_variant(tmp_path / "v1")

    assert "Be a {{ role }} assistant" in variant.prompt.content


def test_load_variant_raises_when_no_prompt(loader: YamlConfigLoader, tmp_path: Path):
    (tmp_path / "experiment.md").write_text(
        "---\nname: test\nmodels: [openai:gpt-4o]\n---\n"
    )
    (tmp_path / "judge.md").write_text("---\nmodel: openai:gpt-4o\n---\nEvaluate.\n")
    variant_dir = tmp_path / "v1"
    variant_dir.mkdir()

    with pytest.raises(YamlConfigLoaderError, match="prompt.md not found"):
        loader.load_variant(variant_dir)


def test_load_variant_raises_when_no_judge(loader: YamlConfigLoader, tmp_path: Path):
    (tmp_path / "experiment.md").write_text(
        "---\nname: test\nmodels: [openai:gpt-4o]\n---\n"
    )
    variant_dir = tmp_path / "v1"
    variant_dir.mkdir()
    (variant_dir / "prompt.md").write_text("Hello")

    with pytest.raises(YamlConfigLoaderError, match="No judge.md found"):
        loader.load_variant(variant_dir)


@pytest.mark.parametrize(
    "system,expected",
    [
        ("You are a translator", "You are a translator"),
        (None, None),
    ],
)
def test_load_variant_system_md(
    loader: YamlConfigLoader, tmp_path: Path, system: str | None, expected: str | None
):
    _create_experiment(tmp_path, system=system)

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.prompt.system_content == expected


def test_load_variant_reads_inputs(loader: YamlConfigLoader, tmp_path: Path):
    _create_experiment(
        tmp_path,
        inputs="- id: greeting-1\n  message: Hello!\n- id: greeting-2\n  message: Hi!\n",
    )

    variant = loader.load_variant(tmp_path / "v1")

    assert len(variant.inputs) == 2
    assert variant.inputs[0].id == "greeting-1"
    assert variant.inputs[0].data["message"] == "Hello!"
    assert variant.inputs[1].id == "greeting-2"


def test_load_variant_defaults_inputs_when_missing(
    loader: YamlConfigLoader, tmp_path: Path
):
    _create_experiment(tmp_path)

    variant = loader.load_variant(tmp_path / "v1")

    assert len(variant.inputs) == 1
    assert variant.inputs[0].id == "default"
    assert variant.inputs[0].data == {}


def test_load_variant_reads_input_runs_override(
    loader: YamlConfigLoader, tmp_path: Path
):
    _create_experiment(
        tmp_path,
        inputs="- id: test-1\n  message: Hello\n  runs: 5\n- id: test-2\n  message: Hi\n",
    )

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.inputs[0].runs == 5
    assert variant.inputs[1].runs is None


# --- judge config via load_variant ---


@pytest.mark.parametrize(
    "judge_yaml,expected",
    [
        ("---\nmodel: openai:gpt-4o\n---\nEvaluate.\n", True),
        ("---\nmodel: openai:gpt-4o\nchain_of_thought: true\n---\nEvaluate.\n", True),
        ("---\nmodel: openai:gpt-4o\nchain_of_thought: false\n---\nEvaluate.\n", False),
    ],
)
def test_load_variant_judge_chain_of_thought(
    loader: YamlConfigLoader, tmp_path: Path, judge_yaml: str, expected: bool
):
    _create_experiment(tmp_path, judge=judge_yaml)

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.judge.chain_of_thought is expected


def test_load_variant_judge_single_model(loader: YamlConfigLoader, tmp_path: Path):
    _create_experiment(tmp_path)

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.judge.model == "openai:gpt-4o"
    assert variant.judge.models is None
    assert variant.judge.is_multi_judge is False
    assert variant.judge.judge_models == ["openai:gpt-4o"]


def test_load_variant_judge_multi_model(loader: YamlConfigLoader, tmp_path: Path):
    _create_experiment(
        tmp_path,
        judge=(
            "---\n"
            "models:\n"
            "  - openai:gpt-4o-mini\n"
            "  - anthropic:claude-sonnet-4-20250514\n"
            "aggregation: mean\n"
            "---\n"
            "Evaluate.\n"
        ),
    )

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.judge.models == [
        "openai:gpt-4o-mini",
        "anthropic:claude-sonnet-4-20250514",
    ]
    assert variant.judge.aggregation == "mean"
    assert variant.judge.is_multi_judge is True


@pytest.mark.parametrize(
    "aggregation_yaml,expected",
    [
        ("", "mean"),
        ("aggregation: mean\n", "mean"),
        ("aggregation: median\n", "median"),
    ],
)
def test_load_variant_judge_aggregation(
    loader: YamlConfigLoader, tmp_path: Path, aggregation_yaml: str, expected: str
):
    _create_experiment(
        tmp_path,
        judge=(
            "---\n"
            "models:\n"
            "  - openai:gpt-4o-mini\n"
            "  - anthropic:claude-sonnet-4-20250514\n"
            f"{aggregation_yaml}"
            "---\n"
            "Evaluate.\n"
        ),
    )

    variant = loader.load_variant(tmp_path / "v1")

    assert variant.judge.aggregation == expected


def test_load_variant_judge_invalid_aggregation(
    loader: YamlConfigLoader, tmp_path: Path
):
    _create_experiment(
        tmp_path,
        judge="---\nmodels:\n  - openai:gpt-4o-mini\naggregation: invalid\n---\nEvaluate.\n",
    )

    with pytest.raises(YamlConfigLoaderError, match="Invalid aggregation"):
        loader.load_variant(tmp_path / "v1")


# --- discover_variants ---


def test_discover_variants_finds_all(loader: YamlConfigLoader):
    variants = loader.discover_variants(SAMPLE_EXPERIMENT)

    assert len(variants) == 2
    variant_names = [v.name for v in variants]
    assert "v1" in variant_names
    assert "v2" in variant_names


def test_discover_variants_raises_when_none(loader: YamlConfigLoader, tmp_path: Path):
    (tmp_path / "experiment.md").write_text(
        "---\nname: test\nmodels: [openai:gpt-4o]\n---\n"
    )

    with pytest.raises(YamlConfigLoaderError, match="No variants found"):
        loader.discover_variants(tmp_path)
