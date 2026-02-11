from pathlib import Path

import pytest
import yaml

from promptlab.application.create_experiment import (
    CreateExperiment,
    CreateExperimentError,
    parse_config,
    validate_spec,
)
from promptlab.domain.contracts.scaffold import ExperimentSpec, JudgeSpec, VariantSpec
from promptlab.infrastructure.experiment_scaffolder import ExperimentScaffolder
from promptlab.infrastructure.yaml_config_loader import YamlConfigLoader

VALID_CONFIG = {
    "name": "test-experiment",
    "models": ["openai:gpt-4o-mini"],
    "inputs": [
        {"id": "case-1", "text": "hello world"},
        {"id": "case-2", "text": "goodbye world"},
    ],
    "variants": {
        "v1": {"prompt": "Process this: {{ text }}"},
    },
    "judge": {
        "model": "openai:gpt-4o",
        "rubric": "Score the response.\n{{ prompt }}\n{{ response }}",
    },
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_file


# --- parse_config tests ---


def test_parse_config_valid(tmp_path):
    config_file = _write_config(tmp_path, VALID_CONFIG)
    spec = parse_config(config_file)

    assert spec.name == "test-experiment"
    assert spec.models == ["openai:gpt-4o-mini"]
    assert len(spec.inputs) == 2
    assert "v1" in spec.variants
    assert spec.variants["v1"].prompt == "Process this: {{ text }}"
    assert spec.judge.model == "openai:gpt-4o"


def test_parse_config_minimal(tmp_path):
    minimal = {
        "name": "minimal",
        "models": ["openai:gpt-4o-mini"],
        "inputs": [{"id": "case-1", "text": "hi"}],
        "variants": {"v1": {"prompt": "Say {{ text }}"}},
    }
    config_file = _write_config(tmp_path, minimal)
    spec = parse_config(config_file)

    assert spec.runs == 5  # default
    assert spec.judge.model == "openai:gpt-4o"  # default
    assert spec.judge.score_range == (1, 10)  # default
    assert spec.judge.chain_of_thought is True  # default
    assert spec.description == ""  # default
    assert spec.hypothesis == ""  # default


def test_parse_config_missing_name(tmp_path):
    config = {k: v for k, v in VALID_CONFIG.items() if k != "name"}
    config_file = _write_config(tmp_path, config)
    with pytest.raises(CreateExperimentError, match="Missing required keys.*name"):
        parse_config(config_file)


def test_parse_config_missing_models(tmp_path):
    config = {k: v for k, v in VALID_CONFIG.items() if k != "models"}
    config_file = _write_config(tmp_path, config)
    with pytest.raises(CreateExperimentError, match="Missing required keys.*models"):
        parse_config(config_file)


def test_parse_config_no_inputs_defaults_to_default(tmp_path):
    config = {k: v for k, v in VALID_CONFIG.items() if k != "inputs"}
    config_file = _write_config(tmp_path, config)
    spec = parse_config(config_file)
    assert len(spec.inputs) == 1
    assert spec.inputs[0]["id"] == "default"


def test_parse_config_missing_variants(tmp_path):
    config = {k: v for k, v in VALID_CONFIG.items() if k != "variants"}
    config_file = _write_config(tmp_path, config)
    with pytest.raises(CreateExperimentError, match="Missing required keys.*variants"):
        parse_config(config_file)


# --- validate_spec tests ---


def test_validate_model_format_invalid():
    spec = ExperimentSpec(
        name="test",
        models=["gpt-4o-mini"],  # missing provider prefix
        inputs=[{"id": "case-1", "text": "hi"}],
        variants={"v1": VariantSpec(prompt="Say {{ text }}")},
    )
    with pytest.raises(CreateExperimentError, match="Invalid model format"):
        validate_spec(spec)


def test_validate_jinja2_vars_match():
    spec = ExperimentSpec(
        name="test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello", "category": "greeting"}],
        variants={"v1": VariantSpec(prompt="Process {{ text }} in {{ category }}")},
    )
    validate_spec(spec)  # should not raise


def test_validate_jinja2_vars_mismatch():
    spec = ExperimentSpec(
        name="test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={"v1": VariantSpec(prompt="Process {{ text }} and {{ missing_var }}")},
    )
    with pytest.raises(
        CreateExperimentError, match="undefined template variables.*missing_var"
    ):
        validate_spec(spec)


def test_validate_judge_rubric_missing_vars():
    spec = ExperimentSpec(
        name="test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hi"}],
        variants={"v1": VariantSpec(prompt="Say {{ text }}")},
        judge=JudgeSpec(rubric="Score this: {{ prompt }}"),  # missing {{ response }}
    )
    with pytest.raises(CreateExperimentError, match="response"):
        validate_spec(spec)


def test_validate_hardcoded_prompt_skips_jinja2_check():
    spec = ExperimentSpec(
        name="test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={"v1": VariantSpec(prompt="Tell me a joke")},
    )
    validate_spec(spec)  # should not raise


# --- slugify tests (via public API) ---


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("sample exp", "sample-exp"),
        ("My Cool Experiment", "my-cool-experiment"),
        ("under_scores", "under-scores"),
        ("special!@#chars", "specialchars"),
        ("UPPERCASE", "uppercase"),
    ],
)
def test_parse_config_slugifies_name(tmp_path, input_name, expected):
    config = {**VALID_CONFIG, "name": input_name}
    config_file = _write_config(tmp_path, config)
    spec = parse_config(config_file)
    assert spec.name == expected


def test_validate_spec_slugifies_name():
    spec = ExperimentSpec(
        name="My Experiment",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={"v1": VariantSpec(prompt="Hello")},
    )
    validate_spec(spec)
    assert spec.name == "my-experiment"


# --- scaffolder tests ---


def test_scaffold_creates_directory_structure(tmp_path):
    spec = ExperimentSpec(
        name="test-exp",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={"v1": VariantSpec(prompt="Say {{ text }}")},
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    assert result_path.exists()
    assert (result_path / "experiment.md").exists()
    assert (result_path / "inputs.yaml").exists()
    assert (result_path / "judge.md").exists()
    assert (result_path / "v1" / "prompt.md").exists()


def test_scaffold_skips_inputs_yaml_for_hardcoded_prompts(tmp_path):
    spec = ExperimentSpec(
        name="test-exp",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={"v1": VariantSpec(prompt="Tell me a joke")},
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    assert result_path.exists()
    assert (result_path / "experiment.md").exists()
    assert not (result_path / "inputs.yaml").exists()
    assert (result_path / "judge.md").exists()
    assert (result_path / "v1" / "prompt.md").exists()


def test_scaffold_experiment_md_content(tmp_path):
    spec = ExperimentSpec(
        name="test-exp",
        description="A test experiment",
        hypothesis="Testing works",
        models=["openai:gpt-4o-mini"],
        runs=3,
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={"v1": VariantSpec(prompt="Say {{ text }}")},
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    import frontmatter

    post = frontmatter.load(result_path / "experiment.md")
    assert post["name"] == "test-exp"
    assert post["description"] == "A test experiment"
    assert post["hypothesis"] == "Testing works"
    assert post["models"] == ["openai:gpt-4o-mini"]
    assert post["runs"] == 3


def test_scaffold_existing_dir_error(tmp_path):
    spec = ExperimentSpec(
        name="existing-exp",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={"v1": VariantSpec(prompt="Say {{ text }}")},
        path=str(tmp_path),
    )
    # Create dir first
    (tmp_path / "existing-exp").mkdir()

    creator = CreateExperiment(ExperimentScaffolder())
    with pytest.raises(CreateExperimentError, match="already exists"):
        creator.from_spec(spec)


# --- system.md tests ---


def test_scaffold_creates_system_md_when_present(tmp_path):
    spec = ExperimentSpec(
        name="test-exp",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={
            "v1": VariantSpec(prompt="Tell me a joke", system="You are a comedian")
        },
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    assert (result_path / "v1" / "system.md").exists()
    assert (result_path / "v1" / "system.md").read_text() == "You are a comedian"


def test_scaffold_skips_system_md_when_absent(tmp_path):
    spec = ExperimentSpec(
        name="test-exp",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={"v1": VariantSpec(prompt="Tell me a joke")},
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    assert not (result_path / "v1" / "system.md").exists()


def test_loader_reads_system_md(tmp_path):
    spec = ExperimentSpec(
        name="sys-test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={
            "v1": VariantSpec(
                prompt="Translate: {{ text }}",
                system="You are a translator",
            )
        },
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    loader = YamlConfigLoader()
    variants = loader.discover_variants(result_path)
    variant = loader.load_variant(variants[0])

    assert variant.prompt.system_content == "You are a translator"
    assert "{{ text }}" in variant.prompt.content


def test_loader_returns_none_system_when_no_file(tmp_path):
    spec = ExperimentSpec(
        name="no-sys-test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "default"}],
        variants={"v1": VariantSpec(prompt="Hello")},
        path=str(tmp_path),
    )
    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    loader = YamlConfigLoader()
    variants = loader.discover_variants(result_path)
    variant = loader.load_variant(variants[0])

    assert variant.prompt.system_content is None


def test_validate_system_template_vars(tmp_path):
    spec = ExperimentSpec(
        name="test",
        models=["openai:gpt-4o-mini"],
        inputs=[{"id": "case-1", "text": "hello"}],
        variants={
            "v1": VariantSpec(
                prompt="Say {{ text }}",
                system="You are a {{ missing_role }}",
            )
        },
    )
    with pytest.raises(
        CreateExperimentError, match="undefined template variables.*missing_role"
    ):
        validate_spec(spec)


def test_parse_config_with_system(tmp_path):
    config = {
        **VALID_CONFIG,
        "variants": {
            "v1": {"prompt": "Process this: {{ text }}", "system": "You are an expert"},
        },
    }
    config_file = _write_config(tmp_path, config)
    spec = parse_config(config_file)
    assert spec.variants["v1"].system == "You are an expert"


# --- Round-trip integration test (KEY) ---


def test_roundtrip_scaffold_then_load(tmp_path):
    spec = ExperimentSpec(
        name="roundtrip-test",
        description="Testing round-trip",
        hypothesis="Scaffolded files are loadable",
        models=["openai:gpt-4o-mini"],
        runs=3,
        inputs=[
            {"id": "case-1", "text": "hello", "category": "greeting"},
            {"id": "case-2", "text": "bye", "category": "farewell"},
        ],
        variants={
            "v1": VariantSpec(prompt="Process {{ text }} as {{ category }}"),
        },
        judge=JudgeSpec(
            model="openai:gpt-4o",
            score_range=(1, 5),
            temperature=0.0,
            rubric="Evaluate.\n{{ prompt }}\n{{ response }}",
        ),
        path=str(tmp_path),
    )

    scaffolder = ExperimentScaffolder()
    result_path = scaffolder.scaffold(spec)

    # Load back with YamlConfigLoader
    loader = YamlConfigLoader()
    experiment = loader.load_experiment(result_path)

    assert experiment.name == "roundtrip-test"
    assert experiment.description == "Testing round-trip"
    assert experiment.hypothesis == "Scaffolded files are loadable"
    assert experiment.models == ["openai:gpt-4o-mini"]
    assert experiment.runs == 3

    # Load variant
    variants = loader.discover_variants(result_path)
    assert len(variants) == 1

    variant = loader.load_variant(variants[0])
    assert "{{ text }}" in variant.prompt.content
    assert "{{ category }}" in variant.prompt.content
    assert len(variant.inputs) == 2
    assert variant.inputs[0].id == "case-1"
    assert variant.inputs[0].data["text"] == "hello"
    assert variant.judge.model == "openai:gpt-4o"
    assert variant.judge.score_range == (1, 5)
