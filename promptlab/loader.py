from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter
import yaml


@dataclass
class ExperimentConfig:
    name: str
    description: str
    models: list[str]
    hypothesis: str = ""
    runs: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    content: str
    models: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    content: str
    model: str = "openai:gpt-4o"
    score_range: tuple[int, int] = (1, 10)
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputCase:
    id: str
    data: dict[str, Any]
    runs: int | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class VariantConfig:
    path: Path
    experiment: ExperimentConfig
    prompt: PromptConfig
    judge: JudgeConfig
    inputs: list[InputCase]
    tools: list[ToolDefinition] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return self.prompt.models or self.experiment.models


class LoaderError(Exception):
    pass


def load_experiment(path: Path) -> ExperimentConfig:
    experiment_file = path / "experiment.md"
    if not experiment_file.exists():
        raise LoaderError(f"experiment.md not found in {path}")

    post = frontmatter.load(experiment_file)
    metadata = dict(post.metadata)

    name = metadata.pop("name", path.name)
    description = metadata.pop("description", "")
    models = metadata.pop("models", [])
    hypothesis = metadata.pop("hypothesis", "")
    runs = metadata.pop("runs", 5)

    if not models:
        raise LoaderError(f"No models specified in {experiment_file}")

    return ExperimentConfig(
        name=name,
        description=description,
        models=models,
        hypothesis=hypothesis,
        runs=int(runs),
        metadata=metadata,
    )


def load_prompt(path: Path) -> PromptConfig:
    prompt_file = path / "prompt.md"
    if not prompt_file.exists():
        raise LoaderError(f"prompt.md not found in {path}")

    post = frontmatter.load(prompt_file)
    metadata = dict(post.metadata)

    models = metadata.pop("models", None)

    return PromptConfig(
        content=post.content,
        models=models,
        metadata=metadata,
    )


def load_judge(variant_path: Path, experiment_path: Path) -> JudgeConfig:
    variant_judge = variant_path / "judge.md"
    if variant_judge.exists():
        return _parse_judge(variant_judge)

    experiment_judge = experiment_path / "judge.md"
    if experiment_judge.exists():
        return _parse_judge(experiment_judge)

    raise LoaderError(f"No judge.md found in {variant_path} or {experiment_path}")


def _parse_judge(path: Path) -> JudgeConfig:
    post = frontmatter.load(path)
    metadata = dict(post.metadata)

    model = metadata.pop("model", "openai:gpt-4o")
    temperature = metadata.pop("temperature", 0.0)

    score_range = metadata.pop("score_range", None)
    if score_range:
        score_min, score_max = score_range
    else:
        score_min = metadata.pop("score_min", 1)
        score_max = metadata.pop("score_max", 10)

    return JudgeConfig(
        content=post.content,
        model=model,
        score_range=(score_min, score_max),
        temperature=float(temperature),
        metadata=metadata,
    )


def load_inputs(variant_path: Path, experiment_path: Path) -> list[InputCase]:
    variant_inputs = variant_path / "inputs.yaml"
    if variant_inputs.exists():
        return _parse_inputs(variant_inputs)

    experiment_inputs = experiment_path / "inputs.yaml"
    if experiment_inputs.exists():
        return _parse_inputs(experiment_inputs)

    # No inputs file = single run with empty data (for static prompts)
    return [InputCase(id="default", data={})]


def _parse_inputs(path: Path) -> list[InputCase]:
    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        # Empty file = single run with empty data
        return [InputCase(id="default", data={})]

    if not isinstance(data, list):
        raise LoaderError("inputs.yaml must contain a list of test cases")

    inputs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise LoaderError(f"Input case {i} must be a dictionary")

        input_id = item.pop("id", f"input-{i}")
        runs = item.pop("runs", None)
        inputs.append(InputCase(id=input_id, data=item, runs=runs))

    return inputs


def load_tools(path: Path) -> list[ToolDefinition]:
    tools_file = path / "tools.yaml"
    if not tools_file.exists():
        return []

    with open(tools_file) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise LoaderError("tools.yaml must contain a list of tool definitions")

    tools = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise LoaderError(f"Tool definition {i} must be a dictionary")

        name = item.get("name")
        description = item.get("description", "")
        parameters = item.get("parameters", {})

        if not name:
            raise LoaderError(f"Tool definition {i} missing 'name'")

        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
            )
        )

    return tools


def load_variant(variant_path: Path) -> VariantConfig:
    variant_path = Path(variant_path).resolve()

    if not variant_path.is_dir():
        raise LoaderError(f"Variant path must be a directory: {variant_path}")

    experiment_path = variant_path.parent

    experiment = load_experiment(experiment_path)
    prompt = load_prompt(variant_path)
    judge = load_judge(variant_path, experiment_path)
    inputs = load_inputs(variant_path, experiment_path)
    tools = load_tools(variant_path)

    return VariantConfig(
        path=variant_path,
        experiment=experiment,
        prompt=prompt,
        judge=judge,
        inputs=inputs,
        tools=tools,
    )


def discover_variants(experiment_path: Path) -> list[Path]:
    experiment_path = Path(experiment_path).resolve()

    if not experiment_path.is_dir():
        raise LoaderError(f"Experiment path must be a directory: {experiment_path}")

    variants = []
    for item in experiment_path.iterdir():
        if item.is_dir() and (item / "prompt.md").exists():
            variants.append(item)

    if not variants:
        raise LoaderError(f"No variants found in {experiment_path}")

    return sorted(variants)
