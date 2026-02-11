import re
from pathlib import Path

import yaml

from ..domain.contracts.scaffold import ExperimentSpec, JudgeSpec, VariantSpec
from ..infrastructure.experiment_scaffolder import ExperimentScaffolder


class CreateExperimentError(Exception):
    pass


def _slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def parse_config(config_path: Path) -> ExperimentSpec:
    if not config_path.exists():
        raise CreateExperimentError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CreateExperimentError(f"Invalid YAML in {config_path}: {e}")

    if not isinstance(data, dict):
        raise CreateExperimentError(
            f"Config must be a YAML dictionary, got {type(data).__name__}"
        )

    # Validate required keys
    required_keys = {"name", "models", "variants"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise CreateExperimentError(
            f"Missing required keys: {', '.join(sorted(missing_keys))}"
        )

    # Parse variants
    variants_data = data.get("variants", {})
    if not isinstance(variants_data, dict):
        raise CreateExperimentError("'variants' must be a dictionary")
    if not variants_data:
        raise CreateExperimentError("At least one variant is required")

    variants = {}
    for variant_name, variant_data in variants_data.items():
        if not isinstance(variant_data, dict):
            raise CreateExperimentError(
                f"Variant '{variant_name}' must be a dictionary"
            )
        if "prompt" not in variant_data:
            raise CreateExperimentError(
                f"Variant '{variant_name}' missing required 'prompt' field"
            )

        variants[variant_name] = VariantSpec(
            prompt=variant_data["prompt"],
            system=variant_data.get("system"),
            description=variant_data.get("description", ""),
            tools=variant_data.get("tools"),
        )

    # Parse judge spec
    judge_data = data.get("judge", {})
    if not isinstance(judge_data, dict):
        raise CreateExperimentError("'judge' must be a dictionary")

    judge = JudgeSpec(
        rubric=judge_data.get("rubric", ""),
        model=judge_data.get("model", "openai:gpt-4o"),
        models=judge_data.get("models"),
        aggregation=judge_data.get("aggregation", "mean"),
        score_range=tuple(judge_data.get("score_range", [1, 10])),
        temperature=judge_data.get("temperature", 0.0),
        chain_of_thought=judge_data.get("chain_of_thought", True),
    )

    # Parse inputs (optional — defaults to single "default" input for hardcoded prompts)
    inputs = data.get("inputs", [{"id": "default"}])
    if not isinstance(inputs, list):
        raise CreateExperimentError("'inputs' must be a list")
    if not inputs:
        inputs = [{"id": "default"}]

    # Parse models
    models = data.get("models", [])
    if not isinstance(models, list):
        raise CreateExperimentError("'models' must be a list")
    if not models:
        raise CreateExperimentError("At least one model is required")

    return ExperimentSpec(
        name=_slugify(data["name"]),
        models=models,
        inputs=inputs,
        variants=variants,
        description=data.get("description", ""),
        hypothesis=data.get("hypothesis", ""),
        runs=data.get("runs", 5),
        path=data.get("path", "experiments"),
        judge=judge,
    )


def validate_spec(spec: ExperimentSpec) -> None:
    # 0. Sanitize name: replace spaces with hyphens
    spec.name = _slugify(spec.name)
    if not spec.name:
        raise CreateExperimentError("Experiment name is required")

    # 1. Model format: each must contain ':'
    for model in spec.models:
        if ":" not in model:
            raise CreateExperimentError(
                f"Invalid model format '{model}'. Expected format: 'provider:model' (e.g., 'openai:gpt-4o-mini')"
            )

    # 2. Each input must have 'id' field
    for i, input_data in enumerate(spec.inputs):
        if not isinstance(input_data, dict):
            raise CreateExperimentError(f"Input {i} must be a dictionary")
        if "id" not in input_data:
            raise CreateExperimentError(f"Input {i} missing required 'id' field")
        if not input_data["id"] or not isinstance(input_data["id"], str):
            raise CreateExperimentError(f"Input {i} 'id' must be a non-empty string")

    # 3. Collect all available fields from inputs (exclude 'id')
    available_fields: set[str] = set()
    for input_data in spec.inputs:
        available_fields.update(k for k in input_data.keys() if k != "id")

    # Extract and validate Jinja2 template variables in variant prompts (only if inputs have fields)
    template_var_pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    if available_fields:
        for variant_name, variant in spec.variants.items():
            template_vars = set(template_var_pattern.findall(variant.prompt))
            if variant.system:
                template_vars.update(template_var_pattern.findall(variant.system))
            invalid_vars = template_vars - available_fields

            if invalid_vars:
                raise CreateExperimentError(
                    f"Variant '{variant_name}' uses undefined template variables: {', '.join(sorted(invalid_vars))}. "
                    f"Available fields from inputs: {', '.join(sorted(available_fields))}"
                )

    # 4. Validate judge rubric if non-empty
    if spec.judge.rubric:
        required_judge_vars = {"prompt", "response"}
        judge_vars = set(template_var_pattern.findall(spec.judge.rubric))
        missing_judge_vars = required_judge_vars - judge_vars

        if missing_judge_vars:
            raise CreateExperimentError(
                f"Judge rubric must contain {', '.join(f'{{{{ {v} }}}}' for v in sorted(missing_judge_vars))} template variables"
            )


class CreateExperiment:
    def __init__(self, scaffolder: ExperimentScaffolder) -> None:
        self._scaffolder = scaffolder

    def from_config(self, config_path: Path) -> Path:
        spec = parse_config(config_path)
        validate_spec(spec)
        return self._create(spec)

    def from_spec(self, spec: ExperimentSpec) -> Path:
        validate_spec(spec)
        return self._create(spec)

    def _create(self, spec: ExperimentSpec) -> Path:
        target = Path(spec.path) / spec.name
        if target.exists():
            raise CreateExperimentError(
                f"Experiment directory already exists: {target}"
            )
        return self._scaffolder.scaffold(spec)
