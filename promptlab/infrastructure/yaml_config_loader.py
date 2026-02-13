from pathlib import Path

import frontmatter
import yaml

from ..domain.contracts.config import (
    ConfigLoaderContract,
    ExperimentConfig,
    InputCase,
    JudgeConfig,
    PromptConfig,
    ToolDefinition,
    VariantConfig,
)


class YamlConfigLoaderError(Exception):
    pass


class YamlConfigLoader(ConfigLoaderContract):
    def load_experiment(self, path: Path) -> ExperimentConfig:
        experiment_file = path / "experiment.md"
        if not experiment_file.exists():
            raise YamlConfigLoaderError(f"experiment.md not found in {path}")

        post = frontmatter.load(experiment_file)
        metadata = dict(post.metadata)

        name = metadata.pop("name", path.name)
        description = metadata.pop("description", "")
        models = metadata.pop("models", [])
        hypothesis = metadata.pop("hypothesis", "")
        runs = metadata.pop("runs", 5)
        key_refs = metadata.pop("key_refs", {})
        if not isinstance(key_refs, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in key_refs.items()
        ):
            raise YamlConfigLoaderError(
                f"key_refs must be a mapping of provider name to env var name in {experiment_file}"
            )

        if not models:
            raise YamlConfigLoaderError(f"No models specified in {experiment_file}")

        return ExperimentConfig(
            name=name,
            description=description,
            models=models,
            hypothesis=hypothesis,
            runs=int(runs),
            metadata=metadata,
            key_refs=key_refs,
        )

    def load_variant(self, variant_path: Path) -> VariantConfig:
        variant_path = Path(variant_path).resolve()

        if not variant_path.is_dir():
            raise YamlConfigLoaderError(
                f"Variant path must be a directory: {variant_path}"
            )

        experiment_path = variant_path.parent

        experiment = self.load_experiment(experiment_path)
        prompt = self._load_prompt(variant_path)
        judge = self._load_judge(variant_path, experiment_path)
        inputs = self._load_inputs(variant_path, experiment_path)
        tools = self._load_tools(variant_path)

        return VariantConfig(
            path=variant_path,
            experiment=experiment,
            prompt=prompt,
            judge=judge,
            inputs=inputs,
            tools=tools,
        )

    def discover_variants(self, experiment_path: Path) -> list[Path]:
        experiment_path = Path(experiment_path).resolve()

        if not experiment_path.is_dir():
            raise YamlConfigLoaderError(
                f"Experiment path must be a directory: {experiment_path}"
            )

        variants = []
        for item in experiment_path.iterdir():
            if item.is_dir() and (item / "prompt.md").exists():
                variants.append(item)

        if not variants:
            raise YamlConfigLoaderError(f"No variants found in {experiment_path}")

        return sorted(variants)

    def _load_prompt(self, path: Path) -> PromptConfig:
        prompt_file = path / "prompt.md"
        if not prompt_file.exists():
            raise YamlConfigLoaderError(f"prompt.md not found in {path}")

        post = frontmatter.load(prompt_file)
        metadata = dict(post.metadata)

        models = metadata.pop("models", None)

        system_content = None
        system_file = path / "system.md"
        if system_file.exists():
            system_content = system_file.read_text().strip()

        return PromptConfig(
            content=post.content,
            system_content=system_content,
            models=models,
            metadata=metadata,
        )

    def _load_judge(self, variant_path: Path, experiment_path: Path) -> JudgeConfig:
        variant_judge = variant_path / "judge.md"
        if variant_judge.exists():
            return self._parse_judge(variant_judge)

        experiment_judge = experiment_path / "judge.md"
        if experiment_judge.exists():
            return self._parse_judge(experiment_judge)

        raise YamlConfigLoaderError(
            f"No judge.md found in {variant_path} or {experiment_path}"
        )

    def _parse_judge(self, path: Path) -> JudgeConfig:
        post = frontmatter.load(path)
        metadata = dict(post.metadata)

        # Single judge (default) vs multi-judge (opt-in)
        model = metadata.pop("model", "openai:gpt-4o")
        models = metadata.pop("models", None)  # Multi-judge: list of models
        aggregation = metadata.pop("aggregation", "mean")

        # Validate aggregation method
        if aggregation not in ("mean", "median"):
            raise YamlConfigLoaderError(
                f"Invalid aggregation '{aggregation}'. Must be 'mean' or 'median'"
            )

        temperature = metadata.pop("temperature", 0.0)
        chain_of_thought = metadata.pop("chain_of_thought", True)

        score_range = metadata.pop("score_range", None)
        if score_range:
            score_min, score_max = score_range
        else:
            score_min = metadata.pop("score_min", 1)
            score_max = metadata.pop("score_max", 10)

        return JudgeConfig(
            content=post.content,
            model=model,
            models=models,
            aggregation=aggregation,
            score_range=(score_min, score_max),
            temperature=float(temperature),
            chain_of_thought=bool(chain_of_thought),
            metadata=metadata,
        )

    def _load_inputs(
        self, variant_path: Path, experiment_path: Path
    ) -> list[InputCase]:
        variant_inputs = variant_path / "inputs.yaml"
        if variant_inputs.exists():
            return self._parse_inputs(variant_inputs)

        experiment_inputs = experiment_path / "inputs.yaml"
        if experiment_inputs.exists():
            return self._parse_inputs(experiment_inputs)

        return [InputCase(id="default", data={})]

    def _parse_inputs(self, path: Path) -> list[InputCase]:
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return [InputCase(id="default", data={})]

        if not isinstance(data, list):
            raise YamlConfigLoaderError("inputs.yaml must contain a list of test cases")

        inputs = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise YamlConfigLoaderError(f"Input case {i} must be a dictionary")

            input_id = item.pop("id", f"input-{i}")
            runs = item.pop("runs", None)
            inputs.append(InputCase(id=input_id, data=item, runs=runs))

        return inputs

    def _load_tools(self, path: Path) -> list[ToolDefinition]:
        tools_file = path / "tools.yaml"
        if not tools_file.exists():
            return []

        with open(tools_file) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise YamlConfigLoaderError(
                "tools.yaml must contain a list of tool definitions"
            )

        tools = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise YamlConfigLoaderError(f"Tool definition {i} must be a dictionary")

            name = item.get("name")
            description = item.get("description", "")
            parameters = item.get("parameters", {})

            if not name:
                raise YamlConfigLoaderError(f"Tool definition {i} missing 'name'")

            tools.append(
                ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        return tools
