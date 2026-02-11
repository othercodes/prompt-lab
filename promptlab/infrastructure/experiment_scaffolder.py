from pathlib import Path

import frontmatter
import yaml

from ..domain.contracts.scaffold import ExperimentSpec


class ExperimentScaffolder:
    def scaffold(self, spec: ExperimentSpec) -> Path:
        experiment_dir = Path(spec.path) / spec.name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        self._write_experiment_md(experiment_dir, spec)
        if self._has_input_fields(spec):
            self._write_inputs_yaml(experiment_dir, spec)
        self._write_judge_md(experiment_dir, spec)
        self._write_variants(experiment_dir, spec)

        return experiment_dir

    def _write_experiment_md(self, experiment_dir: Path, spec: ExperimentSpec) -> None:
        metadata = {
            "name": spec.name,
            "description": spec.description,
            "hypothesis": spec.hypothesis,
            "models": spec.models,
            "runs": spec.runs,
        }

        body = spec.description if spec.description else f"# {spec.name}"

        post = frontmatter.Post(body, **metadata)
        experiment_file = experiment_dir / "experiment.md"
        with open(experiment_file, "w") as f:
            f.write(frontmatter.dumps(post))

    def _has_input_fields(self, spec: ExperimentSpec) -> bool:
        return any(k != "id" for inp in spec.inputs for k in inp)

    def _write_inputs_yaml(self, experiment_dir: Path, spec: ExperimentSpec) -> None:
        inputs_file = experiment_dir / "inputs.yaml"
        with open(inputs_file, "w") as f:
            yaml.dump(spec.inputs, f, default_flow_style=False, sort_keys=False)

    def _write_judge_md(self, experiment_dir: Path, spec: ExperimentSpec) -> None:
        judge = spec.judge

        metadata = {
            "model": judge.model,
            "score_range": list(judge.score_range),
            "temperature": judge.temperature,
        }

        if not judge.chain_of_thought:
            metadata["chain_of_thought"] = False

        if judge.models:
            metadata["models"] = judge.models
            metadata["aggregation"] = judge.aggregation

        rubric = judge.rubric
        if not rubric:
            score_min, score_max = judge.score_range
            rubric = f"""Evaluate the quality of the response.

## Input
{{{{ user_input }}}}

## Response
{{{{ response }}}}

Score the response on a scale of {score_min} to {score_max}."""

        post = frontmatter.Post(rubric, **metadata)
        judge_file = experiment_dir / "judge.md"

        # Write with custom handling for score_range to use flow style
        content = frontmatter.dumps(post)
        # Replace the score_range formatting to match the expected flow style
        content = content.replace(
            f"score_range:\n- {judge.score_range[0]}\n- {judge.score_range[1]}",
            f"score_range: [{judge.score_range[0]}, {judge.score_range[1]}]",
        )

        with open(judge_file, "w") as f:
            f.write(content)

    def _write_variants(self, experiment_dir: Path, spec: ExperimentSpec) -> None:
        for variant_name, variant in spec.variants.items():
            variant_dir = experiment_dir / variant_name
            variant_dir.mkdir(exist_ok=True)

            prompt_file = variant_dir / "prompt.md"
            with open(prompt_file, "w") as f:
                f.write(variant.prompt)

            if variant.system:
                system_file = variant_dir / "system.md"
                with open(system_file, "w") as f:
                    f.write(variant.system)

            # Write tools.yaml if tools are specified
            if variant.tools:
                tools_file = variant_dir / "tools.yaml"
                with open(tools_file, "w") as f:
                    yaml.dump(
                        variant.tools, f, default_flow_style=False, sort_keys=False
                    )
