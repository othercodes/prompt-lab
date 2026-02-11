import asyncio
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from promptlab.application.create_experiment import (
    CreateExperiment,
    CreateExperimentError,
)
from promptlab.application.run_experiment import RunExperiment
from promptlab.domain.contracts.scaffold import ExperimentSpec, JudgeSpec, VariantSpec
from promptlab.infrastructure import FileCache, FileResultRepository, YamlConfigLoader
from promptlab.infrastructure.console_display import (
    display_compare_table,
    display_hypothesis,
    display_response,
    display_results_table,
    display_run_complete,
    progress_bar,
)
from promptlab.infrastructure.experiment_scaffolder import ExperimentScaffolder
from promptlab.infrastructure.providers.factory import get_provider

load_dotenv()

app = typer.Typer(
    name="prompt-lab",
    help="Test prompt variants across LLM providers with LLM-as-judge evaluation",
    no_args_is_help=True,
)

cache_app = typer.Typer(help="Cache management commands")
app.add_typer(cache_app, name="cache")

_config_loader = YamlConfigLoader()
_result_repository = FileResultRepository()
_cache = FileCache()


def _is_variant(path: Path) -> bool:
    return (path / "prompt.md").exists()


def _is_experiment(path: Path) -> bool:
    return (path / "experiment.md").exists()


def _create_runner(use_cache: bool = True) -> RunExperiment:
    return RunExperiment(
        config_loader=_config_loader,
        result_repository=_result_repository,
        cache=_cache if use_cache else None,
        provider_factory=get_provider,
    )


def _run_wizard() -> ExperimentSpec:
    typer.echo("\n  Experiment Setup")
    typer.echo("  " + "─" * 40)

    name = typer.prompt("  Name")
    description = typer.prompt("  Description", default="")
    hypothesis = typer.prompt("  Hypothesis", default="")
    models_str = typer.prompt(
        "  Models (comma-separated)", default="openai:gpt-4o-mini"
    )
    models = [m.strip() for m in models_str.split(",")]
    runs = typer.prompt("  Runs per input", default=5, type=int)

    # Inputs (optional — skip for hardcoded prompts)
    typer.echo("\n  Input Cases")
    typer.echo("  " + "─" * 40)
    inputs: list[dict[str, Any]] = []
    if typer.confirm("  Add input cases? (skip for hardcoded prompts)", default=True):
        while True:
            typer.echo(f"\n  Input #{len(inputs) + 1}:")
            input_id = typer.prompt("    ID")
            input_data: dict[str, Any] = {"id": input_id}
            while True:
                field_name = typer.prompt(
                    "    Field name (empty to finish)", default=""
                )
                if not field_name:
                    break
                field_value = typer.prompt(f"    {field_name}")
                input_data[field_name] = field_value
            inputs.append(input_data)
            if not typer.confirm("  Add another input?", default=False):
                break
    if not inputs:
        inputs = [{"id": "default"}]

    # Collect available variables for hints
    available_vars: set[str] = set()
    for inp in inputs:
        available_vars.update(k for k in inp if k != "id")

    # Judge
    typer.echo("\n  Judge Configuration")
    typer.echo("  " + "─" * 40)
    judge_model = typer.prompt("  Judge model", default="openai:gpt-4o")
    score_min = typer.prompt("  Score range min", default=1, type=int)
    score_max = typer.prompt("  Score range max", default=10, type=int)
    rubric = typer.prompt(
        "  Judge rubric (use {{ prompt }} and {{ response }})",
        default="",
    )

    judge = JudgeSpec(
        rubric=rubric,
        model=judge_model,
        score_range=(score_min, score_max),
    )

    # Prompt v1
    typer.echo("\n  Prompt Variant (v1)")
    typer.echo("  " + "─" * 40)
    if available_vars:
        vars_hint = ", ".join(f"{{{{ {v} }}}}" for v in sorted(available_vars))
        typer.echo(f"  Available variables: {vars_hint}")
    system_text = typer.prompt("  System prompt (optional)", default="")
    prompt_text = typer.prompt("  User prompt template")

    variants = {
        "v1": VariantSpec(
            prompt=prompt_text,
            system=system_text or None,
        )
    }

    return ExperimentSpec(
        name=name,
        description=description,
        hypothesis=hypothesis,
        models=models,
        runs=runs,
        inputs=inputs,
        judge=judge,
        variants=variants,
    )


async def _run_with_progress(
    path: Path,
    models: list[str] | None,
    use_cache: bool,
    is_experiment: bool,
    quiet: bool = False,
) -> list[Any]:
    runner = _create_runner(use_cache)

    if quiet:
        if is_experiment:
            return await runner.run_all_variants(
                path, models=models, use_cache=use_cache
            )
        else:
            return [await runner.run_variant(path, models=models, use_cache=use_cache)]

    if is_experiment:
        total = runner.count_experiment_tasks(path, models)
        description = f"Running {path.name}"
    else:
        total = runner.count_tasks(path, models)
        description = f"Running {path.parent.name}/{path.name}"

    with progress_bar(description, total) as (progress, task_id):

        def on_progress() -> None:
            progress.advance(task_id)

        if is_experiment:
            return await runner.run_all_variants(
                path, models=models, use_cache=use_cache, on_progress=on_progress
            )
        else:
            return [
                await runner.run_variant(
                    path, models=models, use_cache=use_cache, on_progress=on_progress
                )
            ]


@app.command(help="Create a new experiment.")
def new(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to experiment spec YAML")
    ] = None,
) -> None:
    scaffolder = ExperimentScaffolder()
    creator = CreateExperiment(scaffolder)

    try:
        if config:
            config = config.resolve()
            if not config.exists():
                typer.echo(f"Error: Config file not found: {config}", err=True)
                raise typer.Exit(1)
            result_path = creator.from_config(config)
        else:
            spec = _run_wizard()
            result_path = creator.from_spec(spec)

        # Success output
        typer.echo(f"\nCreated experiment at: {result_path}/")
        for item in sorted(result_path.rglob("*")):
            if item.is_file():
                rel = item.relative_to(result_path)
                typer.echo(f"  {rel}")
        typer.echo(f"\nRun it with: prompt-lab run {result_path}")

    except CreateExperimentError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="Run prompt experiment.")
def run(
    path: Annotated[
        Path, typer.Argument(help="Path to variant or experiment directory")
    ],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Run only this model")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable caching")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Hide progress bar")
    ] = False,
) -> None:
    path = path.resolve()

    if not path.exists():
        typer.echo(f"Error: Path not found: {path}", err=True)
        raise typer.Exit(1)

    is_experiment = _is_experiment(path) and not _is_variant(path)

    models = [model] if model else None

    try:
        summaries = asyncio.run(
            _run_with_progress(path, models, not no_cache, is_experiment, quiet)
        )
        if is_experiment and summaries:
            display_hypothesis(summaries[0].hypothesis)

        for summary in summaries:
            display_run_complete(summary, show_hypothesis=not is_experiment)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="Show results table for a variant.")
def results(
    path: Annotated[Path, typer.Argument(help="Path to variant directory")],
    run_timestamp: Annotated[
        str | None, typer.Option("--run", "-r", help="Specific run timestamp")
    ] = None,
) -> None:
    path = path.resolve()

    try:
        summary = _result_repository.load(path, run_timestamp)
        display_results_table(summary)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="Compare results across variants.")
def compare(
    path: Annotated[Path, typer.Argument(help="Path to experiment directory")],
) -> None:
    path = path.resolve()

    try:
        display_compare_table(path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="Show detailed response(s).")
def show(
    path: Annotated[Path, typer.Argument(help="Path to variant directory")],
    input_id: Annotated[
        str | None, typer.Option("--input", "-i", help="Filter by input ID")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Filter by model")
    ] = None,
    run_timestamp: Annotated[
        str | None, typer.Option("--run", "-r", help="Specific run timestamp")
    ] = None,
) -> None:
    path = path.resolve()

    try:
        display_response(
            path, input_id=input_id, model=model, run_timestamp=run_timestamp
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="Clean experiment results.")
def clean(
    path: Annotated[
        Path, typer.Argument(help="Path to variant or experiment directory")
    ],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
) -> None:
    path = path.resolve()

    if not path.exists():
        typer.echo(f"Error: Path not found: {path}", err=True)
        raise typer.Exit(1)

    results_dirs: list[Path] = []

    if _is_experiment(path) and not _is_variant(path):
        try:
            variants = _config_loader.discover_variants(path)
            for variant_path in variants:
                results_base = variant_path / "results"
                if results_base.exists():
                    results_dirs.extend(d for d in results_base.iterdir() if d.is_dir())
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    else:
        results_base = path / "results"
        if results_base.exists():
            results_dirs = [d for d in results_base.iterdir() if d.is_dir()]

    if not results_dirs:
        typer.echo("No results to clean.")
        raise typer.Exit(0)

    typer.echo(f"Found {len(results_dirs)} result(s) to delete:")
    for d in sorted(results_dirs):
        typer.echo(f"  - {d.relative_to(path.parent)}")

    if not yes:
        confirm = typer.confirm("Delete these results?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    deleted = 0
    for d in results_dirs:
        try:
            shutil.rmtree(d)
            deleted += 1
        except Exception as e:
            typer.echo(f"Warning: Failed to delete {d}: {e}", err=True)

    typer.echo(f"Deleted {deleted} result(s).")


@cache_app.command("clear", help="Clear all cached responses.")
def cache_clear() -> None:
    try:
        _cache.clear()
        typer.echo("Cache cleared.")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
