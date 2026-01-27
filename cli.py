import asyncio
import shutil
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from promptlab.application.run_experiment import RunExperiment
from promptlab.infrastructure import FileCache, FileResultRepository, YamlConfigLoader
from promptlab.infrastructure.console_display import (
    display_compare_table,
    display_hypothesis,
    display_response,
    display_results_table,
    display_run_complete,
    progress_bar,
)
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
    """Check if path is a variant (has prompt.md)."""
    return (path / "prompt.md").exists()


def _is_experiment(path: Path) -> bool:
    """Check if path is an experiment (has experiment.md)."""
    return (path / "experiment.md").exists()


def _create_runner(use_cache: bool = True) -> RunExperiment:
    return RunExperiment(
        config_loader=_config_loader,
        result_repository=_result_repository,
        cache=_cache if use_cache else None,
        provider_factory=get_provider,
    )


async def _run_with_progress(
    path: Path,
    models: list[str] | None,
    use_cache: bool,
    is_experiment: bool,
) -> list:
    runner = _create_runner(use_cache)

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
) -> None:
    path = path.resolve()

    if not path.exists():
        typer.echo(f"Error: Path not found: {path}", err=True)
        raise typer.Exit(1)

    is_experiment = _is_experiment(path) and not _is_variant(path)

    models = [model] if model else None

    try:
        summaries = asyncio.run(
            _run_with_progress(path, models, not no_cache, is_experiment)
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
