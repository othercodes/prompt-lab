import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

from .loader import discover_variants, load_experiment
from .runner import (
    RunResult,
    RunSummary,
    SignificanceResult,
    calculate_confidence_interval,
    compare_variants_significance,
    load_results,
)

console = Console()


def display_hypothesis(hypothesis: str) -> None:
    if hypothesis:
        console.print()
        console.print(f"[bold]Hypothesis:[/bold] [italic]{hypothesis}[/italic]")


def display_results_table(summary: RunSummary, show_hypothesis: bool = True) -> None:
    # Use stats view if multiple runs, otherwise show individual results
    if summary.runs_per_input > 1 and summary.stats:
        _display_stats_table(summary, show_hypothesis)
    else:
        _display_individual_results_table(summary, show_hypothesis)


def _display_individual_results_table(
    summary: RunSummary, show_hypothesis: bool = True
) -> None:
    console.print()
    if show_hypothesis and summary.hypothesis:
        console.print(
            f"[bold]Hypothesis:[/bold] [italic]{summary.hypothesis}[/italic]\n"
        )

    table = Table(
        title=f"Results: {summary.experiment}/{summary.variant}",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Input", style="dim")
    table.add_column("Model")
    table.add_column("Score", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cached", justify="center")

    for result in summary.results:
        score = result.judge["score"]
        score_style = _score_style(score)

        table.add_row(
            result.input_id,
            result.model,
            Text(f"{score}/10", style=score_style),
            f"{result.latency_ms}ms",
            f"{result.input_tokens + result.output_tokens}",
            "[green]Yes[/green]" if result.cached else "[dim]No[/dim]",
        )

    console.print(table)
    console.print()

    avg_score = sum(r.judge["score"] for r in summary.results) / len(summary.results)
    avg_latency = sum(r.latency_ms for r in summary.results) / len(summary.results)

    console.print(
        f"[dim]Duration: {summary.duration_seconds}s | "
        f"Avg Score: {avg_score:.1f}/10 | "
        f"Avg Latency: {avg_latency:.0f}ms | "
        f"Cached: {summary.cached_responses}/{len(summary.results)}[/dim]"
    )


def _display_stats_table(summary: RunSummary, show_hypothesis: bool = True) -> None:
    console.print()
    if show_hypothesis and summary.hypothesis:
        console.print(
            f"[bold]Hypothesis:[/bold] [italic]{summary.hypothesis}[/italic]\n"
        )

    table = Table(
        title=f"Results: {summary.experiment}/{summary.variant} ({summary.runs_per_input} runs/input)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Input", style="dim")
    table.add_column("Model")
    table.add_column("Mean", justify="center")
    table.add_column("95% CI", justify="center")
    table.add_column("Range", justify="center")
    table.add_column("Scores", justify="left")

    for stat in summary.stats:
        score_style = _score_style(stat.mean)

        mean_display = f"{stat.mean:.1f}"
        ci_display = f"({stat.ci_lower:.1f}-{stat.ci_upper:.1f})"
        range_display = f"{stat.min_score}-{stat.max_score}"

        scores_display = ", ".join(str(s) for s in stat.scores)

        table.add_row(
            stat.input_id,
            stat.model,
            Text(mean_display, style=score_style),
            f"[dim]{ci_display}[/dim]",
            f"[dim]{range_display}[/dim]",
            f"[dim]{scores_display}[/dim]",
        )

    console.print(table)
    console.print()

    all_means = [s.mean for s in summary.stats]
    overall_mean = sum(all_means) / len(all_means) if all_means else 0

    avg_latency = sum(r.latency_ms for r in summary.results) / len(summary.results)

    # Sample size warning
    if summary.runs_per_input < 5:
        console.print(
            f"[yellow]⚠ Low sample size ({summary.runs_per_input} runs). "
            f"Consider runs: 5+ for reliable statistics.[/yellow]\n"
        )

    console.print(
        f"[dim]Duration: {summary.duration_seconds}s | "
        f"Overall Mean: {overall_mean:.1f}/10 | "
        f"Avg Latency: {avg_latency:.0f}ms | "
        f"Total Runs: {len(summary.results)}[/dim]"
    )


def display_compare_table(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)
    variants = discover_variants(experiment_path)

    console.print()
    if experiment.hypothesis:
        console.print(
            f"[bold]Hypothesis:[/bold] [italic]{experiment.hypothesis}[/italic]\n"
        )

    table = Table(
        title=f"Comparison: {experiment_path.name}",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Variant", style="dim")
    table.add_column("Mean Score", justify="center")
    table.add_column("95% CI", justify="center")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Runs", justify="center")

    variant_summaries: list[tuple[str, RunSummary]] = []

    for variant_path in variants:
        try:
            summary = load_results(variant_path)

            if not summary.results:
                continue

            variant_summaries.append((variant_path.name, summary))

            # Calculate overall CI from all scores
            all_scores = [r.judge["score"] for r in summary.results]
            avg_score = statistics.mean(all_scores)

            if len(all_scores) > 1:
                stddev = statistics.stdev(all_scores)
                ci_lower, ci_upper = calculate_confidence_interval(
                    avg_score, stddev, len(all_scores)
                )
                ci_display = f"({ci_lower:.1f}-{ci_upper:.1f})"
            else:
                ci_display = "-"

            avg_latency = sum(r.latency_ms for r in summary.results) / len(
                summary.results
            )
            score_style = _score_style(avg_score)

            runs_info = f"{len(summary.results)}"
            if summary.runs_per_input > 1:
                runs_info = f"{summary.inputs_count}×{summary.runs_per_input}"

            table.add_row(
                variant_path.name,
                Text(f"{avg_score:.1f}/10", style=score_style),
                f"[dim]{ci_display}[/dim]",
                f"{avg_latency:.0f}ms",
                runs_info,
            )
        except Exception:
            table.add_row(
                variant_path.name,
                "[dim]No results[/dim]",
                "-",
                "-",
                "-",
            )

    console.print(table)
    console.print()

    # Show significance testing results if we have multiple variants
    if len(variant_summaries) >= 2:
        sig_results = compare_variants_significance(variant_summaries)
        _display_significance_results(sig_results)


def _display_significance_results(results: list[SignificanceResult]) -> None:
    if not results:
        return

    console.print("[bold]Statistical Significance (Welch's t-test, α=0.05):[/bold]")
    console.print()

    for r in results:
        if r.significant:
            console.print(
                f"  [green]✓[/green] {r.winner} > {r.variant1 if r.winner == r.variant2 else r.variant2} "
                f"[dim](p≤{r.p_value})[/dim]"
            )
        else:
            console.print(
                f"  [dim]–[/dim] {r.variant1} ≈ {r.variant2} "
                f"[dim](no significant difference)[/dim]"
            )

    console.print()


def display_response(
    variant_path: Path,
    input_id: str | None = None,
    model: str | None = None,
    run_timestamp: str | None = None,
) -> None:
    summary = load_results(variant_path, run_timestamp)

    results = summary.results
    if input_id:
        results = [r for r in results if r.input_id == input_id]
    if model:
        results = [r for r in results if r.model == model]

    if not results:
        console.print("[red]No matching results found[/red]")
        return

    for result in results:
        _display_single_response(result)


def _display_single_response(result: RunResult) -> None:
    content = []

    content.append("[bold cyan]RESPONSE[/bold cyan]")
    content.append("[dim]─────────[/dim]")
    response_text = result.response.get("content", "")
    if len(response_text) > 500:
        content.append(response_text[:500] + "...")
    else:
        content.append(response_text or "[dim]No content[/dim]")
    content.append("")

    tool_calls = result.response.get("tool_calls", [])
    if tool_calls:
        content.append("[bold cyan]TOOL CALLS[/bold cyan]")
        content.append("[dim]──────────[/dim]")
        for tc in tool_calls:
            args = ", ".join(f'{k}="{v}"' for k, v in tc["arguments"].items())
            content.append(f"• {tc['name']}({args})")
        content.append("")

    content.append("[bold cyan]JUDGE[/bold cyan]")
    content.append("[dim]─────[/dim]")
    score = result.judge["score"]
    score_style = _score_style(score)
    content.append(f"Score: [{score_style}]{score}/10[/{score_style}]")
    reasoning = result.judge.get("reasoning", "")
    if reasoning:
        content.append(f"Reasoning: {reasoning}")
    content.append("")

    content.append("[bold cyan]METRICS[/bold cyan]")
    content.append("[dim]───────[/dim]")
    cached = "[green]Yes[/green]" if result.cached else "[dim]No[/dim]"
    content.append(
        f"Latency: {result.latency_ms}ms │ "
        f"Tokens: {result.input_tokens} in / {result.output_tokens} out │ "
        f"Cached: {cached}"
    )

    title = f"[bold]{result.input_id} × {result.model}[/bold]"
    if result.run_number > 1:
        title = (
            f"[bold]{result.input_id} × {result.model} (run {result.run_number})[/bold]"
        )

    panel = Panel(
        "\n".join(content),
        title=title,
        border_style="blue",
    )

    console.print()
    console.print(panel)


@contextmanager
def progress_bar(
    description: str, total: int
) -> Generator[tuple[Progress, TaskID], None, None]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    )
    with progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id


def display_run_complete(summary: RunSummary, show_hypothesis: bool = True) -> None:
    console.print(
        f"\n[bold green]Complete![/bold green] "
        f"Results saved to results/{summary.timestamp}/"
    )
    display_results_table(summary, show_hypothesis)


def _score_style(score: float) -> str:
    if score >= 8:
        return "bold green"
    elif score >= 6:
        return "yellow"
    elif score >= 4:
        return "orange3"
    else:
        return "bold red"
