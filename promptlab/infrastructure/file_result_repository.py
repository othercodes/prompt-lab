import json
from dataclasses import asdict
from pathlib import Path

import yaml

from ..domain.contracts.results import (
    InputStats,
    ResultRepositoryContract,
    RunResult,
    RunSummary,
)


class FileResultRepositoryError(Exception):
    pass


class FileResultRepository(ResultRepositoryContract):
    def save(self, variant_path: Path, summary: RunSummary) -> Path:
        results_dir = variant_path / "results" / summary.timestamp
        responses_dir = results_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        run_meta = {
            "timestamp": summary.timestamp,
            "experiment": summary.experiment,
            "variant": summary.variant,
            "models": summary.models,
            "inputs_count": summary.inputs_count,
            "runs_per_input": summary.runs_per_input,
            "duration_seconds": summary.duration_seconds,
            "cached_responses": summary.cached_responses,
        }

        with open(results_dir / "run.yaml", "w") as f:
            yaml.dump(run_meta, f, default_flow_style=False)

        for result in summary.results:
            filename = (
                f"{result.input_id}_run{result.run_number}_"
                f"{result.model.replace(':', '-')}.json"
            )
            with open(responses_dir / filename, "w") as f:
                json.dump(asdict(result), f, indent=2)

        if summary.stats:
            with open(results_dir / "stats.yaml", "w") as f:
                stats_data = [asdict(s) for s in summary.stats]
                yaml.dump(stats_data, f, default_flow_style=False)

        return results_dir

    def load(self, variant_path: Path, run_timestamp: str | None = None) -> RunSummary:
        results_base = variant_path / "results"
        if not results_base.exists():
            raise FileResultRepositoryError(f"No results found in {variant_path}")

        if run_timestamp:
            run_dir = results_base / run_timestamp
            if not run_dir.exists():
                raise FileResultRepositoryError(f"Run '{run_timestamp}' not found")
        else:
            runs = sorted(results_base.iterdir(), reverse=True)
            if not runs:
                raise FileResultRepositoryError(f"No runs found in {results_base}")
            run_dir = runs[0]

        with open(run_dir / "run.yaml") as f:
            meta = yaml.safe_load(f)

        responses_dir = run_dir / "responses"
        results = []

        if responses_dir.exists():
            for response_file in sorted(responses_dir.glob("*.json")):
                with open(response_file) as f:
                    data = json.load(f)
                    # Handle backward compatibility for old results without run_number
                    if "run_number" not in data:
                        data["run_number"] = 1
                    results.append(RunResult(**data))

        stats = []
        stats_file = run_dir / "stats.yaml"
        if stats_file.exists():
            with open(stats_file) as f:
                stats_data = yaml.safe_load(f)
                if stats_data:
                    stats = [InputStats(**s) for s in stats_data]

        return RunSummary(
            timestamp=meta["timestamp"],
            experiment=meta["experiment"],
            variant=meta["variant"],
            models=meta["models"],
            inputs_count=meta["inputs_count"],
            runs_per_input=meta.get("runs_per_input", 1),
            duration_seconds=meta["duration_seconds"],
            cached_responses=meta["cached_responses"],
            results=results,
            stats=stats,
        )

    def list_runs(self, variant_path: Path) -> list[str]:
        results_base = variant_path / "results"
        if not results_base.exists():
            return []

        return sorted(
            [d.name for d in results_base.iterdir() if d.is_dir()],
            reverse=True,
        )
