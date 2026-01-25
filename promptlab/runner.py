import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .cache import ResponseCache, get_cache
from .judge import evaluate
from .loader import InputCase, discover_variants, load_variant
from .providers.base import (
    Provider,
    ProviderResponse,
    get_provider,
    parse_model_id,
)


@dataclass
class RunResult:
    input_id: str
    model: str
    run_number: int
    cached: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    response: dict[str, Any]
    judge: dict[str, Any]


@dataclass
class InputStats:
    input_id: str
    model: str
    runs: int
    scores: list[int]
    mean: float
    stddev: float
    min_score: int
    max_score: int
    ci_lower: float
    ci_upper: float


@dataclass
class RunSummary:
    timestamp: str
    experiment: str
    variant: str
    models: list[str]
    inputs_count: int
    runs_per_input: int
    duration_seconds: float
    cached_responses: int
    hypothesis: str = ""
    results: list[RunResult] = field(default_factory=list)
    stats: list[InputStats] = field(default_factory=list)


class RunnerError(Exception):
    pass


async def run_single(
    provider: Provider,
    model: str,
    prompt: str,
    input_case: InputCase,
    tools: list[dict[str, Any]] | None,
    judge_config: Any,
    cache: ResponseCache | None,
    use_cache: bool = True,
    run_number: int = 1,
) -> RunResult:
    full_model_id = f"{provider.name}:{model}"

    cached = False
    response: ProviderResponse | None = None

    if use_cache and cache:
        cache_key = cache.make_key(
            prompt=prompt,
            input_data=input_case.data,
            model=full_model_id,
            tools=tools,
        )
        response = cache.get(cache_key)
        if response:
            cached = True

    if not response:
        response = await provider.execute(
            model=model,
            prompt=prompt,
            user_input=input_case.data,
            tools=tools,
        )

        if use_cache and cache:
            cache.put(cache_key, response)

    judge_result = await evaluate(
        judge_config=judge_config,
        prompt=prompt,
        user_input=input_case.data,
        response=response,
    )

    return RunResult(
        input_id=input_case.id,
        model=full_model_id,
        run_number=run_number,
        cached=cached,
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        response={
            "content": response.content,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
        },
        judge={
            "score": judge_result.score,
            "reasoning": judge_result.reasoning,
        },
    )


# t-critical values for 95% CI (two-tailed, alpha=0.05)
# df -> t_critical
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    50: 2.009,
    100: 1.984,
}


def get_t_critical(df: int) -> float:
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    # Find closest value
    for threshold in sorted(T_CRITICAL_95.keys(), reverse=True):
        if df >= threshold:
            return T_CRITICAL_95[threshold]
    return 1.96  # Fallback to z-value for large samples


def calculate_confidence_interval(
    mean: float, stddev: float, n: int
) -> tuple[float, float]:
    if n < 2:
        return (mean, mean)

    df = n - 1
    t_crit = get_t_critical(df)
    margin = t_crit * (stddev / math.sqrt(n))

    return (round(mean - margin, 2), round(mean + margin, 2))


# p-value thresholds for t-distribution (approximate, two-tailed)
# Format: {df: {t_value: p_value}}
P_VALUE_TABLE = {
    2: [(4.303, 0.05), (6.965, 0.01)],
    3: [(3.182, 0.05), (4.541, 0.01)],
    4: [(2.776, 0.05), (3.747, 0.01)],
    5: [(2.571, 0.05), (3.365, 0.01)],
    6: [(2.447, 0.05), (3.143, 0.01)],
    7: [(2.365, 0.05), (2.998, 0.01)],
    8: [(2.306, 0.05), (2.896, 0.01)],
    9: [(2.262, 0.05), (2.821, 0.01)],
    10: [(2.228, 0.05), (2.764, 0.01)],
    15: [(2.131, 0.05), (2.602, 0.01)],
    20: [(2.086, 0.05), (2.528, 0.01)],
    30: [(2.042, 0.05), (2.457, 0.01)],
    50: [(2.009, 0.05), (2.403, 0.01)],
    100: [(1.984, 0.05), (2.364, 0.01)],
}


def get_p_value_approx(t_stat: float, df: int) -> float:
    t_abs = abs(t_stat)

    # Find closest df in table
    closest_df = min(P_VALUE_TABLE.keys(), key=lambda x: abs(x - df))
    thresholds = P_VALUE_TABLE[closest_df]

    for t_thresh, p_val in thresholds:
        if t_abs >= t_thresh:
            return p_val

    return 1.0  # Not significant


@dataclass
class SignificanceResult:
    variant1: str
    variant2: str
    mean1: float
    mean2: float
    t_statistic: float
    p_value: float
    significant: bool
    winner: str | None


def welch_t_test(scores1: list[int], scores2: list[int]) -> tuple[float, float]:
    n1, n2 = len(scores1), len(scores2)

    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1 = statistics.mean(scores1)
    mean2 = statistics.mean(scores2)
    var1 = statistics.variance(scores1)
    var2 = statistics.variance(scores2)

    # Welch's t-test
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        # Zero variance in both groups - if means differ, difference is deterministic
        if mean1 != mean2:
            return float("inf") if mean1 > mean2 else float("-inf"), 0.01
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        df = float(n1 + n2 - 2)
    else:
        df = num / denom

    p_value = get_p_value_approx(t_stat, int(df))

    return round(t_stat, 3), p_value


def compare_variants_significance(
    summaries: list[tuple[str, RunSummary]],
) -> list[SignificanceResult]:
    results = []

    for i, (name1, sum1) in enumerate(summaries):
        for name2, sum2 in summaries[i + 1 :]:
            scores1 = [r.judge["score"] for r in sum1.results]
            scores2 = [r.judge["score"] for r in sum2.results]

            if not scores1 or not scores2:
                continue

            mean1 = statistics.mean(scores1)
            mean2 = statistics.mean(scores2)

            t_stat, p_value = welch_t_test(scores1, scores2)
            significant = p_value <= 0.05

            winner = None
            if significant:
                winner = name1 if mean1 > mean2 else name2

            results.append(
                SignificanceResult(
                    variant1=name1,
                    variant2=name2,
                    mean1=round(mean1, 2),
                    mean2=round(mean2, 2),
                    t_statistic=t_stat,
                    p_value=p_value,
                    significant=significant,
                    winner=winner,
                )
            )

    return results


def calculate_stats(results: list[RunResult]) -> list[InputStats]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        key = (result.input_id, result.model)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    stats_list = []
    for (input_id, model), group in grouped.items():
        scores = [r.judge["score"] for r in group]
        n = len(scores)

        if n == 1:
            mean = float(scores[0])
            stddev = 0.0
        else:
            mean = statistics.mean(scores)
            stddev = statistics.stdev(scores)

        ci_lower, ci_upper = calculate_confidence_interval(mean, stddev, n)

        stats_list.append(
            InputStats(
                input_id=input_id,
                model=model,
                runs=n,
                scores=scores,
                mean=round(mean, 2),
                stddev=round(stddev, 2),
                min_score=min(scores),
                max_score=max(scores),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return stats_list


def count_tasks(variant_path: Path, models: list[str] | None = None) -> int:
    config = load_variant(variant_path)
    run_models = models if models else config.models
    runs_count = config.experiment.runs

    total = 0
    for input_case in config.inputs:
        num_runs = input_case.runs if input_case.runs is not None else runs_count
        total += num_runs * len(run_models)

    return total


def count_experiment_tasks(
    experiment_path: Path, models: list[str] | None = None
) -> int:
    variants = discover_variants(experiment_path)
    return sum(count_tasks(v, models) for v in variants)


async def run_variant(
    variant_path: Path,
    models: list[str] | None = None,
    use_cache: bool = True,
    on_progress: Any | None = None,
) -> RunSummary:
    start_time = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config = load_variant(variant_path)

    run_models = models if models else config.models
    runs_count = config.experiment.runs

    for model in run_models:
        if model not in config.models:
            raise RunnerError(
                f"Model '{model}' not in experiment config. Available: {config.models}"
            )

    cache = get_cache() if use_cache else None

    tools_list = None
    if config.tools:
        tools_list = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in config.tools
        ]

    tasks = []
    for input_case in config.inputs:
        num_runs = input_case.runs if input_case.runs is not None else runs_count
        for run_num in range(1, num_runs + 1):
            for model_id in run_models:
                provider_name, model = parse_model_id(model_id)
                provider = get_provider(provider_name)

                # For multiple runs, disable cache to get independent responses
                effective_cache = cache if num_runs == 1 else None

                task = run_single(
                    provider=provider,
                    model=model,
                    prompt=config.prompt.content,
                    input_case=input_case,
                    tools=tools_list,
                    judge_config=config.judge,
                    cache=effective_cache,
                    use_cache=use_cache and num_runs == 1,
                    run_number=run_num,
                )
                tasks.append(task)

    results_list = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results_list.append(result)
        if on_progress:
            on_progress()

    duration = time.perf_counter() - start_time
    cached_count = sum(1 for r in results_list if r.cached)
    stats = calculate_stats(results_list)

    summary = RunSummary(
        timestamp=timestamp,
        experiment=config.experiment.name,
        variant=config.path.name,
        models=run_models,
        inputs_count=len(config.inputs),
        runs_per_input=runs_count,
        duration_seconds=round(duration, 2),
        cached_responses=cached_count,
        hypothesis=config.experiment.hypothesis,
        results=results_list,
        stats=stats,
    )

    save_results(config.path, summary)

    return summary


def save_results(variant_path: Path, summary: RunSummary) -> Path:
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
        filename = f"{result.input_id}_run{result.run_number}_{result.model.replace(':', '-')}.json"
        with open(responses_dir / filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

    if summary.stats:
        with open(results_dir / "stats.yaml", "w") as f:
            stats_data = [asdict(s) for s in summary.stats]
            yaml.dump(stats_data, f, default_flow_style=False)

    return results_dir


async def run_experiment(
    experiment_path: Path,
    models: list[str] | None = None,
    use_cache: bool = True,
    on_progress: Any | None = None,
) -> list[RunSummary]:
    variants = discover_variants(experiment_path)
    summaries = []

    for variant_path in variants:
        summary = await run_variant(
            variant_path=variant_path,
            models=models,
            use_cache=use_cache,
            on_progress=on_progress,
        )
        summaries.append(summary)

    return summaries


def load_results(variant_path: Path, run_timestamp: str | None = None) -> RunSummary:
    results_base = variant_path / "results"
    if not results_base.exists():
        raise RunnerError(f"No results found in {variant_path}")

    if run_timestamp:
        run_dir = results_base / run_timestamp
        if not run_dir.exists():
            raise RunnerError(f"Run '{run_timestamp}' not found")
    else:
        runs = sorted(results_base.iterdir(), reverse=True)
        if not runs:
            raise RunnerError(f"No runs found in {results_base}")
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

    # Load stats if available
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
