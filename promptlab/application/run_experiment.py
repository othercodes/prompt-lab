import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from jinja2 import Template

from ..domain.contracts.cache import CacheContract
from ..domain.contracts.config import ConfigLoaderContract, InputCase
from ..domain.contracts.provider import ProviderContract, ProviderResponse
from ..domain.contracts.results import ResultRepositoryContract, RunResult, RunSummary
from ..domain.statistics import calculate_stats
from .evaluate_response import EvaluateResponse


class RunExperimentError(Exception):
    pass


class RunExperiment:
    def __init__(
        self,
        config_loader: ConfigLoaderContract,
        result_repository: ResultRepositoryContract,
        cache: CacheContract | None,
        provider_factory: Callable[[str], ProviderContract],
    ) -> None:
        self._config_loader = config_loader
        self._result_repository = result_repository
        self._cache = cache
        self._provider_factory = provider_factory
        self._evaluator = EvaluateResponse(provider_factory)

    async def run_variant(
        self,
        variant_path: Path,
        models: list[str] | None = None,
        use_cache: bool = True,
        on_progress: Callable[[], None] | None = None,
    ) -> RunSummary:
        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        config = self._config_loader.load_variant(variant_path)

        run_models = models if models else config.models
        runs_count = config.experiment.runs

        for model in run_models:
            if model not in config.models:
                raise RunExperimentError(
                    f"Model '{model}' not in experiment config. "
                    f"Available: {config.models}"
                )

        tools_list = None
        if config.tools:
            tools_list = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in config.tools
            ]

        tasks = []
        for input_case in config.inputs:
            num_runs = input_case.runs if input_case.runs is not None else runs_count
            for run_num in range(1, num_runs + 1):
                for model_id in run_models:
                    # For multiple runs, disable cache to get independent responses
                    effective_cache = (
                        self._cache if use_cache and num_runs == 1 else None
                    )

                    task = self._run_single(
                        model_id=model_id,
                        prompt=config.prompt.content,
                        system_prompt=config.prompt.system_content,
                        input_case=input_case,
                        tools=tools_list,
                        judge_config=config.judge,
                        cache=effective_cache,
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

        self._result_repository.save(config.path, summary)

        return summary

    async def run_all_variants(
        self,
        experiment_path: Path,
        models: list[str] | None = None,
        use_cache: bool = True,
        on_progress: Callable[[], None] | None = None,
    ) -> list[RunSummary]:
        variants = self._config_loader.discover_variants(experiment_path)
        summaries = []

        for variant_path in variants:
            summary = await self.run_variant(
                variant_path=variant_path,
                models=models,
                use_cache=use_cache,
                on_progress=on_progress,
            )
            summaries.append(summary)

        return summaries

    def count_tasks(self, variant_path: Path, models: list[str] | None = None) -> int:
        config = self._config_loader.load_variant(variant_path)
        run_models = models if models else config.models
        runs_count = config.experiment.runs

        total = 0
        for input_case in config.inputs:
            num_runs = input_case.runs if input_case.runs is not None else runs_count
            total += num_runs * len(run_models)

        return total

    def count_experiment_tasks(
        self, experiment_path: Path, models: list[str] | None = None
    ) -> int:
        variants = self._config_loader.discover_variants(experiment_path)
        return sum(self.count_tasks(v, models) for v in variants)

    async def _run_single(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        input_case: InputCase,
        tools: list[dict[str, Any]] | None,
        judge_config: Any,
        cache: CacheContract | None,
        run_number: int,
    ) -> RunResult:
        provider_name, model = self._parse_model_id(model_id)
        provider = self._provider_factory(provider_name)
        full_model_id = f"{provider.name}:{model}"

        cached = False
        response: ProviderResponse | None = None

        if cache:
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
                system_prompt=system_prompt,
            )

            if cache:
                cache.put(cache_key, response)

        rendered_prompt = Template(prompt).render(**input_case.data)
        rendered_system = (
            Template(system_prompt).render(**input_case.data) if system_prompt else ""
        )

        judge_result = await self._evaluator.execute(
            judge_config=judge_config,
            prompt=rendered_prompt,
            system_prompt=rendered_system,
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

    def _parse_model_id(self, model_id: str) -> tuple[str, str]:
        if ":" not in model_id:
            raise ValueError(
                f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
            )
        provider, model = model_id.split(":", 1)
        return provider, model
