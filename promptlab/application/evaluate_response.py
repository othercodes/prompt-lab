import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from ..domain.contracts.config import JudgeConfig
from ..domain.contracts.provider import ProviderContract, ProviderResponse
from .prompts import get_cot_prefix, get_judge_suffix


@dataclass
class IndividualJudgeResult:
    model: str
    score: int
    reasoning: str
    raw: dict[str, Any]


@dataclass
class JudgeResult:
    score: int  # Final score (aggregated if multi-judge)
    reasoning: str
    raw: dict[str, Any]
    individual_results: list[IndividualJudgeResult] = field(default_factory=list)

    @property
    def is_multi_judge(self) -> bool:
        return len(self.individual_results) > 1


class JudgeError(Exception):
    pass


def _aggregate_scores(scores: list[int], method: str) -> int:
    if method == "mean":
        return round(statistics.mean(scores))
    elif method == "median":
        return round(statistics.median(scores))
    else:
        # Default to mean
        return round(statistics.mean(scores))


def _parse_model_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise ValueError(
            f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
        )
    provider, model = model_id.split(":", 1)
    return provider, model


class EvaluateResponse:
    def __init__(
        self,
        provider_factory: Callable[[str], ProviderContract],
    ) -> None:
        self._provider_factory = provider_factory

    async def execute(
        self,
        judge_config: JudgeConfig,
        prompt: str,
        system_prompt: str,
        response: ProviderResponse,
    ) -> JudgeResult:
        judge_input = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response.content,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
        }

        score_min, score_max = judge_config.score_range
        judge_suffix = get_judge_suffix(score_min, score_max)

        if judge_config.chain_of_thought:
            cot_prefix = get_cot_prefix()
            judge_prompt = cot_prefix + judge_config.content + "\n\n" + judge_suffix
        else:
            judge_prompt = judge_config.content + "\n\n" + judge_suffix

        # Evaluate with all judge models
        individual_results: list[IndividualJudgeResult] = []
        for model_id in judge_config.judge_models:
            result = await self._evaluate_single(
                model_id=model_id,
                judge_prompt=judge_prompt,
                judge_input=judge_input,
                temperature=judge_config.temperature,
                score_range=judge_config.score_range,
            )
            individual_results.append(result)

        # Aggregate scores if multi-judge
        if len(individual_results) == 1:
            # Single judge: use result directly
            single = individual_results[0]
            return JudgeResult(
                score=single.score,
                reasoning=single.reasoning,
                raw=single.raw,
                individual_results=individual_results,
            )
        else:
            # Multi-judge: aggregate scores
            scores = [r.score for r in individual_results]
            aggregated_score = _aggregate_scores(scores, judge_config.aggregation)

            # Combine reasoning from all judges
            combined_reasoning = "\n\n".join(
                f"**{r.model}** (score: {r.score}):\n{r.reasoning}"
                for r in individual_results
            )

            return JudgeResult(
                score=aggregated_score,
                reasoning=combined_reasoning,
                raw={
                    "aggregation": judge_config.aggregation,
                    "individual_scores": [
                        {"model": r.model, "score": r.score} for r in individual_results
                    ],
                },
                individual_results=individual_results,
            )

    async def _evaluate_single(
        self,
        model_id: str,
        judge_prompt: str,
        judge_input: dict[str, Any],
        temperature: float,
        score_range: tuple[int, int],
    ) -> IndividualJudgeResult:
        provider_name, model = _parse_model_id(model_id)
        provider = self._provider_factory(provider_name)

        score_min, score_max = score_range

        try:
            result = await provider.execute_json(
                model=model,
                prompt=judge_prompt,
                user_input=judge_input,
                temperature=temperature,
            )
        except Exception as e:
            raise JudgeError(f"Judge evaluation failed ({model_id}): {e}")

        if "score" not in result:
            raise JudgeError(f"Judge response missing 'score' ({model_id}): {result}")

        score = result["score"]
        if not isinstance(score, int) or not (score_min <= score <= score_max):
            raise JudgeError(
                f"Invalid score {score} from {model_id}. "
                f"Must be integer between {score_min} and {score_max}"
            )

        reasoning = result.get("reasoning", "")

        return IndividualJudgeResult(
            model=model_id,
            score=score,
            reasoning=reasoning,
            raw=result,
        )
