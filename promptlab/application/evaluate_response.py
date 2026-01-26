from dataclasses import dataclass
from typing import Any, Callable

from ..domain.contracts.config import JudgeConfig
from ..domain.contracts.provider import ProviderContract, ProviderResponse
from .prompts import get_judge_suffix


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    raw: dict[str, Any]


class JudgeError(Exception):
    pass


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
        user_input: dict[str, Any],
        response: ProviderResponse,
    ) -> JudgeResult:
        provider_name, model = self._parse_model_id(judge_config.model)
        provider = self._provider_factory(provider_name)

        judge_input = {
            "original_prompt": prompt,
            "user_input": user_input,
            "response": response.content,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
        }

        score_min, score_max = judge_config.score_range
        judge_suffix = get_judge_suffix(score_min, score_max)
        judge_prompt = judge_config.content + "\n\n" + judge_suffix

        try:
            result = await provider.execute_json(
                model=model,
                prompt=judge_prompt,
                user_input=judge_input,
                temperature=judge_config.temperature,
            )
        except Exception as e:
            raise JudgeError(f"Judge evaluation failed: {e}")

        if "score" not in result:
            raise JudgeError(f"Judge response missing 'score': {result}")

        score = result["score"]
        if not isinstance(score, int) or not (score_min <= score <= score_max):
            raise JudgeError(
                f"Invalid score {score}. "
                f"Must be integer between {score_min} and {score_max}"
            )

        reasoning = result.get("reasoning", "")

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            raw=result,
        )

    def _parse_model_id(self, model_id: str) -> tuple[str, str]:
        if ":" not in model_id:
            raise ValueError(
                f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
            )
        provider, model = model_id.split(":", 1)
        return provider, model
