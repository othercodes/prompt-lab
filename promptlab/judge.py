from dataclasses import dataclass
from typing import Any

from .loader import JudgeConfig
from .prompts import get_judge_suffix
from .providers.base import ProviderResponse, get_provider, parse_model_id


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    raw: dict[str, Any]


class JudgeError(Exception):
    pass


async def evaluate(
    judge_config: JudgeConfig,
    prompt: str,
    user_input: dict[str, Any],
    response: ProviderResponse,
) -> JudgeResult:
    provider_name, model = parse_model_id(judge_config.model)
    provider = get_provider(provider_name)

    judge_input = {
        "original_prompt": prompt,
        "user_input": user_input,
        "response": response.content,
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
        ],
    }

    score_min, score_max = judge_config.score_range
    judge_suffix = get_judge_suffix(score_min, score_max)
    judge_prompt = judge_config.content + "\n\n" + judge_suffix

    try:
        if hasattr(provider, "execute_json"):
            result = await provider.execute_json(
                model=model,
                prompt=judge_prompt,
                user_input=judge_input,
                temperature=judge_config.temperature,
            )
        else:
            response = await provider.execute(
                model=model,
                prompt=judge_prompt,
                user_input=judge_input,
            )
            import json

            result = json.loads(response.content)

    except Exception as e:
        raise JudgeError(f"Judge evaluation failed: {e}")

    if "score" not in result:
        raise JudgeError(f"Judge response missing 'score': {result}")

    score = result["score"]
    if not isinstance(score, int) or not (score_min <= score <= score_max):
        raise JudgeError(
            f"Invalid score {score}. Must be integer between {score_min} and {score_max}"
        )

    reasoning = result.get("reasoning", "")

    return JudgeResult(
        score=score,
        reasoning=reasoning,
        raw=result,
    )
