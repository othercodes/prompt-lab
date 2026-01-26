from typing import Any

import pytest

from promptlab.domain.contracts.provider import ProviderResponse, ToolCall
from promptlab.infrastructure.providers.base import Provider
from promptlab.infrastructure.providers.factory import get_provider, parse_model_id


class _DummyProvider(Provider):
    @property
    def name(self) -> str:
        return "dummy"

    async def execute(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(content="")

    async def execute_json(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return {}


@pytest.mark.parametrize(
    "model_id,expected_provider,expected_model",
    [
        ("openai:gpt-4o", "openai", "gpt-4o"),
        ("anthropic:claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514"),
    ],
)
def test_parse_model_id_valid(
    model_id: str, expected_provider: str, expected_model: str
):
    provider, model = parse_model_id(model_id)

    assert provider == expected_provider
    assert model == expected_model


def test_parse_model_id_raises_for_invalid():
    with pytest.raises(ValueError, match="Invalid model ID"):
        parse_model_id("gpt-4o")


def test_get_provider_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown")


def test_provider_response_default_values():
    response = ProviderResponse(content="Hello")

    assert response.content == "Hello"
    assert response.tool_calls == []
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.latency_ms == 0


def test_provider_response_with_tool_calls():
    tool_call = ToolCall(name="search", arguments={"query": "test"})
    response = ProviderResponse(
        content="Result",
        tool_calls=[tool_call],
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "search"


def test_provider_format_prompt_with_variables():
    provider = _DummyProvider()
    result = provider.format_prompt(
        "Hello {{ name }}, you are {{ age }}",
        {"name": "Alice", "age": 30},
    )

    assert result == "Hello Alice, you are 30"


def test_provider_format_prompt_raises_for_missing():
    provider = _DummyProvider()

    with pytest.raises(ValueError, match="Missing input variable"):
        provider.format_prompt("Hello {{ name }}", {})
