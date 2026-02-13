from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from promptlab.domain.contracts.config import ExperimentConfig
from promptlab.domain.contracts.provider import ProviderResponse, ToolCall
from promptlab.infrastructure.providers.anthropic import AnthropicProvider
from promptlab.infrastructure.providers.base import Provider
from promptlab.infrastructure.providers.factory import (
    get_provider,
    known_providers,
    parse_model_id,
)
from promptlab.infrastructure.providers.openai import OpenAIProvider


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
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(content="")

    async def execute_json(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        temperature: float = 0.0,
        system_prompt: str | None = None,
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


def test_build_messages_without_system():
    provider = _DummyProvider()
    system, user = provider.build_messages("Hello {{ name }}", {"name": "Alice"})

    assert system is None
    assert user == "Hello Alice"


def test_build_messages_with_system():
    provider = _DummyProvider()
    system, user = provider.build_messages(
        "Translate: {{ text }}",
        {"text": "hello", "role": "translator"},
        system_prompt="You are a {{ role }}",
    )

    assert system == "You are a translator"
    assert user == "Translate: hello"


def test_build_messages_hardcoded_no_vars():
    provider = _DummyProvider()
    system, user = provider.build_messages("Tell me a joke", {})

    assert system is None
    assert user == "Tell me a joke"


# Tests for custom key_refs feature


@patch("promptlab.infrastructure.providers.openai.AsyncOpenAI")
def test_openai_provider_custom_env_var(
    mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MY_OPENAI_KEY", "test-key-123")

    provider = OpenAIProvider(api_key_env_var="MY_OPENAI_KEY")

    assert provider.name == "openai"
    mock_openai_client.assert_called_once_with(api_key="test-key-123")


@patch("promptlab.infrastructure.providers.anthropic.AsyncAnthropic")
def test_anthropic_provider_custom_env_var(
    mock_anthropic_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MY_ANTHROPIC_KEY", "test-key-456")

    provider = AnthropicProvider(api_key_env_var="MY_ANTHROPIC_KEY")

    assert provider.name == "anthropic"
    mock_anthropic_client.assert_called_once_with(api_key="test-key-456")


@patch("promptlab.infrastructure.providers.openai.AsyncOpenAI")
def test_openai_provider_default_env_var(
    mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "default-key-123")

    provider = OpenAIProvider()

    assert provider.name == "openai"
    mock_openai_client.assert_called_once_with(api_key="default-key-123")


@patch("promptlab.infrastructure.providers.anthropic.AsyncAnthropic")
def test_anthropic_provider_default_env_var(
    mock_anthropic_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "default-key-456")

    provider = AnthropicProvider()

    assert provider.name == "anthropic"
    mock_anthropic_client.assert_called_once_with(api_key="default-key-456")


def test_provider_custom_env_var_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)

    with pytest.raises(ValueError, match="MY_CUSTOM_KEY environment variable not set"):
        OpenAIProvider(api_key_env_var="MY_CUSTOM_KEY")


@patch("promptlab.infrastructure.providers.openai.AsyncOpenAI")
def test_get_provider_forwards_api_key_env_var(
    mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MY_KEY", "forwarded-key-789")

    provider = get_provider("openai", api_key_env_var="MY_KEY")

    assert provider.name == "openai"
    mock_openai_client.assert_called_once_with(api_key="forwarded-key-789")


@patch("promptlab.infrastructure.providers.openai.AsyncOpenAI")
def test_get_provider_no_env_var_uses_default(
    mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "default-factory-key")

    provider = get_provider("openai")

    assert provider.name == "openai"
    mock_openai_client.assert_called_once_with(api_key="default-factory-key")


def test_experiment_config_key_refs_default():
    config = ExperimentConfig(
        name="test-experiment",
        description="Test description",
        models=["openai:gpt-4o"],
    )

    assert config.key_refs == {}
    assert isinstance(config.key_refs, dict)


def test_key_refs_merge_cli_overrides_config():
    """CLI key_refs take precedence over experiment.md key_refs."""
    config_refs = {"openai": "CONFIG_OPENAI_KEY", "anthropic": "CONFIG_ANTHROPIC_KEY"}
    cli_refs = {"openai": "CLI_OPENAI_KEY"}

    merged = {**config_refs, **cli_refs}

    assert merged["openai"] == "CLI_OPENAI_KEY"
    assert merged["anthropic"] == "CONFIG_ANTHROPIC_KEY"


# --- known_providers ---


def test_known_providers_returns_frozenset():
    result = known_providers()

    assert isinstance(result, frozenset)


def test_known_providers_contains_registered():
    result = known_providers()

    assert "openai" in result
    assert "anthropic" in result


def test_known_providers_excludes_unknown():
    result = known_providers()

    assert "unknown" not in result
    assert "gemini" not in result
