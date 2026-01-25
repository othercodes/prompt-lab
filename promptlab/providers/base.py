from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from jinja2 import StrictUndefined, Template, UndefinedError


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def execute(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        pass

    def format_prompt(self, prompt: str, user_input: dict[str, Any]) -> str:
        try:
            template = Template(prompt, undefined=StrictUndefined)
            return template.render(**user_input)
        except UndefinedError as e:
            raise ValueError(f"Missing input variable: {e}")


def parse_model_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise ValueError(
            f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
        )

    provider, model = model_id.split(":", 1)
    return provider, model


def get_provider(provider_name: str) -> Provider:
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {', '.join(providers.keys())}"
        )

    return providers[provider_name]()
