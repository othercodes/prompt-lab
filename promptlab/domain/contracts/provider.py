from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


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


class ProviderConstructor(Protocol):
    def __call__(self, api_key_env_var: str = ...) -> "ProviderContract": ...


class ProviderFactory(Protocol):
    def __call__(
        self, provider_name: str, api_key_env_var: str | None = None
    ) -> "ProviderContract": ...


class ProviderContract(ABC):
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
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        pass

    @abstractmethod
    async def execute_json(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        pass
