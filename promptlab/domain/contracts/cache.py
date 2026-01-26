from abc import ABC, abstractmethod
from typing import Any

from .provider import ProviderResponse


class CacheContract(ABC):
    @abstractmethod
    def make_key(
        self,
        prompt: str,
        input_data: dict[str, Any],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        pass

    @abstractmethod
    def get(self, key: str) -> ProviderResponse | None:
        pass

    @abstractmethod
    def put(self, key: str, response: ProviderResponse) -> None:
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
