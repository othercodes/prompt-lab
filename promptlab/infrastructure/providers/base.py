from abc import ABC

from jinja2 import StrictUndefined, Template, UndefinedError

from ...domain.contracts.provider import ProviderContract, ProviderResponse, ToolCall
from .factory import get_provider, parse_model_id

# Re-export for backward compatibility
__all__ = [
    "Provider",
    "ProviderContract",
    "ProviderResponse",
    "ToolCall",
    "get_provider",
    "parse_model_id",
]


class Provider(ProviderContract, ABC):
    def format_prompt(self, prompt: str, user_input: dict[str, object]) -> str:
        try:
            template = Template(prompt, undefined=StrictUndefined)
            return template.render(**user_input)
        except UndefinedError as e:
            raise ValueError(f"Missing input variable: {e}")
