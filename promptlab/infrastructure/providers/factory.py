from ...domain.contracts.provider import ProviderConstructor, ProviderContract


def parse_model_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise ValueError(
            f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
        )

    provider, model = model_id.split(":", 1)
    return provider, model


def _build_registry() -> dict[str, ProviderConstructor]:
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider

    return {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }


def known_providers() -> frozenset[str]:
    return frozenset(_build_registry().keys())


def get_provider(
    provider_name: str, api_key_env_var: str | None = None
) -> ProviderContract:
    registry = _build_registry()

    if provider_name not in registry:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {', '.join(registry.keys())}"
        )

    if api_key_env_var is not None:
        return registry[provider_name](api_key_env_var=api_key_env_var)
    return registry[provider_name]()
