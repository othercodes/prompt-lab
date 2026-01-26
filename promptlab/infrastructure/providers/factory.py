from ...domain.contracts.provider import ProviderContract


def parse_model_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise ValueError(
            f"Invalid model ID '{model_id}'. Expected format: 'provider:model'"
        )

    provider, model = model_id.split(":", 1)
    return provider, model


def get_provider(provider_name: str) -> ProviderContract:
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider

    providers: dict[str, type[ProviderContract]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {', '.join(providers.keys())}"
        )

    return providers[provider_name]()
