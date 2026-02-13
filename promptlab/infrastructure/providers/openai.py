import json
import os
import time
from typing import Any

from openai import AsyncOpenAI

from ...domain.contracts.provider import ProviderResponse, ToolCall
from .base import Provider


class OpenAIProvider(Provider):
    def __init__(self, api_key_env_var: str = "OPENAI_API_KEY") -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"{api_key_env_var} environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    async def execute(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        system_content, user_content = self.build_messages(
            prompt, user_input, system_prompt
        )

        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = self._format_tools(tools)
            kwargs["tool_choice"] = "auto"

        start_time = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        message = response.choices[0].message
        content = message.content or ""

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )

    async def execute_json(
        self,
        model: str,
        prompt: str,
        user_input: dict[str, Any],
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        system_content, user_content = self.build_messages(
            prompt, user_input, system_prompt
        )

        messages: list[dict[str, Any]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        response = await self.client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted = []
        for tool in tools:
            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return formatted
