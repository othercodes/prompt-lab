import json
import os
import time
from typing import Any

from anthropic import AsyncAnthropic

from ...domain.contracts.provider import ProviderResponse, ToolCall
from .base import Provider


class AnthropicProvider(Provider):
    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

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

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }

        if system_content:
            kwargs["system"] = system_content

        if tools:
            kwargs["tools"] = self._format_tools(tools)

        start_time = time.perf_counter()
        response = await self.client.messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }

        if system_content:
            kwargs["system"] = system_content

        response = await self.client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return json.loads(content)

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted = []
        for tool in tools:
            formatted.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return formatted
