import json
import os
import time
from typing import Any

from anthropic import AsyncAnthropic

from .base import Provider, ProviderResponse, ToolCall


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
    ) -> ProviderResponse:
        formatted_prompt = self.format_prompt(prompt, user_input)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": formatted_prompt,
            "messages": [
                {"role": "user", "content": json.dumps(user_input)},
            ],
        }

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
    ) -> dict[str, Any]:
        formatted_prompt = self.format_prompt(prompt, user_input)

        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=formatted_prompt,
            messages=[
                {"role": "user", "content": json.dumps(user_input)},
            ],
            temperature=temperature,
        )

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
