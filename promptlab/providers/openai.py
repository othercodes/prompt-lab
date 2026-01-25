import json
import os
import time
from typing import Any

from openai import AsyncOpenAI

from .base import Provider, ProviderResponse, ToolCall


class OpenAIProvider(Provider):
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
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
    ) -> ProviderResponse:
        formatted_prompt = self.format_prompt(prompt, user_input)

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": json.dumps(user_input)},
        ]

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
    ) -> dict[str, Any]:
        formatted_prompt = self.format_prompt(prompt, user_input)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": json.dumps(user_input)},
        ]

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
