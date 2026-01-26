import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..domain.contracts.cache import CacheContract
from ..domain.contracts.provider import ProviderResponse, ToolCall

DEFAULT_CACHE_DIR = Path(".cache")


class FileCache(CacheContract):
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def make_key(
        self,
        prompt: str,
        input_data: dict[str, Any],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        key_data = {
            "prompt": prompt,
            "input": input_data,
            "model": model,
            "tools": tools or [],
        }
        serialized = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, key: str) -> ProviderResponse | None:
        path = self._key_path(key)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        tool_calls = [
            ToolCall(name=tc["name"], arguments=tc["arguments"])
            for tc in data.get("tool_calls", [])
        ]

        return ProviderResponse(
            content=data["content"],
            tool_calls=tool_calls,
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            latency_ms=data["latency_ms"],
            raw=data.get("raw", {}),
        )

    def put(self, key: str, response: ProviderResponse) -> None:
        data = {
            "content": response.content,
            "tool_calls": [asdict(tc) for tc in response.tool_calls],
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "raw": response.raw,
        }
        with open(self._key_path(key), "w") as f:
            json.dump(data, f)

    def has(self, key: str) -> bool:
        return self._key_path(key).exists()

    def clear(self) -> None:
        for path in self._cache_dir.glob("*.json"):
            path.unlink()
