from pathlib import Path
from typing import Generator

import pytest

from promptlab.domain.contracts.provider import ProviderResponse, ToolCall
from promptlab.infrastructure.file_cache import FileCache


@pytest.fixture
def cache(tmp_path: Path) -> Generator[FileCache, None, None]:
    yield FileCache(tmp_path / ".cache")


def test_make_key_is_deterministic(cache: FileCache):
    key1 = cache.make_key(
        prompt="Hello {name}",
        input_data={"name": "World"},
        model="openai:gpt-4o",
    )
    key2 = cache.make_key(
        prompt="Hello {name}",
        input_data={"name": "World"},
        model="openai:gpt-4o",
    )

    assert key1 == key2


def test_make_key_differs_for_different_inputs(cache: FileCache):
    key1 = cache.make_key(
        prompt="Hello {name}",
        input_data={"name": "World"},
        model="openai:gpt-4o",
    )
    key2 = cache.make_key(
        prompt="Hello {name}",
        input_data={"name": "Universe"},
        model="openai:gpt-4o",
    )

    assert key1 != key2


def test_put_and_get(cache: FileCache):
    key = cache.make_key(
        prompt="test",
        input_data={},
        model="openai:gpt-4o",
    )
    response = ProviderResponse(
        content="Hello!",
        tool_calls=[ToolCall(name="search", arguments={"q": "test"})],
        input_tokens=10,
        output_tokens=5,
        latency_ms=100,
    )

    cache.put(key, response)
    retrieved = cache.get(key)

    assert retrieved is not None
    assert retrieved.content == "Hello!"
    assert len(retrieved.tool_calls) == 1
    assert retrieved.tool_calls[0].name == "search"
    assert retrieved.input_tokens == 10


def test_get_returns_none_for_missing_key(cache: FileCache):
    result = cache.get("nonexistent")
    assert result is None


def test_clear_removes_all_entries(cache: FileCache):
    key = cache.make_key(prompt="test", input_data={}, model="test")
    response = ProviderResponse(content="test")
    cache.put(key, response)

    cache.clear()

    assert cache.get(key) is None
