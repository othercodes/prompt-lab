from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeSpec:
    rubric: str = ""
    model: str = "openai:gpt-4o"
    models: list[str] | None = None
    aggregation: str = "mean"
    score_range: tuple[int, int] = (1, 10)
    temperature: float = 0.0
    chain_of_thought: bool = True


@dataclass
class VariantSpec:
    prompt: str
    system: str | None = None
    description: str = ""
    tools: list[dict[str, Any]] | None = None


@dataclass
class ExperimentSpec:
    name: str
    models: list[str]
    inputs: list[dict[str, Any]]
    variants: dict[str, VariantSpec]
    description: str = ""
    hypothesis: str = ""
    runs: int = 5
    path: str = "experiments"
    judge: JudgeSpec = field(default_factory=JudgeSpec)
