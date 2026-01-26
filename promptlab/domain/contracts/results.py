from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    input_id: str
    model: str
    run_number: int
    cached: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    response: dict[str, Any]
    judge: dict[str, Any]


@dataclass
class InputStats:
    input_id: str
    model: str
    runs: int
    scores: list[int]
    mean: float
    stddev: float
    min_score: int
    max_score: int
    ci_lower: float
    ci_upper: float


@dataclass
class RunSummary:
    timestamp: str
    experiment: str
    variant: str
    models: list[str]
    inputs_count: int
    runs_per_input: int
    duration_seconds: float
    cached_responses: int
    hypothesis: str = ""
    results: list[RunResult] = field(default_factory=list)
    stats: list[InputStats] = field(default_factory=list)


class ResultRepositoryContract(ABC):
    @abstractmethod
    def save(self, variant_path: Path, summary: RunSummary) -> Path:
        pass

    @abstractmethod
    def load(self, variant_path: Path, run_timestamp: str | None = None) -> RunSummary:
        pass

    @abstractmethod
    def list_runs(self, variant_path: Path) -> list[str]:
        pass
