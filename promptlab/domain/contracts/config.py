from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    name: str
    description: str
    models: list[str]
    hypothesis: str = ""
    runs: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    content: str
    models: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    content: str
    model: str = "openai:gpt-4o"
    models: list[str] | None = None  # Multi-judge: list of models (opt-in)
    aggregation: str = "mean"  # mean | median (only used with multi-judge)
    score_range: tuple[int, int] = (1, 10)
    temperature: float = 0.0
    chain_of_thought: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_judge(self) -> bool:
        """Check if multi-judge mode is enabled."""
        return self.models is not None and len(self.models) > 1

    @property
    def judge_models(self) -> list[str]:
        """Return list of judge models (single or multi)."""
        if self.models:
            return self.models
        return [self.model]


@dataclass
class InputCase:
    id: str
    data: dict[str, Any]
    runs: int | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class VariantConfig:
    path: Path
    experiment: ExperimentConfig
    prompt: PromptConfig
    judge: JudgeConfig
    inputs: list[InputCase]
    tools: list[ToolDefinition] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return self.prompt.models or self.experiment.models


class ConfigLoaderContract(ABC):
    @abstractmethod
    def load_experiment(self, path: Path) -> ExperimentConfig:
        pass

    @abstractmethod
    def load_variant(self, variant_path: Path) -> VariantConfig:
        pass

    @abstractmethod
    def discover_variants(self, experiment_path: Path) -> list[Path]:
        pass
