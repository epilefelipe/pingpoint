from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Profile:
    platform: str
    cpu: str
    cpu_cores: int
    ram_gb: float
    gpu: list[str]
    vram_gb: Optional[float]
    ollama_models: list[str]
    ollama_running: bool
    score: float = 0.0

    @property
    def has_gpu(self) -> bool:
        return len(self.gpu) > 0 and self.gpu[0] != "Unknown"

    @property
    def capability(self) -> str:
        if self.has_gpu and self.vram_gb and self.vram_gb >= 16:
            return "high"
        if self.has_gpu:
            return "medium"
        return "low"


@dataclass
class Task:
    id: str
    title: str
    description: str
    prompt: str
    test_prompt: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    issue_url: Optional[str] = None
    issue_number: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SolutionMetadata:
    model: str
    temperature: float
    max_tokens: int
    hardware: str
    execution_time_s: float
    ollama_version: str


@dataclass
class Solution:
    task_id: str
    version: int
    prompt_used: str
    output: str
    metadata: SolutionMetadata
    previous_output: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    author: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.task_id}/v{self.version}"


@dataclass
class TestResult:
    task_id: str
    version: int
    passed: bool
    score: float
    summary: str
    details: str
    improvement_found: bool
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def id(self) -> str:
        return f"{self.task_id}/v{self.version}/test"
