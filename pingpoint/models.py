import hashlib
import json
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
    previous_hash: Optional[str] = None
    previous_output: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_number: int = 1  # 1-3 within a round
    round: int = 1
    author: Optional[str] = None
    handoff_instructions: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.task_id}/v{self.version}"

    def compute_hash(self) -> str:
        data = {
            "task_id": self.task_id,
            "version": self.version,
            "run_number": self.run_number,
            "round": self.round,
            "prompt_used": self.prompt_used,
            "output": self.output,
            "previous_hash": self.previous_hash,
            "created_at": self.created_at,
            "model": self.metadata.model,
            "hardware": self.metadata.hardware,
            "handoff_instructions": self.handoff_instructions,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


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
