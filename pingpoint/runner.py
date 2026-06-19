import subprocess
import time
from typing import Optional, Tuple

from pingpoint.models import Solution, SolutionMetadata


def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> Optional[Tuple[str, float]]:
    try:
        start = time.time()
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output:
            return None

        return output, elapsed

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_ollama_version() -> str:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "not installed"


def run_solution(
    model: str,
    task_prompt: str,
    improvement_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    hardware_summary: str = "Unknown",
) -> Optional[Solution]:
    prompt = improvement_prompt if improvement_prompt else task_prompt

    result = call_ollama(model, prompt, temperature, max_tokens)
    if result is None:
        return None

    output, elapsed = result

    metadata = SolutionMetadata(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        hardware=hardware_summary,
        execution_time_s=round(elapsed, 1),
        ollama_version=get_ollama_version(),
    )

    return Solution(
        task_id="",
        version=0,
        prompt_used=prompt,
        output=output,
        metadata=metadata,
        previous_output=None,
    )
