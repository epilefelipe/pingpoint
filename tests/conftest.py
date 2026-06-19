from pingpoint.models import Profile, Task, Solution, SolutionMetadata, TestResult

import pytest


@pytest.fixture
def low_profile():
    return Profile(
        platform="Linux",
        cpu="Unknown",
        cpu_cores=2,
        ram_gb=4.0,
        gpu=["Unknown"],
        vram_gb=None,
        ollama_models=[],
        ollama_running=True,
        score=5.0,
    )


@pytest.fixture
def medium_profile():
    return Profile(
        platform="Linux",
        cpu="Intel i5",
        cpu_cores=4,
        ram_gb=16.0,
        gpu=["NVIDIA GTX 1060"],
        vram_gb=6.0,
        ollama_models=["llama3.2", "mistral"],
        ollama_running=True,
        score=50.0,
    )


@pytest.fixture
def high_profile():
    return Profile(
        platform="Linux",
        cpu="AMD Ryzen 9",
        cpu_cores=16,
        ram_gb=64.0,
        gpu=["NVIDIA RTX 4090"],
        vram_gb=24.0,
        ollama_models=["llama3.2", "mistral", "llava"],
        ollama_running=True,
        score=90.0,
    )


@pytest.fixture
def sample_task():
    return Task(
        id="test-task",
        title="Test Task",
        description="A test task",
        prompt="Write a Python script to do X",
        test_prompt="Does the solution do X?",
        tags=["code", "reasoning"],
        issue_url="https://github.com/test/repo/issues/1",
        issue_number=1,
    )


@pytest.fixture
def sample_metadata():
    return SolutionMetadata(
        model="llama3.2",
        temperature=0.7,
        max_tokens=2048,
        hardware="Intel i5 | 16GB RAM",
        execution_time_s=10.5,
        ollama_version="0.1.0",
    )


@pytest.fixture
def sample_solution(sample_task, sample_metadata):
    sol = Solution(
        task_id=sample_task.id,
        version=1,
        prompt_used=sample_task.prompt,
        output="print('hello world')",
        metadata=sample_metadata,
        previous_hash=None,
        previous_output=None,
    )
    object.__setattr__(sol, "created_at", "2024-01-01T00:00:00+00:00")
    return sol


@pytest.fixture
def sample_solution_v2(sample_task, sample_metadata, sample_solution):
    sol = Solution(
        task_id=sample_task.id,
        version=2,
        prompt_used="improved prompt",
        output="print('hello world improved')",
        metadata=sample_metadata,
        previous_hash=sample_solution.compute_hash(),
        previous_output=sample_solution.output,
    )
    object.__setattr__(sol, "created_at", "2024-01-02T00:00:00+00:00")
    return sol


@pytest.fixture
def sample_test_result(sample_task):
    return TestResult(
        task_id=sample_task.id,
        version=1,
        passed=True,
        score=85.0,
        summary="Good solution",
        details="Details here",
        improvement_found=True,
    )
