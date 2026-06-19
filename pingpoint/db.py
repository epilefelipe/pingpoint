import json
from pathlib import Path
from typing import Optional

from pingpoint.models import Task, Solution, TestResult


class Database:
    def __init__(self, base_path: Path):
        self.base = base_path
        self.base.mkdir(parents=True, exist_ok=True)

    # --- Tasks ---

    def save_task(self, task: Task) -> None:
        task_dir = self.base / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / f"{task.id}.json"
        path.write_text(json.dumps(task.to_dict(), indent=2))

    def load_task(self, task_id: str) -> Optional[Task]:
        path = self.base / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Task(**data)

    def list_tasks(self) -> list[Task]:
        task_dir = self.base / "tasks"
        if not task_dir.exists():
            return []
        tasks = []
        for path in sorted(task_dir.glob("*.json")):
            data = json.loads(path.read_text())
            tasks.append(Task(**data))
        return tasks

    # --- Solutions ---

    def save_solution(self, solution: Solution) -> None:
        sol_dir = self.base / "solutions" / solution.task_id
        sol_dir.mkdir(parents=True, exist_ok=True)
        path = sol_dir / f"v{solution.version}.json"
        path.write_text(json.dumps({
            "task_id": solution.task_id,
            "version": solution.version,
            "prompt_used": solution.prompt_used,
            "output": solution.output,
            "previous_output": solution.previous_output,
            "metadata": {
                "model": solution.metadata.model,
                "temperature": solution.metadata.temperature,
                "max_tokens": solution.metadata.max_tokens,
                "hardware": solution.metadata.hardware,
                "execution_time_s": solution.metadata.execution_time_s,
                "ollama_version": solution.metadata.ollama_version,
            },
            "created_at": solution.created_at,
            "author": solution.author,
        }, indent=2))

    def load_solution(self, task_id: str, version: int) -> Optional[Solution]:
        path = self.base / "solutions" / task_id / f"v{version}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        from pingpoint.models import SolutionMetadata
        return Solution(
            task_id=data["task_id"],
            version=data["version"],
            prompt_used=data["prompt_used"],
            output=data["output"],
            previous_output=data.get("previous_output"),
            metadata=SolutionMetadata(**data["metadata"]),
            created_at=data.get("created_at", ""),
            author=data.get("author"),
        )

    def list_solutions(self, task_id: str) -> list[Solution]:
        sol_dir = self.base / "solutions" / task_id
        if not sol_dir.exists():
            return []
        solutions = []
        for path in sorted(sol_dir.glob("v*.json")):
            data = json.loads(path.read_text())
            from pingpoint.models import SolutionMetadata
            solutions.append(Solution(
                task_id=data["task_id"],
                version=data["version"],
                prompt_used=data["prompt_used"],
                output=data["output"],
                previous_output=data.get("previous_output"),
                metadata=SolutionMetadata(**data["metadata"]),
                created_at=data.get("created_at", ""),
                author=data.get("author"),
            ))
        return solutions

    def latest_solution(self, task_id: str) -> Optional[Solution]:
        solutions = self.list_solutions(task_id)
        return solutions[-1] if solutions else None

    # --- Test results ---

    def save_test_result(self, result: TestResult) -> None:
        test_dir = self.base / "tests" / result.task_id
        test_dir.mkdir(parents=True, exist_ok=True)
        path = test_dir / f"v{result.version}.json"
        path.write_text(json.dumps({
            "task_id": result.task_id,
            "version": result.version,
            "passed": result.passed,
            "score": result.score,
            "summary": result.summary,
            "details": result.details,
            "improvement_found": result.improvement_found,
            "created_at": result.created_at,
        }, indent=2))
