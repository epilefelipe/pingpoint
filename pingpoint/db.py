import json
from pathlib import Path
from typing import Optional

from pingpoint.models import Task, Solution, SolutionMetadata, TestResult


class Database:
    def __init__(self, base_path: Path):
        self.base = base_path
        self.base.mkdir(parents=True, exist_ok=True)

    def repo_solutions_dir(self) -> Path:
        return Path("solutions")

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

    # --- Solutions (stored in repo for tamper-proof chain) ---

    def save_solution(self, solution: Solution) -> None:
        sol_dir = self.repo_solutions_dir() / solution.task_id
        sol_dir.mkdir(parents=True, exist_ok=True)
        path = sol_dir / f"v{solution.version}.json"
        data = {
            "task_id": solution.task_id,
            "version": solution.version,
            "prompt_used": solution.prompt_used,
            "output": solution.output,
            "previous_hash": solution.previous_hash,
            "hash": solution.compute_hash(),
            "previous_output": solution.previous_output,
            "metadata": {
                "model": solution.metadata.model,
                "temperature": solution.metadata.temperature,
                "max_tokens": solution.metadata.max_tokens,
                "hardware": solution.metadata.hardware,
                "execution_time_s": solution.metadata.execution_time_s,
                "ollama_version": solution.metadata.ollama_version,
            },
            "run_number": solution.run_number,
            "round": solution.round,
            "created_at": solution.created_at,
            "author": solution.author,
            "handoff_instructions": solution.handoff_instructions,
        }
        path.write_text(json.dumps(data, indent=2))

    def _solution_from_data(self, data: dict) -> Solution:
        return Solution(
            task_id=data["task_id"],
            version=data["version"],
            prompt_used=data["prompt_used"],
            output=data["output"],
            previous_hash=data.get("previous_hash"),
            previous_output=data.get("previous_output"),
            metadata=SolutionMetadata(**data["metadata"]),
            run_number=data.get("run_number", 1),
            round=data.get("round", 1),
            created_at=data.get("created_at", ""),
            author=data.get("author"),
            handoff_instructions=data.get("handoff_instructions"),
        )

    def verify_chain(self, task_id: str) -> list[str]:
        solutions = self.list_solutions(task_id)
        errors = []
        for sol in solutions:
            path = self.repo_solutions_dir() / task_id / f"v{sol.version}.json"
            data = json.loads(path.read_text())
            stored_hash = data.get("hash")
            computed = sol.compute_hash()
            if stored_hash and stored_hash != computed:
                errors.append(f"v{sol.version}: hash mismatch (tampered)")
            if sol.version > 1:
                prev_path = self.repo_solutions_dir() / task_id / f"v{sol.version - 1}.json"
                if prev_path.exists():
                    prev_data = json.loads(prev_path.read_text())
                    if sol.previous_hash != prev_data.get("hash"):
                        errors.append(f"v{sol.version}: previous_hash chain broken")
        return errors

    def load_solution(self, task_id: str, version: int) -> Optional[Solution]:
        path = self.repo_solutions_dir() / task_id / f"v{version}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return self._solution_from_data(data)

    def list_solutions(self, task_id: str) -> list[Solution]:
        sol_dir = self.repo_solutions_dir() / task_id
        if not sol_dir.exists():
            return []
        solutions = []
        for path in sorted(sol_dir.glob("v*.json")):
            data = json.loads(path.read_text())
            solutions.append(self._solution_from_data(data))
        return solutions

    def latest_solution(self, task_id: str) -> Optional[Solution]:
        solutions = self.list_solutions(task_id)
        return solutions[-1] if solutions else None

    # --- Test results ---

    def save_test_result(self, result: TestResult) -> None:
        test_dir = self.repo_solutions_dir() / result.task_id / "tests"
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

    def list_test_results(self, task_id: str) -> list[TestResult]:
        test_dir = self.repo_solutions_dir() / task_id / "tests"
        if not test_dir.exists():
            return []
        results = []
        for path in sorted(test_dir.glob("v*.json")):
            data = json.loads(path.read_text())
            results.append(TestResult(**data))
        return results
