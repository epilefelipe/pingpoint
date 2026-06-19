import json
from pathlib import Path
from typing import Optional

from pingpoint.models import Task, Solution, SolutionMetadata, TestResult, compute_solution_hash


def _solution_to_dict(solution: Solution, hash_val: str) -> dict:
    return {
        "task_id": solution.task_id,
        "version": solution.version,
        "prompt_used": solution.prompt_used,
        "output": solution.output,
        "previous_hash": solution.previous_hash,
        "hash": hash_val,
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


class Database:
    def __init__(self, base_path: Path):
        self.base = base_path
        self.base.mkdir(parents=True, exist_ok=True)
        self._db_path = self.base / "db.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._db_path.exists():
            try:
                return json.loads(self._db_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"tasks": {}, "solutions": {}, "test_results": {}}

    def _save(self) -> None:
        self._db_path.write_text(json.dumps(self._data, indent=2, default=str))

    def repo_solutions_dir(self) -> Path:
        return Path("solutions")

    # --- Tasks ---

    def save_task(self, task: Task) -> None:
        self._data.setdefault("tasks", {})[task.id] = task.to_dict()
        self._save()

    def load_task(self, task_id: str) -> Optional[Task]:
        raw = self._data.get("tasks", {}).get(task_id)
        if raw is None:
            return None
        return Task(**raw)

    def list_tasks(self) -> list[Task]:
        return [Task(**raw) for raw in self._data.get("tasks", {}).values()]

    # --- Solutions (also written to repo for tamper-proof chain) ---

    def save_solution(self, solution: Solution) -> None:
        hash_val = solution.compute_hash()
        sol_dict = _solution_to_dict(solution, hash_val)

        sol_dir = self.repo_solutions_dir() / solution.task_id
        sol_dir.mkdir(parents=True, exist_ok=True)
        (sol_dir / f"v{solution.version}.json").write_text(
            json.dumps(sol_dict, indent=2)
        )

        self._data.setdefault("solutions", {}).setdefault(solution.task_id, [])
        existing = [s for s in self._data["solutions"][solution.task_id]
                    if s.get("version") == solution.version]
        if existing:
            existing[0] = sol_dict
        else:
            self._data["solutions"][solution.task_id].append(sol_dict)
        self._save()

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
        errors = []
        sol_dir = self.repo_solutions_dir() / task_id
        if not sol_dir.exists():
            return errors
        for path in sorted(sol_dir.glob("v*.json")):
            data = json.loads(path.read_text())
            stored_hash = data.get("hash")
            meta = data.get("metadata", {})
            computed = compute_solution_hash(
                task_id=data.get("task_id", ""),
                version=data.get("version", 0),
                run_number=data.get("run_number", 1),
                round=data.get("round", 1),
                prompt_used=data.get("prompt_used", ""),
                output=data.get("output", ""),
                previous_hash=data.get("previous_hash"),
                created_at=data.get("created_at", ""),
                model=meta.get("model", ""),
                hardware=meta.get("hardware", ""),
                handoff_instructions=data.get("handoff_instructions"),
            )
            if stored_hash and stored_hash != computed:
                errors.append(f"{path.name}: hash mismatch (tampered)")
            ver = data.get("version", 0)
            if ver > 1:
                prev_path = self.repo_solutions_dir() / task_id / f"v{ver - 1}.json"
                if prev_path.exists():
                    prev_data = json.loads(prev_path.read_text())
                    if data.get("previous_hash") != prev_data.get("hash"):
                        errors.append(f"{path.name}: previous_hash chain broken")
        return errors

    def load_solution(self, task_id: str, version: int) -> Optional[Solution]:
        sols = self._data.get("solutions", {}).get(task_id, [])
        for s in sols:
            if s.get("version") == version:
                return self._solution_from_data(s)
        path = self.repo_solutions_dir() / task_id / f"v{version}.json"
        if path.exists():
            return self._solution_from_data(json.loads(path.read_text()))
        return None

    def list_solutions(self, task_id: str) -> list[Solution]:
        return sorted(
            (self._solution_from_data(s) for s in self._data.get("solutions", {}).get(task_id, [])),
            key=lambda s: s.version,
        )

    def latest_solution(self, task_id: str) -> Optional[Solution]:
        solutions = self.list_solutions(task_id)
        return solutions[-1] if solutions else None

    # --- Test results ---

    def save_test_result(self, result: TestResult) -> None:
        tr_dict = {
            "task_id": result.task_id,
            "version": result.version,
            "passed": result.passed,
            "score": result.score,
            "summary": result.summary,
            "details": result.details,
            "improvement_found": result.improvement_found,
            "created_at": result.created_at,
        }

        test_dir = self.repo_solutions_dir() / result.task_id / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"v{result.version}.json").write_text(json.dumps(tr_dict, indent=2))

        self._data.setdefault("test_results", {}).setdefault(result.task_id, [])
        existing = [t for t in self._data["test_results"][result.task_id]
                    if t.get("version") == result.version]
        if existing:
            existing[0] = tr_dict
        else:
            self._data["test_results"][result.task_id].append(tr_dict)
        self._save()

    # --- Verifications (social trust) ---

    def _verifications_path(self, task_id: str) -> Path:
        return self.repo_solutions_dir() / task_id / "verifications.json"

    def add_verification(self, task_id: str, verifier: str) -> dict:
        from datetime import datetime, timezone
        entry = {"verifier": verifier, "timestamp": datetime.now(timezone.utc).isoformat(), "result": "valid"}
        path = self._verifications_path(task_id)
        existing = []
        if path.exists():
            existing = json.loads(path.read_text())
        existing.append(entry)
        path.write_text(json.dumps(existing, indent=2))
        return entry

    def list_verifications(self, task_id: str) -> list[dict]:
        path = self._verifications_path(task_id)
        if path.exists():
            return json.loads(path.read_text())
        return []

    def list_test_results(self, task_id: str) -> list[TestResult]:
        return [
            TestResult(**t)
            for t in sorted(
                self._data.get("test_results", {}).get(task_id, []),
                key=lambda x: x.get("version", 0),
            )
        ]
