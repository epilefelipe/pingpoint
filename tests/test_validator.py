import json
from pathlib import Path

from pingpoint.db import Database
from pingpoint.validator import validate_task, validate_solution, validate_all


class TestValidateTask:
    def test_valid_task(self, tmp_path):
        path = tmp_path / "task.json"
        data = {
            "id": "test", "title": "T", "description": "D",
            "prompt": "P", "test_prompt": "TP",
        }
        path.write_text(json.dumps(data))
        errors = validate_task(path)
        assert errors == []

    def test_missing_field(self, tmp_path):
        path = tmp_path / "task.json"
        path.write_text(json.dumps({"id": "test"}))
        errors = validate_task(path)
        assert len(errors) >= 1

    def test_null_optional(self, tmp_path):
        path = tmp_path / "task.json"
        data = {
            "id": "test", "title": "T", "description": "D",
            "prompt": "P", "test_prompt": "TP", "tags": None,
        }
        path.write_text(json.dumps(data))
        errors = validate_task(path)
        assert any("'tags' is null" in e for e in errors)

    def test_missing_file(self, tmp_path):
        errors = validate_task(tmp_path / "nonexistent.json")
        assert any("Missing" in e for e in errors)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        errors = validate_task(path)
        assert any("Invalid JSON" in e for e in errors)


class TestValidateSolution:
    def test_valid_solution(self, tmp_path, sample_solution):
        path = tmp_path / "v1.json"
        sol = sample_solution
        data = {
            "task_id": sol.task_id, "version": sol.version,
            "run_number": sol.run_number, "round": sol.round,
            "prompt_used": sol.prompt_used, "output": sol.output,
            "previous_hash": sol.previous_hash,
            "created_at": sol.created_at,
            "hash": sol.compute_hash(),
            "metadata": {
                "model": sol.metadata.model,
                "temperature": sol.metadata.temperature,
                "max_tokens": sol.metadata.max_tokens,
                "hardware": sol.metadata.hardware,
                "execution_time_s": sol.metadata.execution_time_s,
                "ollama_version": sol.metadata.ollama_version,
            },
        }
        path.write_text(json.dumps(data))
        errors = validate_solution(path)
        assert errors == []

    def test_missing_required_field(self, tmp_path):
        path = tmp_path / "v1.json"
        path.write_text(json.dumps({"version": 1}))
        errors = validate_solution(path)
        assert len(errors) >= 1

    def test_invalid_version_type(self, tmp_path):
        path = tmp_path / "v1.json"
        data = {
            "task_id": "t", "version": "one",
            "run_number": 1, "round": 1,
            "prompt_used": "p", "output": "o",
            "previous_hash": None, "hash": None,
            "created_at": "now",
            "metadata": {
                "model": "m", "temperature": 0.7, "max_tokens": 2048,
                "hardware": "h", "execution_time_s": 1.0, "ollama_version": "v",
            },
        }
        path.write_text(json.dumps(data))
        errors = validate_solution(path)
        assert any("version" in e for e in errors)

    def test_invalid_run_number(self, tmp_path):
        path = tmp_path / "v1.json"
        data = {
            "task_id": "t", "version": 1,
            "run_number": 5, "round": 1,
            "prompt_used": "p", "output": "o",
            "previous_hash": None, "hash": None,
            "created_at": "now",
            "metadata": {
                "model": "m", "temperature": 0.7, "max_tokens": 2048,
                "hardware": "h", "execution_time_s": 1.0, "ollama_version": "v",
            },
        }
        path.write_text(json.dumps(data))
        errors = validate_solution(path)
        assert any("run_number" in e for e in errors)

    def test_self_reference_hash(self, tmp_path):
        path = tmp_path / "v2.json"
        h = "abc" + "0" * 61
        data = {
            "task_id": "t", "version": 2,
            "run_number": 1, "round": 1,
            "prompt_used": "p", "output": "o",
            "previous_hash": h, "hash": h,
            "created_at": "now",
            "metadata": {
                "model": "m", "temperature": 0.7, "max_tokens": 2048,
                "hardware": "h", "execution_time_s": 1.0, "ollama_version": "v",
            },
        }
        path.write_text(json.dumps(data))
        errors = validate_solution(path)
        assert any("self-reference" in e for e in errors)

    def test_missing_metadata_field(self, tmp_path):
        path = tmp_path / "v1.json"
        data = {
            "task_id": "t", "version": 1,
            "run_number": 1, "round": 1,
            "prompt_used": "p", "output": "o",
            "previous_hash": None, "hash": None,
            "created_at": "now",
            "metadata": {"model": "m"},
        }
        path.write_text(json.dumps(data))
        errors = validate_solution(path)
        assert any("metadata" in e for e in errors)


class TestValidateAll:
    def test_validate_all_no_solutions(self, tmp_path):
        from pingpoint.validator import validate_all
        result = validate_all("nonexistent")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_all_valid(self, sample_solution, sample_solution_v2):
        db_dir = Path.home() / ".pingpoint"
        db_dir.mkdir(parents=True, exist_ok=True)
        db = Database(db_dir)
        from pingpoint.models import Task
        db.save_task(Task(
            id="test-task", title="Test", description="D",
            prompt="P", test_prompt="TP",
            issue_url="https://example.com", issue_number=1,
        ))

        sol_dir = Path("solutions") / "test-task"
        sol_dir.mkdir(parents=True, exist_ok=True)

        sample_solution_v2.run_number = 2
        for sol in [sample_solution, sample_solution_v2]:
            data = {
                "task_id": sol.task_id, "version": sol.version,
                "run_number": sol.run_number, "round": sol.round,
                "prompt_used": sol.prompt_used, "output": sol.output,
                "previous_hash": sol.previous_hash,
                "created_at": sol.created_at,
                "hash": sol.compute_hash(),
                "metadata": {
                    "model": sol.metadata.model,
                    "temperature": sol.metadata.temperature,
                    "max_tokens": sol.metadata.max_tokens,
                    "hardware": sol.metadata.hardware,
                    "execution_time_s": sol.metadata.execution_time_s,
                    "ollama_version": sol.metadata.ollama_version,
                },
            }
            (sol_dir / f"v{sol.version}.json").write_text(json.dumps(data, indent=2))

        result = validate_all("test-task")
        assert result["valid"] is True
        assert len(result["solutions"]) == 2
