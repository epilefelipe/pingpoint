import json
import os
from pathlib import Path

import pytest

from pingpoint.db import Database
from pingpoint.models import Task, Solution, SolutionMetadata, TestResult


@pytest.fixture
def isolated_solutions(tmp_path):
    old = Path.cwd()
    os.chdir(tmp_path)
    yield
    os.chdir(old)


class TestDatabaseTasks:
    def test_save_and_load_task(self, tmp_path, sample_task):
        db = Database(tmp_path)
        db.save_task(sample_task)
        loaded = db.load_task("test-task")
        assert loaded is not None
        assert loaded.id == "test-task"
        assert loaded.title == "Test Task"
        assert loaded.tags == ["code", "reasoning"]

    def test_list_tasks(self, tmp_path, sample_task):
        db = Database(tmp_path)
        db.save_task(sample_task)
        tasks = db.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "test-task"

    def test_list_tasks_empty(self, tmp_path):
        db = Database(tmp_path)
        assert db.list_tasks() == []

    def test_load_nonexistent(self, tmp_path):
        db = Database(tmp_path)
        assert db.load_task("nonexistent") is None

    def test_save_task_creates_directory(self, tmp_path, sample_task):
        db = Database(tmp_path)
        db.save_task(sample_task)
        assert (tmp_path / "db.json").exists()
        loaded = db.load_task("test-task")
        assert loaded is not None
        assert loaded.title == "Test Task"


@pytest.mark.usefixtures("isolated_solutions")
class TestDatabaseSolutions:
    def test_save_and_load_solution(self, tmp_path, sample_solution):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        loaded = db.load_solution("test-task", 1)
        assert loaded is not None
        assert loaded.version == 1
        assert loaded.output == "print('hello world')"
        assert loaded.metadata.model == "llama3.2"

    def test_includes_hash_in_file(self, tmp_path, sample_solution):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        path = tmp_path / ".." / "solutions" / "test-task" / "v1.json"
        # resolve relative path: Database.repo_solutions_dir returns Path("solutions")
        # so relative to cwd. We'll find it.
        sol_dir = Path("solutions") / "test-task"
        saved_path = sol_dir / "v1.json"
        assert saved_path.exists()
        data = json.loads(saved_path.read_text())
        assert "hash" in data
        assert data["task_id"] == "test-task"
        assert data["version"] == 1

    def test_list_solutions(self, tmp_path, sample_solution, sample_solution_v2):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        db.save_solution(sample_solution_v2)
        solutions = db.list_solutions("test-task")
        assert len(solutions) == 2
        assert solutions[0].version == 1
        assert solutions[1].version == 2

    def test_list_solutions_empty(self, tmp_path):
        db = Database(tmp_path)
        assert db.list_solutions("nonexistent") == []

    def test_latest_solution(self, tmp_path, sample_solution, sample_solution_v2):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        db.save_solution(sample_solution_v2)
        latest = db.latest_solution("test-task")
        assert latest is not None
        assert latest.version == 2

    def test_latest_solution_single(self, tmp_path, sample_solution):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        latest = db.latest_solution("test-task")
        assert latest is not None
        assert latest.version == 1

    def test_latest_solution_none(self, tmp_path):
        db = Database(tmp_path)
        assert db.latest_solution("test-task") is None

    def test_load_nonexistent_version(self, tmp_path):
        db = Database(tmp_path)
        assert db.load_solution("test-task", 99) is None


@pytest.mark.usefixtures("isolated_solutions")
class TestDatabaseVerifyChain:
    def test_verify_valid_chain(self, tmp_path, sample_solution, sample_solution_v2):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        db.save_solution(sample_solution_v2)
        errors = db.verify_chain("test-task")
        assert errors == []

    def test_verify_single_solution(self, tmp_path, sample_solution):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        errors = db.verify_chain("test-task")
        assert errors == []

    def test_verify_tampered_hash(self, tmp_path, sample_solution, sample_solution_v2):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        db.save_solution(sample_solution_v2)
        # Tamper with v2's stored hash
        path = Path("solutions") / "test-task" / "v2.json"
        data = json.loads(path.read_text())
        data["hash"] = "tampered" + "0" * 58
        path.write_text(json.dumps(data, indent=2))
        errors = db.verify_chain("test-task")
        assert len(errors) >= 1
        assert any("hash mismatch" in e for e in errors)

    def test_verify_broken_chain(self, tmp_path, sample_solution, sample_solution_v2):
        db = Database(tmp_path)
        db.save_solution(sample_solution)
        # Manually break v2's previous_hash
        sol2 = sample_solution_v2
        sol2.previous_hash = "wronghash"
        db.save_solution(sol2)
        errors = db.verify_chain("test-task")
        assert any("chain broken" in e for e in errors)

    def test_verify_empty(self, tmp_path):
        db = Database(tmp_path)
        errors = db.verify_chain("nonexistent")
        assert errors == []


@pytest.mark.usefixtures("isolated_solutions")
class TestDatabaseTestResults:
    def test_save_and_list_test_results(self, tmp_path, sample_test_result):
        db = Database(tmp_path)
        db.save_test_result(sample_test_result)
        results = db.list_test_results("test-task")
        assert len(results) == 1
        assert results[0].version == 1
        assert results[0].passed is True
        assert results[0].score == 85.0

    def test_list_test_results_empty(self, tmp_path):
        db = Database(tmp_path)
        assert db.list_test_results("test-task") == []

    def test_test_result_file_created(self, tmp_path, sample_test_result):
        db = Database(tmp_path)
        db.save_test_result(sample_test_result)
        path = Path("solutions") / "test-task" / "tests" / "v1.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["passed"] is True
