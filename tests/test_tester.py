import json
from unittest.mock import patch, MagicMock

from pingpoint.models import Solution, SolutionMetadata
from pingpoint.tester import _parse_rubric
from pingpoint import tester as tester_module


class TestParseRubric:
    def test_parses_all_fields(self):
        output = """CORRECTNESS: 85
COMPLETENESS: 70
QUALITY: 90
IMPROVEMENT: 60
PASS: yes
SUMMARY: A solid solution with minor gaps."""
        r = _parse_rubric(output)
        assert r["correctness"] == 85
        assert r["completeness"] == 70
        assert r["quality"] == 90
        assert r["improvement"] == 60
        assert r["passed"] is True
        assert "solid solution" in r["summary"]

    def test_parses_fail(self):
        output = """CORRECTNESS: 30
COMPLETENESS: 20
QUALITY: 40
IMPROVEMENT: 10
PASS: no
SUMMARY: Does not solve the task."""
        r = _parse_rubric(output)
        assert r["passed"] is False
        assert r["correctness"] == 30

    def test_defaults_on_missing(self):
        output = "PASS: yes\nSOME_OTHER: thing"
        r = _parse_rubric(output)
        assert r["correctness"] == 50.0
        assert r["passed"] is True

    def test_scores_at_boundaries(self):
        output = """CORRECTNESS: 0
COMPLETENESS: 100
QUALITY: 50
IMPROVEMENT: 100
PASS: yes
SUMMARY: Test."""
        r = _parse_rubric(output)
        assert r["correctness"] == 0
        assert r["completeness"] == 100


class TestTestSolution:
    def _make_solution(self, task_id="test-task", version=1, output="print('hello')"):
        return Solution(
            task_id=task_id,
            version=version,
            prompt_used="prompt",
            output=output,
            metadata=SolutionMetadata(
                model="llama3.2", temperature=0.7, max_tokens=2048,
                hardware="CPU", execution_time_s=1.0, ollama_version="0.1.0",
            ),
        )

    @patch.object(tester_module, "call_ollama_api")
    def test_basic_evaluation(self, mock_api):
        mock_api.return_value = ("""CORRECTNESS: 90
COMPLETENESS: 80
QUALITY: 85
IMPROVEMENT: 50
PASS: yes
SUMMARY: Good solution.""", 2.0)

        new = self._make_solution()
        result = tester_module.test_solution(
            task_prompt="Write code",
            task_test_prompt="Should work",
            new_solution=new,
        )
        assert result.passed is True
        assert 70.0 <= result.score <= 85.0
        assert result.summary == "Good solution."

    @patch.object(tester_module, "call_ollama_api")
    def test_with_previous_solution(self, mock_api):
        mock_api.return_value = ("""CORRECTNESS: 95
COMPLETENESS: 90
QUALITY: 90
IMPROVEMENT: 80
PASS: yes
SUMMARY: Clearly improved.""", 2.0)

        new = self._make_solution(version=2, output="improved")
        prev = self._make_solution(version=1, output="original")
        result = tester_module.test_solution(
            task_prompt="Write code",
            task_test_prompt="Should work",
            new_solution=new,
            previous_solution=prev,
        )
        assert result.passed is True
        assert result.improvement_found is True

    @patch.object(tester_module, "call_ollama_api")
    def test_no_improvement_detected(self, mock_api):
        mock_api.return_value = ("""CORRECTNESS: 60
COMPLETENESS: 50
QUALITY: 50
IMPROVEMENT: 30
PASS: yes
SUMMARY: Slightly worse.""", 2.0)

        new = self._make_solution(version=2)
        prev = self._make_solution(version=1)
        result = tester_module.test_solution(
            task_prompt="Write code",
            task_test_prompt="Should work",
            new_solution=new,
            previous_solution=prev,
        )
        assert result.improvement_found is False

    @patch.object(tester_module, "call_ollama_api")
    def test_evaluator_failure_fallback(self, mock_api):
        mock_api.return_value = None

        new = self._make_solution()
        result = tester_module.test_solution(
            task_prompt="Write code",
            task_test_prompt="Should work",
            new_solution=new,
        )
        assert result.passed is True
        assert result.score == 50.0
        assert "Could not evaluate" in result.summary

    @patch.object(tester_module, "call_ollama_api")
    def test_uses_judge_model(self, mock_api):
        mock_api.return_value = ("""CORRECTNESS: 80
COMPLETENESS: 80
QUALITY: 80
IMPROVEMENT: 50
PASS: yes
SUMMARY: OK.""", 1.0)

        new = self._make_solution()
        tester_module.test_solution(
            task_prompt="Write code",
            task_test_prompt="Should work",
            new_solution=new,
            judge_model="mixtral",
        )
        args, kwargs = mock_api.call_args
        assert args[0] == "mixtral"
