import json
import re
from typing import Optional

from pingpoint.models import Solution, TestResult
from pingpoint.runner import call_ollama_api, clean_ansi


RUBRIC_PROMPT = """You are evaluating a solution to a task.

TASK:
{task_prompt}

TEST_CRITERIA:
{test_prompt}

SOLUTION:
{solution_output}

PREVIOUS SOLUTION (for comparison):
{previous_output}

Rate the solution on each criterion from 0 to 100.
- CORRECTNESS: Does it correctly solve the task?
- COMPLETENESS: Does it cover all required aspects?
- QUALITY: Is the solution well-structured, clear, and maintainable?
- IMPROVEMENT: How much better is it than the previous solution? (If no previous, score 50)

Respond in this exact format (one per line):
CORRECTNESS: <0-100>
COMPLETENESS: <0-100>
QUALITY: <0-100>
IMPROVEMENT: <0-100>
PASS: yes/no
SUMMARY: <one sentence explaining the evaluation>
"""


def _parse_rubric(output: str) -> dict:
    result = {
        "correctness": 50.0,
        "completeness": 50.0,
        "quality": 50.0,
        "improvement": 50.0,
        "passed": True,
        "summary": output[:300],
    }
    for line in output.split("\n"):
        ls = line.strip()
        lower = ls.lower()
        if lower.startswith("correctness:"):
            try:
                result["correctness"] = float(re.search(r"[\d.]+", ls.split(":")[-1]).group())
            except (AttributeError, ValueError):
                pass
        elif lower.startswith("completeness:"):
            try:
                result["completeness"] = float(re.search(r"[\d.]+", ls.split(":")[-1]).group())
            except (AttributeError, ValueError):
                pass
        elif lower.startswith("quality:"):
            try:
                result["quality"] = float(re.search(r"[\d.]+", ls.split(":")[-1]).group())
            except (AttributeError, ValueError):
                pass
        elif lower.startswith("improvement:"):
            try:
                result["improvement"] = float(re.search(r"[\d.]+", ls.split(":")[-1]).group())
            except (AttributeError, ValueError):
                pass
        elif lower.startswith("pass:"):
            val = ls.split(":")[-1].strip().lower()
            result["passed"] = val == "yes"
        elif lower.startswith("summary:"):
            result["summary"] = ls.split(":", 1)[-1].strip()
    return result


def _weighted_score(rubric: dict) -> float:
    return (
        rubric["correctness"] * 0.35
        + rubric["completeness"] * 0.25
        + rubric["quality"] * 0.20
        + rubric["improvement"] * 0.20
    )


def _call_evaluator(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 512,
    retries: int = 1,
) -> Optional[str]:
    for attempt in range(retries + 1):
        result = call_ollama_api(model, prompt, temperature, max_tokens)
        if result is not None:
            output, _ = result
            return clean_ansi(output)
    return None


def test_solution(
    task_prompt: str,
    task_test_prompt: str,
    new_solution: Solution,
    previous_solution: Optional[Solution] = None,
    model: str = "llama3.2",
    temperature: float = 0.3,
    judge_model: Optional[str] = None,
) -> TestResult:
    eval_model = judge_model or model

    previous_output = previous_solution.output if previous_solution else "(no previous solution)"

    prompt = RUBRIC_PROMPT.format(
        task_prompt=task_prompt,
        test_prompt=task_test_prompt or "(no specific test criteria)",
        solution_output=new_solution.output,
        previous_output=previous_output,
    )

    output = _call_evaluator(prompt, eval_model, temperature)
    if output is None:
        return TestResult(
            task_id=new_solution.task_id,
            version=new_solution.version,
            passed=True,
            score=50.0,
            summary="Could not evaluate, accepted by default",
            details="Evaluator model failed to respond after retries",
            improvement_found=True,
        )

    rubric = _parse_rubric(output)
    final_score = round(_weighted_score(rubric), 1)

    improvement_found = rubric["improvement"] >= 50.0

    return TestResult(
        task_id=new_solution.task_id,
        version=new_solution.version,
        passed=rubric["passed"],
        score=final_score,
        summary=rubric["summary"],
        details=output,
        improvement_found=improvement_found,
    )



