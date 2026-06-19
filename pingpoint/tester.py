from pingpoint.models import Solution, TestResult
from pingpoint.runner import call_ollama, clean_ansi


EVALUATOR_PROMPT = """You are evaluating AI solutions. Compare the NEW solution against the PREVIOUS one.

Task: {task_prompt}

Previous solution:
{previous_output}

New solution:
{new_output}

Evaluate:
1. Does the new solution solve the original task? (yes/no)
2. Does the new solution add meaningful improvements over the previous one? (yes/no)
3. Score from 0-100 how much better the new solution is.

Respond in this exact format:
PASS: yes/no
IMPROVEMENT: yes/no
SCORE: 0-100
SUMMARY: One sentence explaining the evaluation
"""


def test_solution(
    task_prompt: str,
    new_solution: Solution,
    previous_solution: Solution | None,
    model: str = "llama3.2",
    temperature: float = 0.3,
) -> TestResult:
    if previous_solution is None:
        prompt = f"""You are evaluating if a solution solves a task.

Task: {task_prompt}

Solution:
{new_solution.output}

Does this solution solve the task? Respond:
PASS: yes/no
SCORE: 0-100
SUMMARY: One sentence explaining why"""

        result = call_ollama(model, prompt, temperature, max_tokens=512)
        if result is None:
            return TestResult(
                task_id=new_solution.task_id,
                version=new_solution.version,
                passed=True,
                score=50.0,
                summary="Could not evaluate, accepted by default",
                details="Evaluator model failed to respond",
                improvement_found=True,
            )

        output, _ = result
        output = clean_ansi(output)
        passed = "PASS: yes" in output.lower()

        score = 50.0
        for line in output.split("\n"):
            if "SCORE:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

        return TestResult(
            task_id=new_solution.task_id,
            version=new_solution.version,
            passed=passed,
            score=score,
            summary=output[:300],
            details=output,
            improvement_found=True,
        )

    prompt = EVALUATOR_PROMPT.format(
        task_prompt=task_prompt,
        previous_output=previous_solution.output,
        new_output=new_solution.output,
    )

    result = call_ollama(model, prompt, temperature, max_tokens=512)
    if result is None:
        return TestResult(
            task_id=new_solution.task_id,
            version=new_solution.version,
            passed=True,
            score=50.0,
            summary="Could not evaluate, accepted by default",
            details="Evaluator model failed to respond",
            improvement_found=True,
        )

    output, _ = result
    output = clean_ansi(output)
    lines = output.lower().split("\n")

    passed = any("pass: yes" in line for line in lines)
    improvement = any("improvement: yes" in line for line in lines)

    score = 50.0
    for line in lines:
        if "score:" in line:
            try:
                score = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    summary = output
    for line in output.split("\n"):
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
            break

    return TestResult(
        task_id=new_solution.task_id,
        version=new_solution.version,
        passed=passed,
        score=score,
        summary=summary,
        details=output,
        improvement_found=improvement,
    )
