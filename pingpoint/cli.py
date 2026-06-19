import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from pingpoint import __version__
from pingpoint.db import Database
from pingpoint.models import Task, Solution, SolutionMetadata
from pingpoint.profiler import profile as get_profile
from pingpoint.matcher import find_best_task
from pingpoint.runner import call_ollama, clean_ansi, get_ollama_version as get_ov
from pingpoint.tester import test_solution
from pingpoint.validator import validate_all, print_validation

app = typer.Typer()

DATA_DIR = Path.home() / ".pingpoint"
db = Database(DATA_DIR)


@app.command()
def version():
    """Show pingpoint version."""
    print(f"pingpoint v{__version__}")


@app.command()
def profile():
    """Detect your hardware and available AI models."""
    print("Analyzing your machine...")
    p = get_profile()

    print(f"\nPlatform:       {p.platform}")
    print(f"CPU:            {p.cpu}")
    print(f"Cores:          {p.cpu_cores}")
    print(f"RAM:            {p.ram_gb} GB")
    print(f"GPU:            {', '.join(p.gpu)}")
    if p.vram_gb:
        print(f"VRAM:           {p.vram_gb} GB")
    print(f"Capability:     {p.capability.upper()}")
    print(f"Score:          {p.score}")
    print(f"Ollama running: {'Yes' if p.ollama_running else 'No'}")
    print(f"Ollama models:  {', '.join(p.ollama_models) or 'None'}")

    if not p.ollama_running:
        print("\nOllama is not running. Start it with: ollama serve")
    if not p.ollama_models:
        print("\nNo models found. Pull one: ollama pull llama3.2")


@app.command()
def assign():
    """Find the best task for your machine."""
    p = get_profile()

    if not p.ollama_running:
        print("Ollama is not running. Start it with: ollama serve")
        raise typer.Exit(1)

    if not p.ollama_models:
        print("No models found. Pull one: ollama pull llama3.2")
        raise typer.Exit(1)

    tasks_dir = Path("tasks")
    if not tasks_dir.exists():
        print("No tasks/ directory found. Run in the pingpoint repo root.")
        raise typer.Exit(1)

    task_files = list(tasks_dir.glob("*.yaml")) + list(tasks_dir.glob("*.yml"))
    if not task_files:
        print("No tasks found in tasks/. Create a YAML file or GitHub Issue.")
        raise typer.Exit(1)

    tasks: list[Task] = []
    for tf in task_files:
        data = yaml.safe_load(tf.read_text())
        required_yaml = ["title", "description", "prompt", "test_prompt"]
        for field in required_yaml:
            if field not in data or not data[field]:
                print(f"  [X] {tf.name}: missing required field '{field}'")
                raise typer.Exit(1)
        task = Task(
            id=tf.stem,
            title=data.get("title", tf.stem),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            test_prompt=data.get("test_prompt", ""),
            tags=data.get("tags", []),
            issue_url=data.get("issue_url"),
            issue_number=data.get("issue_number"),
        )
        tasks.append(task)
        db.save_task(task)
        print(f"  [OK] {tf.name} imported")

    solution_counts = {}
    for task in tasks:
        solutions = db.list_solutions(task.id)
        solution_counts[task.id] = len(solutions)

    best = find_best_task(p, tasks, solution_counts)

    if best is None:
        print("No suitable task found.")
        raise typer.Exit(1)

    version_count = solution_counts.get(best.id, 0)

    print(f"\nTask:           {best.title}")
    print(f"Description:    {best.description}")
    print(f"Tags:           {', '.join(best.tags)}")
    print(f"Versions so far: {version_count}")
    if best.issue_number:
        print(f"Issue:          #{best.issue_number}")
    if best.issue_url:
        print(f"URL:            {best.issue_url}")

    print(f"\nPrompt:\n{best.prompt[:500]}")
    print(f"\nRun 'pingpoint run' to generate a solution for this task.")


MAX_PROMPTS = 3


def _print_pass_baton(task: Task, latest: Optional[Solution]) -> None:
    print(f"\n=== PASS THE BATON! ===")
    print(f"Task: {task.title}")
    print(f"Issue: {task.issue_url or 'N/A'}")
    if latest:
        print(f"Round {latest.round or 1} complete")
    print(f"\nNext steps for the next collaborator:")
    print(f"  1. git pull")
    print(f"  2. pingpoint assign")
    print(f"  3. pingpoint validate {task.id}")
    print(f"  4. pingpoint run")
    if latest and latest.handoff_instructions:
        print(f"\nHandoff from previous collaborator:")
        for line in latest.handoff_instructions.split("\n"):
            print(f"  {line}")
    print(f"\nThe baton is yours!")


@app.command()
def run(
    challenge: Optional[str] = typer.Option(None, "--challenge", "-c", help="Your challenge prompt for the AI"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Max tokens"),
):
    """Generate a solution. First run uses task prompt. Next runs need --challenge. Max 3 prompts total."""
    p = get_profile()

    if not p.ollama_running:
        print("Ollama is not running. Start it with: ollama serve")
        raise typer.Exit(1)

    if not p.ollama_models:
        print("No models found. Pull one: ollama pull llama3.2")
        raise typer.Exit(1)

    selected_model = model or p.ollama_models[0]
    if selected_model not in p.ollama_models:
        print(f"Model '{selected_model}' not found. Available: {', '.join(p.ollama_models)}")
        raise typer.Exit(1)

    tasks = db.list_tasks()
    if not tasks:
        print("No tasks found. Run 'pingpoint assign' first.")
        raise typer.Exit(1)

    solution_counts = {t.id: len(db.list_solutions(t.id)) for t in tasks}
    best = find_best_task(p, tasks, solution_counts)
    if best is None:
        print("No suitable task found.")
        raise typer.Exit(1)

    latest = db.latest_solution(best.id)

    # --- Determine round and run_number ---
    if challenge:
        # Continuing a round
        if latest is None:
            print("No solution to challenge. Run without --challenge first.")
            raise typer.Exit(1)
        run_number = (latest.run_number or 1) + 1
        current_round = latest.round or 1
        if run_number > MAX_PROMPTS:
            _print_pass_baton(best, latest)
            raise typer.Exit(0)
        prompt_to_use = f"""The original task is: {best.prompt}

Current solution:
{latest.output}

The user challenges you: {challenge}

Generate an improved version that addresses this challenge."""
        print(f"Prompt {run_number}/{MAX_PROMPTS} — Your challenge: {challenge}")
    else:
        # Starting fresh (new round)
        if latest is None:
            current_round = 1
            run_number = 1
            prompt_to_use = best.prompt
            print(f"Prompt 1/{MAX_PROMPTS} — Task prompt")
        else:
            current_round = (latest.round or 1) + 1
            run_number = 1
            prompt_to_use = f"""The original task is: {best.prompt}

Current best solution:
{latest.output}

Improve upon this solution or create a better one."""
            print(f"Prompt 1/{MAX_PROMPTS} — New round (round {current_round})")

    print(f"Task: {best.title}")
    print(f"Model: {selected_model}")

    print("Generating solution...")
    result = call_ollama(selected_model, prompt_to_use, temperature, max_tokens)

    if result is None:
        print("Failed to generate solution. Check that Ollama is running.")
        raise typer.Exit(1)

    output, elapsed = result

    gpu_str = ", ".join(p.gpu) if p.gpu and p.gpu[0] != "Unknown" else "CPU only"
    hardware = f"{p.cpu} | {p.ram_gb}GB RAM"
    if p.has_gpu:
        hardware += f" | {gpu_str}"
        if p.vram_gb:
            hardware += f" | {p.vram_gb}GB VRAM"

    new_version = (latest.version + 1) if latest else 1

    solution = Solution(
        task_id=best.id,
        version=new_version,
        run_number=run_number,
        round=current_round,
        prompt_used=clean_ansi(prompt_to_use),
        output=output,
        previous_hash=latest.compute_hash() if latest else None,
        previous_output=latest.output if latest else None,
        metadata=SolutionMetadata(
            model=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
            hardware=hardware,
            execution_time_s=round(elapsed, 1),
            ollama_version=get_ov(),
        ),
    )

    db.save_solution(solution)

    print(f"\nSolution v{new_version} saved! ({elapsed:.1f}s)")

    print("Testing solution...")
    test_result = test_solution(
        best.prompt,
        solution,
        latest,
        model=selected_model,
        temperature=0.3,
    )

    db.save_test_result(test_result)

    status = "PASSED" if test_result.passed else "FAILED"
    print(f"Test: {status} (score: {test_result.score:.0f}/100)")
    print(f"Adds new value: {'Yes' if test_result.improvement_found else 'No'}")
    print(f"Summary: {test_result.summary}")

    print(f"\nOutput:\n{output[:600]}{'...' if len(output) > 600 else ''}")
    print(f"\nSolution saved to: solutions/{best.id}/v{new_version}.json")

    is_last_prompt = run_number == MAX_PROMPTS
    if is_last_prompt:
        print(f"\n=== PROMPT {run_number}/{MAX_PROMPTS} — LAST ONE! ===")
        print("Write your handoff instructions for the next collaborator.")
        print("(End with an empty line, or Ctrl+C to skip)")
        print("---")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass
        handoff = "\n".join(lines).strip() if lines else ""
        if handoff:
            solution.handoff_instructions = handoff
            db.save_solution(solution)
            print(f"\nHandoff instructions saved.")
        else:
            print(f"\nNo handoff instructions written.")

        all_solutions = db.list_solutions(best.id)
        _print_pass_baton(best, all_solutions[-1] if all_solutions else None)
    else:
        print("Commit it to the repo to make it permanent: git add solutions/ && git commit")


@app.command()
def show(
    task_id: str = typer.Argument(..., help="Task ID or issue number"),
):
    """Show the full process for a solution."""
    solutions = db.list_solutions(task_id)
    if not solutions:
        print(f"No solutions found for task: {task_id}")
        raise typer.Exit(1)

    task = db.load_task(task_id)

    print(f"Task: {task.title if task else task_id}")
    print(f"Solutions: {len(solutions)}")

    for sol in solutions:
        print(f"\n--- v{sol.version} (round {sol.round or 1}, prompt {sol.run_number}/{MAX_PROMPTS}) ---")
        print(f"Model: {sol.metadata.model}")
        print(f"Temperature: {sol.metadata.temperature}")
        print(f"Hardware: {sol.metadata.hardware}")
        print(f"Time: {sol.metadata.execution_time_s}s")

        print(f"\nPrompt used:\n{sol.prompt_used[:1000]}")

        print(f"\nOutput:\n{sol.output[:1000]}")
        if sol.handoff_instructions:
            print(f"\nHandoff instructions:\n{sol.handoff_instructions}")

    print(f"\nFull data at: solutions/{task_id}/")


@app.command()
def verify(
    task_id: str = typer.Argument(..., help="Task ID to verify"),
):
    """Verify the integrity of the solution hash chain."""
    errors = db.verify_chain(task_id)
    if not errors:
        print(f"Chain for {task_id}: VALID")
    else:
        print(f"Chain for {task_id}: TAMPERED")
        for e in errors:
            print(f"  ! {e}")


@app.command()
def validate(
    task_id: str = typer.Argument(..., help="Task ID to validate"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Validate all solutions for a task against the schema."""
    result = validate_all(task_id)
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print_validation(result)
    raise typer.Exit(0 if result["valid"] else 1)


@app.command()
def report(
    task_id: str = typer.Argument(..., help="Task ID to report"),
):
    """Generate a comprehensive JSON report of the full challenge process."""
    task = db.load_task(task_id)
    solutions = db.list_solutions(task_id)
    test_results = db.list_test_results(task_id)

    validation = validate_all(task_id)

    tests_by_version = {r.version: r for r in test_results}

    total_time = sum(s.metadata.execution_time_s for s in solutions)
    rounds = set(s.round for s in solutions)

    report_data = {
        "task": {
            "id": task_id,
            "title": task.title if task else task_id,
            "description": task.description if task else "",
            "issue_url": task.issue_url if task else None,
            "issue_number": task.issue_number if task else None,
        },
        "summary": {
            "total_versions": len(solutions),
            "total_rounds": len(rounds),
            "total_execution_time_s": round(total_time, 1),
            "chain_valid": not db.verify_chain(task_id),
            "validation_pass": validation["valid"],
        },
        "chain_errors": db.verify_chain(task_id),
        "validation": validation,
        "rounds": [],
    }

    for rnd in sorted(rounds):
        round_sols = [s for s in solutions if s.round == rnd]
        round_entry = {
            "round": rnd,
            "prompts": [],
        }
        for sol in round_sols:
            test = tests_by_version.get(sol.version)
            prompt_entry = {
                "version": sol.version,
                "run_number": sol.run_number,
                "model": sol.metadata.model,
                "temperature": sol.metadata.temperature,
                "execution_time_s": sol.metadata.execution_time_s,
                "hardware": sol.metadata.hardware,
                "created_at": sol.created_at,
                "handoff_instructions": sol.handoff_instructions,
                "test": {
                    "passed": test.passed if test else None,
                    "score": test.score if test else None,
                    "improvement_found": test.improvement_found if test else None,
                    "summary": test.summary if test else None,
                } if test else None,
            }
            round_entry["prompts"].append(prompt_entry)
        report_data["rounds"].append(round_entry)

    output_path = Path("solutions") / task_id / "report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_data, indent=2))

    print(f"Report saved to: {output_path}")
    print(json.dumps(report_data, indent=2))
