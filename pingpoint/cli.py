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

    solutions = db.list_solutions(best.id)
    prompt_count = len(solutions) + 1

    if prompt_count > MAX_PROMPTS:
        print(f"Max {MAX_PROMPTS} prompts reached for this task. Pass the baton!")
        print("The next collaborator should take over.")
        raise typer.Exit(0)

    latest = db.latest_solution(best.id)

    if latest and not challenge:
        print(f"This task already has {len(solutions)} prompt(s). Provide a challenge: pingpoint run --challenge \"your prompt\"")
        raise typer.Exit(1)

    if prompt_count == 1:
        prompt_to_use = best.prompt
        print(f"Prompt 1/{MAX_PROMPTS} — Task prompt")
    else:
        prompt_to_use = f"""The original task is: {best.prompt}

Current solution:
{latest.output}

The user challenges you: {challenge}

Generate an improved version that addresses this challenge."""

        print(f"Prompt {prompt_count}/{MAX_PROMPTS} — Your challenge: {challenge}")

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
        prompt_used=clean_ansi(prompt_to_use),
        output=output,
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
    print(f"\nSolution saved to: ~/.pingpoint/solutions/{best.id}/v{new_version}.json")


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
        print(f"\n--- v{sol.version} ---")
        print(f"Model: {sol.metadata.model}")
        print(f"Temperature: {sol.metadata.temperature}")
        print(f"Hardware: {sol.metadata.hardware}")
        print(f"Time: {sol.metadata.execution_time_s}s")

        print(f"\nPrompt used:\n{sol.prompt_used[:1000]}")

        print(f"\nOutput:\n{sol.output[:1000]}")

    print(f"\nFull data at: ~/.pingpoint/solutions/{task_id}/")
