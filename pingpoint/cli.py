import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from pingpoint import __version__
from pingpoint.db import Database
from pingpoint.models import Task, Solution, SolutionMetadata, Profile
from pingpoint.profiler import profile as get_profile, get_ollama_version
from pingpoint.matcher import find_best_task
from pingpoint.runner import call_ollama, get_ollama_version as get_ov
from pingpoint.tester import test_solution

app = typer.Typer()
console = Console()

DATA_DIR = Path.home() / ".pingpoint"
db = Database(DATA_DIR)


@app.callback()
def callback():
    pass


@app.command()
def version():
    """Show pingpoint version."""
    console.print(f"pingpoint v{__version__}")


@app.command()
def profile():
    """Detect your hardware and available AI models."""
    with console.status("[bold green]Analyzing your machine..."):
        p = get_profile()

    console.print(Panel(f"[bold]Your Profile[/bold]\n", expand=False))

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Platform", p.platform)
    table.add_row("CPU", p.cpu)
    table.add_row("Cores", str(p.cpu_cores))
    table.add_row("RAM", f"{p.ram_gb} GB")
    table.add_row("GPU", ", ".join(p.gpu))
    if p.vram_gb:
        table.add_row("VRAM", f"{p.vram_gb} GB")
    table.add_row("Capability", p.capability.upper())
    table.add_row("Score", str(p.score))
    table.add_row("Ollama running", "Yes" if p.ollama_running else "No")
    table.add_row("Ollama models", ", ".join(p.ollama_models) or "None")

    console.print(table)

    if not p.ollama_running:
        console.print("\n[yellow]Ollama is not running. Start it with: ollama serve[/yellow]")
    if not p.ollama_models:
        console.print("\n[yellow]No models found. Pull one: ollama pull llama3.2[/yellow]")


@app.command()
def assign():
    """Find the best task for your machine."""
    p = get_profile()

    if not p.ollama_running:
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    if not p.ollama_models:
        console.print("[red]No models found. Pull one: ollama pull llama3.2[/red]")
        raise typer.Exit(1)

    tasks_dir = Path("tasks")
    if not tasks_dir.exists():
        console.print("[yellow]No tasks/ directory found. Run in the pingpoint repo root.[/yellow]")
        raise typer.Exit(1)

    task_files = list(tasks_dir.glob("*.yaml")) + list(tasks_dir.glob("*.yml"))
    if not task_files:
        console.print("[yellow]No tasks found in tasks/. Create one or run: pingpoint sync[/yellow]")
        raise typer.Exit(1)

    import yaml
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
        console.print("[red]No suitable task found.[/red]")
        raise typer.Exit(1)

    version_count = solution_counts.get(best.id, 0)

    console.print(Panel(f"[bold]Assigned Task[/bold]\n", expand=False))

    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Task", best.title)
    table.add_row("Description", best.description)
    table.add_row("Tags", ", ".join(best.tags))
    table.add_row("Versions so far", str(version_count))
    if best.issue_number:
        table.add_row("Issue", f"#{best.issue_number}")
    if best.issue_url:
        table.add_row("URL", best.issue_url)

    console.print(table)

    console.print("\n[bold]Prompt:[/bold]")
    console.print(Panel(best.prompt[:500], expand=False))

    console.print(f"\n[green]Run [bold]pingpoint run[/bold] to generate a solution for this task.[/green]")


@app.command()
def run(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Max tokens"),
):
    """Run the assigned task with your local AI and save the solution."""
    p = get_profile()

    if not p.ollama_running:
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    if not p.ollama_models:
        console.print("[red]No models found. Pull one: ollama pull llama3.2[/red]")
        raise typer.Exit(1)

    selected_model = model or p.ollama_models[0]
    if selected_model not in p.ollama_models:
        console.print(f"[red]Model '{selected_model}' not found. Available: {', '.join(p.ollama_models)}[/red]")
        raise typer.Exit(1)

    tasks = db.list_tasks()
    if not tasks:
        console.print("[red]No tasks found. Run [bold]pingpoint assign[/bold] first.[/red]")
        raise typer.Exit(1)

    solution_counts = {t.id: len(db.list_solutions(t.id)) for t in tasks}
    best = find_best_task(p, tasks, solution_counts)
    if best is None:
        console.print("[red]No suitable task found.[/red]")
        raise typer.Exit(1)

    latest = db.latest_solution(best.id)

    # Build improvement prompt if a previous solution exists
    improvement_prompt = None
    if latest:
        improvement_prompt = f"""The task is: {best.prompt}

Here is the current best solution:
{latest.output}

Your job is to improve this solution. Add something new, refine the ideas, make it better. Do not simply rephrase — add real value."""

    console.print(f"[bold]Running task:[/bold] {best.title}")
    console.print(f"[bold]Model:[/bold] {selected_model}")
    console.print(f"[bold]Temperature:[/bold] {temperature}")

    if improvement_prompt:
        console.print("[yellow]Previous solution exists — improving it...[/yellow]")

    with console.status("[bold green]Generating solution..."):
        prompt_to_use = improvement_prompt if improvement_prompt else best.prompt
        result = call_ollama(selected_model, prompt_to_use, temperature, max_tokens)

    if result is None:
        console.print("[red]Failed to generate solution. Check that Ollama is running.[/red]")
        raise typer.Exit(1)

    output, elapsed = result

    # Build hardware summary
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
        prompt_used=prompt_to_use,
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

    console.print(f"\n[green]Solution v{new_version} saved![/green]")
    console.print(f"Took {elapsed:.1f}s")

    # Run the test
    console.print("\n[bold]Testing solution...[/bold]")
    with console.status("[bold yellow]Evaluating..."):
        test_result = test_solution(
            best.prompt,
            solution,
            latest,
            model=selected_model,
            temperature=0.3,
        )

    db.save_test_result(test_result)

    console.print(f"\n[bold]Test Result:[/bold]")
    if test_result.passed:
        console.print(f"  [green]PASSED[/green] (score: {test_result.score:.0f}/100)")
    else:
        console.print(f"  [red]FAILED[/red] (score: {test_result.score:.0f}/100)")

    if test_result.improvement_found:
        console.print(f"  [green]Adds new value: Yes[/green]")
    else:
        console.print(f"  [red]Adds new value: No[/red]")

    console.print(f"  Summary: {test_result.summary}")

    # Show output preview
    console.print("\n[bold]Output preview:[/bold]")
    preview = output[:600] + "..." if len(output) > 600 else output
    console.print(Panel(preview, expand=False))

    console.print(f"\n[bold]Solution saved to:[/bold] ~/.pingpoint/solutions/{best.id}/v{new_version}.json")
    console.print("\n[green]Run [bold]pingpoint submit[/bold] to create a PR with your solution.[/green]")


@app.command()
def show(
    task_id: str = typer.Argument(..., help="Task ID or issue number"),
):
    """Show the full process for a solution."""
    solutions = db.list_solutions(task_id)
    if not solutions:
        console.print(f"[red]No solutions found for task: {task_id}[/red]")
        raise typer.Exit(1)

    task = db.load_task(task_id)

    console.print(f"[bold]Task:[/bold] {task.title if task else task_id}")
    console.print(f"[bold]Solutions:[/bold] {len(solutions)}")

    for sol in solutions:
        console.print(f"\n[bold cyan]--- v{sol.version} ---[/bold cyan]")
        console.print(f"[bold]Model:[/bold] {sol.metadata.model}")
        console.print(f"[bold]Temperature:[/bold] {sol.metadata.temperature}")
        console.print(f"[bold]Hardware:[/bold] {sol.metadata.hardware}")
        console.print(f"[bold]Time:[/bold] {sol.metadata.execution_time_s}s")

        console.print(f"\n[bold]Prompt used:[/bold]")
        console.print(Panel(sol.prompt_used[:1000], expand=False))

        console.print(f"\n[bold]Output:[/bold]")
        console.print(Panel(sol.output[:1000], expand=False))

    console.print("\n[yellow]Full data stored at:[/yellow]")
    console.print(f"  ~/.pingpoint/solutions/{task_id}/")


@app.command()
def web():
    """Launch the local web UI."""
    from pingpoint.web import start_web
    start_web()


@app.command()
def submit():
    """Prepare your solution for submission as a PR."""
    console.print("[bold]Submit your solutions[/bold]")
    console.print("\nTo share your solutions with the community:")
    console.print("  1. Fork the repository on GitHub")
    console.print("  2. Copy your solutions to the repo:")
    console.print(f"     cp -r ~/.pingpoint/solutions/* solutions/")
    console.print("  3. Commit and push:")
    console.print("     git add solutions/")
    console.print('     git commit -m "Add solution for task X"')
    console.print("     git push")
    console.print("  4. Create a Pull Request")
    console.print("\nOr use the gh CLI:")
    console.print("     gh pr create --title \"Add solution\" --body \"Added my AI-generated solution\"")


@app.command()
def sync():
    """Sync tasks from GitHub Issues."""
    console.print("[yellow]GitHub sync requires the 'gh' CLI.[/yellow]")
    console.print("\nInstall it from: https://cli.github.com/")
    console.print("\nThen authenticate:")
    console.print("  gh auth login")
    console.print("\nAnd run this command again to pull issues as tasks.")

    try:
        import subprocess
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            console.print("\n[red]gh CLI not found. Install it first.[/red]")
            return
    except FileNotFoundError:
        console.print("\n[red]gh CLI not found. Install it first.[/red]")
        return

    console.print("\n[green]gh CLI detected! Fetching issues...[/green]")

    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--json", "number,title,body,labels",
             "--limit", "50"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            console.print(f"[red]Failed to fetch issues: {result.stderr}[/red]")
            return

        import json as json_mod
        issues = json_mod.loads(result.stdout)

        import yaml
        tasks_dir = Path("tasks")
        tasks_dir.mkdir(exist_ok=True)

        count = 0
        for issue in issues:
            task_id = f"issue-{issue['number']}"
            task_file = tasks_dir / f"{task_id}.yaml"
            if task_file.exists():
                continue

            labels = [label["name"] for label in issue.get("labels", [])]
            if not labels:
                labels = ["general"]

            task_data = {
                "title": issue["title"],
                "description": (issue.get("body") or "")[:500],
                "prompt": issue.get("body") or issue["title"],
                "test_prompt": f"Test: does this solve '{issue['title']}'?",
                "tags": labels,
                "issue_number": issue["number"],
                "issue_url": f"https://github.com/unknown/issues/{issue['number']}",
            }

            task_file.write_text(yaml.dump(task_data, allow_unicode=True))
            count += 1

        console.print(f"\n[green]Synced {count} new tasks from Issues![/green]")

    except Exception as e:
        console.print(f"[red]Error syncing issues: {e}[/red]")


@app.command()
def init():
    """Initialize pingpoint in the current directory."""
    tasks_dir = Path("tasks")
    tasks_dir.mkdir(exist_ok=True)
    solutions_dir = Path("solutions")
    solutions_dir.mkdir(exist_ok=True)

    example_task = {
        "title": "Write a short poem about AI",
        "description": "Generate a creative poem about artificial intelligence and its impact on humanity.",
        "prompt": "Write a short poem (8-12 lines) about artificial intelligence. Make it creative, thoughtful, and slightly optimistic.",
        "test_prompt": "Does the poem mention AI, is it creative, and is it at least 8 lines?",
        "tags": ["creative", "writing"],
    }

    import yaml
    example_path = tasks_dir / "example-poem.yaml"
    if not example_path.exists():
        example_path.write_text(yaml.dump(example_task, allow_unicode=True))
        console.print("[green]Created example task: tasks/example-poem.yaml[/green]")

    console.print("[bold green]pingpoint initialized![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. [bold]pingpoint profile[/bold] — check your setup")
    console.print("  2. [bold]pingpoint assign[/bold] — get your first task")
    console.print("  3. [bold]pingpoint run[/bold] — generate a solution")
