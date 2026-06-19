import json
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import typer
import yaml

from pingpoint import __version__
from pingpoint.context import build_context_prompt, get_repo_tree
from pingpoint.db import Database
from pingpoint.models import Task, Solution, SolutionMetadata, TASK_TYPES
from pingpoint.profiler import profile as get_profile
from pingpoint.matcher import find_best_task
from pingpoint.runner import call_ollama, clean_ansi, get_ollama_version as get_ov
from pingpoint.tester import test_solution
from pingpoint.validator import validate_all, print_validation

app = typer.Typer()

DATA_DIR = Path.home() / ".pingpoint"
db = Database(DATA_DIR)


def _get_repo_from_git() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5
        )
        url = result.stdout.strip()
        match = re.search(r'(?:github\.com[:\/])([\w.-]+/[\w.-]+?)(?:\.git)?$', url)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_author() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5
        )
        name = result.stdout.strip()
        if name:
            return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "anonymous"


@app.command()
def version():
    """Show pingpoint version."""
    print(f"pingpoint v{__version__}")


@app.command()
def fetch_issue(
    issue_number: int = typer.Argument(..., help="GitHub issue number"),
    repo: Optional[str] = typer.Option(
        None, "--repo", "-r", help="Repository as owner/name (e.g. 'user/repo')"
    ),
):
    """Fetch a GitHub issue and create a task YAML from it. Uses the public API (no token needed)."""
    if repo is None:
        detected = _get_repo_from_git()
        if detected is None:
            print("Could not detect repo from git remote. Use --repo owner/name")
            raise typer.Exit(1)
        repo = detected
        print(f"Detected repo: {repo}")

    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    print(f"Fetching {url} ...")

    req = urllib.request.Request(url, headers={"User-Agent": "pingpoint/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"GitHub API error: {e.code} {e.reason}")
        if e.code == 404:
            print(f"Issue #{issue_number} not found in {repo}")
        elif e.code == 403:
            print("Rate limit exceeded. Try again later or use a token via GITHUB_TOKEN env var.")
        raise typer.Exit(1)
    except (urllib.error.URLError, OSError) as e:
        print(f"Network error: {e}")
        raise typer.Exit(1)

    if data.get("pull_request"):
        print("That's a pull request, not an issue. Use the issue number instead.")
        raise typer.Exit(1)

    title = data.get("title", "Untitled")
    body = data.get("body", "") or ""
    labels = [lb["name"] for lb in data.get("labels", [])]
    html_url = data.get("html_url", f"https://github.com/{repo}/issues/{issue_number}")

    task_type = "project"
    for lb in labels:
        if lb.startswith("type: "):
            candidate = lb.replace("type: ", "").strip()
            if candidate in TASK_TYPES:
                task_type = candidate

    description = body.strip().split("\n\n")[0][:500] if body else title

    if task_type == "question":
        prompt = f"Answer the following question:\n\n{body[:2000]}"
        test_prompt = f"Does the answer correctly address: {title}?"
    else:
        prompt = f"Implement the following:\n\n{body[:2000]}" if body else f"Implement: {title}"
        test_prompt = f"Does the solution correctly implement: {title}?"

    task_data = {
        "title": title,
        "description": description,
        "prompt": prompt,
        "test_prompt": test_prompt,
        "tags": labels if labels else ["general"],
        "task_type": task_type,
        "issue_number": issue_number,
        "issue_url": html_url,
    }

    tasks_dir = Path("tasks")
    tasks_dir.mkdir(exist_ok=True)
    yaml_path = tasks_dir / f"issue-{issue_number}.yaml"
    yaml_path.write_text(yaml.dump(task_data, default_flow_style=False, allow_unicode=True).strip() + "\n")

    print(f"\nTask created: {yaml_path}")
    print(f"Title: {title}")
    print(f"Type: {task_type}")
    print(f"Tags: {', '.join(labels) if labels else 'general'}")
    print(f"\nEdit {yaml_path} to refine the prompt and test_prompt before running 'pingpoint assign'.")


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
            task_type=data.get("task_type", "project"),
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
    print(f"Type:           {best.task_type}")
    print(f"Description:    {best.description}")
    print(f"Tags:           {', '.join(best.tags)}")
    print(f"Versions so far: {version_count}")
    if best.issue_number:
        print(f"Issue:          #{best.issue_number}")
    if best.issue_url:
        print(f"URL:            {best.issue_url}")

    print(f"\nPrompt:\n{best.prompt[:500]}")

    latest = db.latest_solution(best.id)
    if latest and latest.handoff_instructions:
        print(f"\nHandoff from previous collaborator:")
        for line in latest.handoff_instructions.split("\n"):
            print(f"  {line}")
        print()

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
    author = get_author()

    # --- Determine round and run_number ---
    if challenge:
        # Continuing a round
        if latest is None:
            print("No solution to challenge. Run without --challenge first.")
            raise typer.Exit(1)
        run_number = (latest.run_number or 1) + 1
        current_round = latest.round or 1

        if latest.author and latest.author != author:
            print(f"This round belongs to {latest.author}. Start a new round: pingpoint run")
            raise typer.Exit(1)

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

    if best.task_type in ("bug", "feature"):
        prompt_to_use = build_context_prompt(best.task_type, prompt_to_use)
        print(f"[Injecting repo context for {best.task_type}]")

    if best.task_type == "question":
        print("Answering question (no code generation, no versioning)...")
        result = call_ollama(selected_model, prompt_to_use, temperature, max_tokens)
        if result is None:
            print("Failed to generate answer. Check that Ollama is running.")
            raise typer.Exit(1)
        output, elapsed = result
        print(f"\nAnswer:\n{output}")
        print(f"\n({elapsed:.1f}s)")
        return

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
        author=author,
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
        print(f"Author: {sol.author or 'unknown'}")
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
    sign: bool = typer.Option(False, "--sign", "-s", help="Record your verification"),
    show: bool = typer.Option(False, "--show", help="Show all verifiers"),
):
    """Verify the integrity of the solution hash chain.

    Use --sign to record your verification after a successful check.
    Use --show to see everyone who has verified this task.
    """
    if show:
        verifications = db.list_verifications(task_id)
        if not verifications:
            print(f"No verifications recorded for {task_id}")
        else:
            print(f"Verifications for {task_id}:")
            for v in verifications:
                print(f"  [{v['result'].upper()}] {v['verifier']} — {v['timestamp'][:19]}")
        return

    errors = db.verify_chain(task_id)
    if not errors:
        existing = db.list_verifications(task_id)
        verifier_count = len(existing)
        print(f"Chain for {task_id}: VALID  ({verifier_count} verifier{'s' if verifier_count != 1 else ''})")

        if sign:
            author = get_author()
            db.add_verification(task_id, author)
            print(f"Verification recorded: {author}")
            new_count = verifier_count + 1
            print(f"Total verifiers: {new_count}")
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

    verifications = db.list_verifications(task_id)
    trust_score = sum(1 for v in verifications if v.get("result") == "valid")

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
            "verifiers": trust_score,
            "verified_by": [v["verifier"] for v in verifications if v.get("result") == "valid"],
        },
        "chain_errors": db.verify_chain(task_id),
        "validation": validation,
        "verifications": verifications,
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
                "author": sol.author,
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
