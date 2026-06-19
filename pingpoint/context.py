import os
import subprocess
from pathlib import Path

MAX_FILE_SIZE = 50 * 1024
MAX_FILES = 15
IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules",
               ".venv", "venv", "env", ".egg-info", "dist", "build",
               ".github", "solutions", "tasks"}


def get_repo_tree(root: str | None = None) -> str:
    root = root or os.getcwd()
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS
                       and not d.startswith(".")]
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            depth = 0
        else:
            depth = rel.count(os.sep) + 1
        prefix = "  " * depth
        if depth > 0:
            lines.append(f"{prefix}{os.path.basename(dirpath)}/")
        for f in filenames:
            lines.append(f"{prefix}  {f}")
    return "\n".join(lines)


def get_repo_file_contents(root: str | None = None) -> list[tuple[str, str]]:
    root = root or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, timeout=5,
        )
        tracked = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tracked = []

    if not tracked:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for f in filenames:
                tracked.append(os.path.relpath(os.path.join(dirpath, f), root))

    contents: list[tuple[str, str]] = []
    for fp in tracked[:MAX_FILES]:
        full = os.path.join(root, fp)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                   ".woff", ".woff2", ".ttf", ".eot", ".lock", ".sum"):
            continue
        if os.path.getsize(full) > MAX_FILE_SIZE:
            continue
        try:
            text = Path(full).read_text(encoding="utf-8", errors="replace")
            contents.append((fp, text))
        except Exception:
            pass

    return contents


def build_context_prompt(task_type: str, task_prompt: str,
                         target_files: list[str] | None = None) -> str:
    if task_type in ("bug", "feature"):
        tree = get_repo_tree()
        parts = [f"Current repository structure:\n```\n{tree}\n```\n"]

        files = get_repo_file_contents()
        if target_files:
            files = [(p, c) for p, c in files if p in target_files]

        if files:
            parts.append("Relevant file contents:")
            for path, content in files:
                ext = os.path.splitext(path)[1].lstrip(".")
                parts.append(f"\n--- {path} ---\n```{ext}\n{content}\n```")

        context = "\n".join(parts)

        if task_type == "bug":
            instruction = (
                "Identify the bug and provide the exact file changes needed to fix it. "
                "Output each modified file with its full path and content."
            )
        else:
            instruction = (
                "Extend the existing code to implement the requested feature. "
                "Output each new or modified file with its full path and content."
            )

        return f"{context}\n\nTask: {task_prompt}\n\n{instruction}"

    return task_prompt
