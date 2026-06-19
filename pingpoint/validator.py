import json
from pathlib import Path
from typing import Optional

from pingpoint.db import Database

from pingpoint.models import TASK_TYPES

TASK_REQUIRED = ["id", "title", "description", "prompt", "test_prompt"]
TASK_OPTIONAL = ["tags", "task_type", "issue_url", "issue_number", "created_at"]

SOLUTION_REQUIRED = [
    "task_id", "version", "run_number", "round",
    "prompt_used", "output", "previous_hash",
    "metadata", "created_at",
]
SOLUTION_METADATA_REQUIRED = [
    "model", "temperature", "max_tokens",
    "hardware", "execution_time_s", "ollama_version",
]


def validate_task(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"Missing task file: {path}"]
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"Invalid JSON in {path}: {e}"]
    for field in TASK_REQUIRED:
        if field not in data:
            errors.append(f"Missing required field '{field}' in {path.name}")
    for field in TASK_OPTIONAL:
        if field in data and data[field] is None:
            errors.append(f"'{field}' is null in {path.name}")
    return errors


def validate_solution(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"Missing solution file: {path}"]
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"Invalid JSON in {path.name}: {e}"]

    for field in SOLUTION_REQUIRED:
        if field not in data:
            errors.append(f"{path.name}: missing required field '{field}'")

    if "version" in data:
        if not isinstance(data["version"], int) or data["version"] < 1:
            errors.append(f"{path.name}: 'version' must be a positive integer")

    if "run_number" in data:
        rn = data["run_number"]
        if not isinstance(rn, int) or rn < 1 or rn > 3:
            errors.append(f"{path.name}: 'run_number' must be 1-3, got {rn}")

    if "round" in data:
        if not isinstance(data["round"], int) or data["round"] < 1:
            errors.append(f"{path.name}: 'round' must be a positive integer")

    version_val = data.get("version", 0)
    if "previous_hash" in data and "hash" in data and data["hash"] == data.get("previous_hash"):
        if isinstance(version_val, int) and version_val > 1:
            errors.append(f"{path.name}: 'hash' equals 'previous_hash' (self-reference)")

    if "metadata" in data:
        meta = data["metadata"]
        for field in SOLUTION_METADATA_REQUIRED:
            if field not in meta:
                errors.append(f"{path.name}/metadata: missing '{field}'")

    return errors


def validate_all(task_id: str) -> dict:
    result = {
        "task_id": task_id,
        "valid": True,
        "errors": [],
        "warnings": [],
        "solutions": [],
    }

    db = Database(Path.home() / ".pingpoint")
    task = db.load_task(task_id)
    if task is None:
        result["valid"] = False
        result["errors"].append(f"Task '{task_id}' not found in local database")
    else:
        task_dict = task.to_dict()
        for field in TASK_REQUIRED:
            if field not in task_dict:
                result["valid"] = False
                result["errors"].append(f"Missing required field '{field}' in task '{task_id}'")
        for field in TASK_OPTIONAL:
            if field in task_dict and task_dict[field] is None:
                result["valid"] = False
                result["errors"].append(f"'{field}' is null in task '{task_id}'")
        task_type = task_dict.get("task_type", "project")
        if task_type not in TASK_TYPES:
            result["valid"] = False
            result["errors"].append(
                f"Invalid task_type '{task_type}'. Must be one of: {', '.join(TASK_TYPES)}"
            )

    sol_dir = Path("solutions") / task_id
    if not sol_dir.exists():
        result["valid"] = False
        result["errors"].append(f"No solutions directory: solutions/{task_id}/")
        return result

    sol_files = sorted(sol_dir.glob("v*.json"))
    if not sol_files:
        result["valid"] = False
        result["errors"].append(f"No solution files in solutions/{task_id}/")
        return result

    prev_hash = None
    prev_version = None
    prev_run_number = 0
    prev_round = None

    for path in sol_files:
        entry = {"file": path.name, "errors": [], "warnings": []}
        entry_errors = validate_solution(path)
        if entry_errors:
            result["valid"] = False
            entry["errors"].extend(entry_errors)

        try:
            data = json.loads(path.read_text())
            ver = data.get("version", 0)
            stored_hash = data.get("hash")
            prev_h = data.get("previous_hash")
            rn = data.get("run_number", 1)
            rnd = data.get("round", 1)

            # Version sequence
            if prev_version is not None and ver != prev_version + 1:
                entry["errors"].append(
                    f"version gap: v{prev_version} -> v{ver} (expected v{prev_version + 1})"
                )
                result["valid"] = False

            # Hash chain
            if prev_version is not None and stored_hash and prev_h:
                if prev_h != prev_hash:
                    entry["errors"].append(
                        f"previous_hash mismatch: got {prev_h[:16]}..., "
                        f"expected {prev_hash[:16]}... (from v{prev_version})"
                    )
                    result["valid"] = False

            # Round consistency
            if prev_round is not None:
                if rnd < prev_round:
                    entry["errors"].append(
                        f"round decreased: {prev_round} -> {rnd}"
                    )
                    result["valid"] = False
                elif rnd == prev_round:
                    if rn != prev_run_number + 1:
                        entry["warnings"].append(
                            f"run_number in round {rnd}: {prev_run_number} -> {rn} "
                            f"(expected {prev_run_number + 1})"
                        )
                else:
                    if rn != 1:
                        entry["warnings"].append(
                            f"new round {rnd} should start at run_number=1, got {rn}"
                        )

            prev_hash = stored_hash
            prev_version = ver
            prev_run_number = rn
            prev_round = rnd

        except (json.JSONDecodeError, OSError):
            pass

        result["solutions"].append(entry)

    return result


def print_validation(result: dict) -> None:
    print(f"\n=== Validation: {result['task_id']} ===")
    print(f"Status: {'PASS' if result['valid'] else 'FAIL'}")
    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  [X] {e}")
    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"  [!] {w}")
    for entry in result["solutions"]:
        if entry["errors"] or entry["warnings"]:
            print(f"\n  {entry['file']}:")
            for e in entry["errors"]:
                print(f"    [X] {e}")
            for w in entry["warnings"]:
                print(f"    [!] {w}")
    print()
