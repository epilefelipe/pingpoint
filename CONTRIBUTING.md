# Contributing to pingpoint

## How this project works

pingpoint is a **decentralized compute cooperative**. People create tasks (GitHub Issues), and collaborators solve them using local AI (Ollama). Solutions are shared via git.

Everyone has a role:

- **Task creator** — opens an issue describing what they need. No hardware required.
- **Collaborator** — has hardware, runs Ollama, generates solutions, passes the baton.
- **Reviewer** — reviews PRs, checks validation, keeps quality high.

---

## For task creators

### 1. Open a GitHub Issue

Use the [task template](https://github.com/epilefelipe/pingpoint/issues/new/choose) to create a structured issue. The template asks for:

- **Description** — what needs to be done
- **Prompt** — instructions the AI will receive
- **Test prompt** — how to evaluate the result
- **Task type** — one of: `proyecto`, `bug`, `feature`, `pregunta`
- **Target files** — for bugs/features, which files to modify

### 2. Add labels

| Label | Meaning |
|---|---|
| `good first issue` | Beginner-friendly task |
| `tipo: proyecto` | Create something from scratch |
| `tipo: bug` | Fix a problem in existing code |
| `tipo: feature` | Add new functionality |
| `tipo: pregunta` | Answer a question (no code) |

Labels control how pingpoint handles the task. `tipo: bug` and `tipo: feature` inject the repository code into the AI prompt; `tipo: proyecto` does not.

### 3. Wait for collaborators

Once the issue is open, collaborators with hardware will pick it up, generate solutions, and submit PRs.

---

## For collaborators (you have hardware)

### Setup

```bash
git clone https://github.com/epilefelipe/pingpoint
cd pingpoint
pip install -e .
ollama pull llama3.2   # or any model you prefer
```

### Step-by-step workflow

#### 1. Pull the latest tasks

```bash
git pull
pingpoint assign
```

This imports all tasks from `tasks/` into the local database and picks the best one for your machine.

#### 2. Generate a solution (round 1, prompt 1/3)

```bash
pingpoint run
```

- Uses your local Ollama model
- Generates version 1 of the solution
- Runs an automatic test
- For `tipo: bug` / `tipo: feature`, the entire repo structure is injected for context

#### 3. Refine (prompts 2/3 and 3/3)

```bash
pingpoint run --challenge "Add error handling"
pingpoint run --challenge "Improve performance"
```

Each challenge improves the previous solution. You get up to 3 prompts per round.

#### 4. Pass the baton

After prompt 3/3, write handoff instructions for the next collaborator:

```
=== PROMPT 3/3 — LAST ONE! ===
Write your handoff instructions for the next collaborator.
(End with an empty line, or Ctrl+C to skip)
---
What I tried, what worked, what needs improvement...
```

#### 5. Validate and report

```bash
pingpoint validate <task-id>
pingpoint verify <task-id>
pingpoint report <task-id>
```

- `validate` — checks schema, structure, and hash chain
- `verify` — confirms no solution was tampered with
- `report` — generates a summary JSON in `solutions/<task-id>/report.json`

All three must pass before opening a PR.

#### 6. Commit and PR

```bash
git add solutions/
git commit -m "Solve <task-id>: <summary>"
git push
```

Then open a pull request. The PR will be auto-validated by GitHub Actions.

---

## For reviewers

1. Check that `pingpoint validate <task-id>` passes
2. Check that `pingpoint verify <task-id>` shows VALID
3. Review the solution output and handoff notes
4. Merge if everything is clean

---

## Task types explained

| Type | When to use | What pingpoint does |
|---|---|---|
| `proyecto` | New project from scratch | Sends the task prompt as-is. No repo context. |
| `bug` | Something is broken | Injects repo tree + file contents into the prompt. AI generates specific file fixes. |
| `feature` | Add to existing code | Same as bug — injects repo context for informed changes. |
| `pregunta` | Answer a question | Runs the model, prints the answer. No versioning, no hash chain. |

---

## Code conventions

- Python 3.10+
- Type hints on all functions
- Dataclasses for data models
- Tests in `tests/` using pytest
- CLI commands use Typer
- Keep functions under 50 lines where possible

### Running tests

```bash
pytest
```

All tests must pass before a PR.

---

## Good first issues

Check [issues labeled `good first issue`](https://github.com/epilefelipe/pingpoint/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

These are self-contained tasks with a clear prompt and test criteria. To work on one:

```bash
pingpoint fetch-issue <number>
pingpoint assign
pingpoint run
```

---

## Need help?

Open a [Discussion](https://github.com/epilefelipe/pingpoint/discussions) or ask in the issue itself.
