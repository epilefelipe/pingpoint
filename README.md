# pingpoint

**The decentralized compute cooperative for AI.**

Not everyone has a powerful GPU. Not everyone can afford AI APIs. But everyone deserves access to artificial intelligence.

**pingpoint** is an open system where:

- **If you have hardware** → you run AI models locally and generate solutions for the community
- **If you don't have hardware** → you publish tasks and receive solutions from the network
- **Everyone wins** → every solution is public and strengthens collective knowledge

No central servers. No paid APIs. No middlemen. Just git + Ollama + people collaborating.

---

## How it works

```
1. [Anyone] Creates a GitHub Issue describing a task
2. [Matcher] Scans tasks and analyzes your machine
3. [Runner] Your local AI generates the solution
4. [Tester] The solution is validated automatically
5. [Validator] Schema, chain, and structure are checked
6. [PR] The solution is shared with the world
```

---

## The concept: rounds & the relay

Each collaborator gets **3 prompts per round**. You start fresh, refine twice, and pass the baton.

```
Task published
    ↓
┌─ Round 1 ──────────────────────────────┐
│  pingpoint run                          │
│    → Prompt 1/3 (task prompt) → v1     │
│  pingpoint run --challenge "..."        │
│    → Prompt 2/3 (your challenge) → v2  │
│  pingpoint run --challenge "..."        │
│    → Prompt 3/3 (your challenge) → v3  │
│    → Handoff instructions               │
│    → PASS THE BATON                     │
└─────────────────────────────────────────┘
    ↓
┌─ Round 2 (next collaborator) ─────────┐
│  git pull                               │
│  pingpoint assign                       │
│  pingpoint run                          │
│    → Prompt 1/3 builds on v3           │
│  ...                                     │
└─────────────────────────────────────────┘
```

---

## Integrity & validation

Every solution is part of a **tamper-proof hash chain**:

- Each solution stores the SHA256 hash of the previous one
- Any modification breaks the chain
- The `validate` command checks everything before a PR

---

## Repository structure

```
pingpoint/
├── tasks/              # Task definitions in YAML
├── solutions/          # Solutions (hash-chain verified)
│   └── <task-id>/
│       ├── v1.json     # Solution version 1
│       ├── v2.json     # Solution version 2
│       ├── v3.json     # Solution version 3
│       └── report.json # Full process report
├── .github/
│   └── workflows/
│       └── validate.yml  # PR validation
├── pingpoint/
│   ├── cli.py         # CLI commands
│   ├── profiler.py    # Hardware/model detection
│   ├── matcher.py     # Task assignment engine
│   ├── runner.py      # Local AI execution (Ollama)
│   ├── tester.py      # Solution evaluation
│   ├── validator.py   # Schema & structure validation
│   ├── db.py          # Local storage
│   ├── models.py      # Data models
│   └── __main__.py    # python -m pingpoint support
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Git
- At least one model pulled in Ollama (`ollama pull llama3.2`)

---

## Installation

```bash
git clone https://github.com/epilefelipe/pingpoint
cd pingpoint
pip install -e .
```

---

## Usage

```bash
# Check your machine specs and available models
pingpoint profile

# Import tasks from tasks/ directory
pingpoint assign

# --- Start a round (prompt 1/3) ---
pingpoint run

# --- Continue refining (prompts 2-3/3) ---
pingpoint run --challenge "Add error handling"
pingpoint run --challenge "Improve the design"

# --- View solutions ---
pingpoint show issue-1

# --- Validate schema, chain, and structure ---
pingpoint validate issue-1

# --- Generate comprehensive process report ---
pingpoint report issue-1

# --- Verify hash chain integrity ---
pingpoint verify issue-1
```

---

## How to contribute

### As a collaborator (you have hardware)

1. `git pull`
2. `pingpoint assign` — picks the best task for your machine
3. `pingpoint run` — starts a new round (prompt 1/3)
4. `pingpoint run --challenge "..."` — refine (prompts 2-3/3)
5. Write handoff instructions for the next person
6. `git add solutions/ && git commit`
7. `pingpoint validate <task-id>` — confirms everything is correct
8. `pingpoint report <task-id>` — generates the process report
9. Create a PR

### As a task creator (no hardware needed)

1. Create a YAML file in `tasks/`:

```yaml
title: "Your task title"
description: "Describe what you need"
prompt: "Instructions for the AI"
test_prompt: "Instructions for evaluator"
tags: [code, web]
issue_url: "https://github.com/your/repo/issues/1"
issue_number: 1
```

2. Open a PR with the task
3. Wait for collaborators to pick it up

### PR checklist

Before opening a PR, run:

```bash
pingpoint validate <task-id>    # Must show PASS
pingpoint verify <task-id>      # Must show VALID
pingpoint report <task-id>      # Generates report.json
```

The PR will be automatically validated by GitHub Actions.

---

## License

MIT — free to use, modify, and distribute.
