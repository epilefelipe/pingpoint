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
1. [Anyone] Creates a GitHub Issue describing a task:
   "Write a poem about artificial intelligence"
   "Optimize this code for performance"
   "Translate this document to Spanish"

2. [Matcher] Scans open Issues and analyzes your machine:
   GPU? CPU? What AI models do you have installed?

3. [Matcher] Assigns the best Issue for you:
   Powerful GPU → complex tasks (vision, code, reasoning)
   Normal laptop → text tasks, analysis, translation
   No GPU → tasks that need human validation

4. [Runner] Your local AI generates the solution:
   Uses your models (Llama, Mistral, DeepSeek, etc.)
   No data is sent to any external server

5. [Test] The solution is tested automatically:
   Does it solve the original task?
   Does it add something new to previous versions?

6. [Repository] The solution is saved and shared via PR:
   Other collaborators pick it up and improve it
   The best version emerges from collective intelligence
   Everyone accesses it for free
```

---

## The concept: the relay

One person starts a solution. Finishes it. The next person picks it up and improves it. Like a relay race:

```
Task published
    ↓
Collaborator A runs their local AI → generates v1
    ↓  test: v1 must solve the original task
    ↓  (done, passes the baton)
Collaborator B takes v1, improves with another model → v2
    ↓  test: v2 must ADD something new vs v1
    ↓  (done, passes the baton)
Collaborator C takes v2, strengthens it → v3
    ↓  test: v3 must ADD something new vs v2
    ↓
...
    ↓
Final solution strengthened by the entire chain
```

Each link in the chain:
- Runs **on their own machine** (no costs)
- Uses **their own model** (different perspective)
- Sees **the full history** (prompts, outputs, hardware)
- **Improves** what already exists
- **Must add something new** to pass the test
- **Passes the baton** to the next person

## The test

Before any solution is accepted, it must pass a test:

```
1. Take the current best solution (v1)
2. Run a simple test prompt related to the task
3. Check if the result actually solves the problem
4. If yes → generate v2 by improving v1
5. Compare v2 vs v1 — does v2 add something new?
   YES → accepted, pass the baton
   NO  → rejected, try again
```

The test runs locally using your own AI models. No external validation needed.

Simple. Powerful. Collaborative.

---

## Total transparency

Every solution saves **the entire process**:

```
Original task
    ↓
Used prompt (verbatim)
    ↓
Model + parameters (temperature, tokens, etc.)
    ↓
Collaborator's hardware (GPU, RAM, execution time)
    ↓
Raw AI output
    ↓
Iterations and improvements from each collaborator
```

Nothing is hidden. The prompt, the model, the parameters, the hardware, every attempt — **everything is public** so anyone can reproduce, learn, and improve.

---

## Strengthening chain

Each task passes from hand to hand. Each collaborator receives the full history and improves it:

```
Task published
    ↓
┌─ Collaborator A (Mistral, temp=0.7)
│  Prompt: "..."
│  Output: "..."
│  Hardware: RTX 3060, 12GB VRAM
│  Time: 45s
│  Passes baton → 
└──────────────────────────────────────┘
    ↓
┌─ Collaborator B (receives v1, Llama, temp=0.5)
│  Prompt: "Improve this: [v1 full]"
│  Output: "..."
│  Hardware: M3 Pro 18GB
│  Time: 30s
│  Passes baton → 
└──────────────────────────────────────┘
    ↓
┌─ Collaborator C (receives v2, DeepSeek)
│  Prompt: "Refine this: [v2 + history]"
│  Output: "..."
│  Hardware: RTX 4090, 24GB VRAM
│  Time: 20s
│  Passes baton → 
└──────────────────────────────────────┘
    ↓
Final solution strengthened by the entire chain
```

The more people participate, the more robust the solution. And the entire process — every prompt, every output, every model — is documented.

---

## Repository structure

```
pingpoint/
├── tasks/            # Task definitions in YAML
├── pingpoint/        # Python package
│   ├── cli.py        # CLI commands
│   ├── profiler.py   # Hardware/model detection
│   ├── matcher.py    # Task assignment engine
│   ├── runner.py     # Local AI execution (Ollama)
│   ├── tester.py     # Solution validation
│   └── db.py         # Local storage (~/.pingpoint)
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
git clone https://github.com/your-user/pingpoint
cd pingpoint
pip install -e .
```

---

## Usage

```bash
# Check your machine specs and available models
pingpoint profile

# Scan tasks and assign the best one for you
pingpoint assign

# Run the task with your local AI (saves prompt, output, model, hardware)
pingpoint run

# View the full process of any solution
pingpoint show <task-id>
```

---

## Contributing

**You don't need to know how to code to contribute.**

- **Create a task YAML**: describe a problem you want solved by the community
- **With a GPU**: run complex tasks, strengthen solutions
- **Without a GPU**: validate results, improve prompts, share the project
- **Developers**: improve the matcher, the runner, the tester

Every contribution counts. pingpoint's power is in its network of collaborators.

---

## License

MIT — free to use, modify, and distribute.

---

## Vision

A world where access to artificial intelligence depends not on how much money you have, but on your willingness to collaborate.
