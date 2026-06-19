import math
from typing import Optional

from pingpoint.models import Profile, Task


DIMENSIONS = [
    "vision", "code", "reasoning", "creative", "translation",
    "analysis", "writing", "brainstorming", "general",
]

MODEL_FEATURES: dict[str, dict[str, float]] = {
    "llama": {"code": 3, "reasoning": 3, "general": 2},
    "mistral": {"code": 3, "reasoning": 2, "general": 2},
    "codellama": {"code": 8, "reasoning": 4},
    "deepseek": {"code": 5, "reasoning": 4},
    "mixtral": {"reasoning": 6, "code": 5, "general": 3},
    "vision": {"vision": 10},
    "llava": {"vision": 10},
    "gemma": {"general": 2, "code": 2},
    "phi": {"code": 3, "reasoning": 2},
    "qwen": {"code": 3, "general": 2},
    "neural": {"creative": 3, "writing": 3},
    "nous": {"reasoning": 3, "code": 2},
    "command": {"reasoning": 3, "general": 2},
    "dbrx": {"reasoning": 4, "code": 3},
    "starcoder": {"code": 7},
    "wizard": {"code": 4, "reasoning": 3},
    "zephyr": {"general": 2, "reasoning": 2},
    "tinyllama": {"general": 1},
    "orca": {"reasoning": 3, "general": 2},
    "solar": {"reasoning": 3, "general": 2},
    "goliath": {"reasoning": 4, "code": 3, "general": 3},
    "yi": {"general": 2, "code": 2},
    "falcon": {"general": 2, "code": 2},
    "dolphin": {"general": 2, "reasoning": 2},
    "openchat": {"general": 2, "code": 2},
    "starling": {"reasoning": 3, "general": 2},
    "math": {"reasoning": 6, "code": 3},
    "coder": {"code": 6},
}


def compute_dimension_scores(profile: Profile) -> dict[str, float]:
    gpu = profile.has_gpu
    vram = profile.vram_gb or 0
    cores = profile.cpu_cores
    ram = profile.ram_gb
    score_val = profile.score

    def clip(v: float) -> float:
        return max(0.0, min(10.0, v))

    base = score_val / 10.0

    gpu_bonus = 1.0 + (min(vram / 8, 3.0) if gpu else 0.0)
    cpu_bonus = min(cores / 4, 2.0)
    ram_bonus = min(ram / 8, 2.0)

    return {
        "vision": clip(base * (gpu_bonus + 0.5) / 1.5),
        "code": clip(base * (cpu_bonus + ram_bonus + 1) / 3.0 + (2 if gpu else 0)),
        "reasoning": clip(base * (cpu_bonus + ram_bonus + 1) / 3.0),
        "creative": clip(base * 1.0),
        "translation": clip(base * 0.8 + (1 if ram > 16 else 0)),
        "analysis": clip(base * (ram_bonus + 1) / 2.0 + (1 if gpu else 0)),
        "writing": clip(base * 0.9 + (1 if ram > 8 else 0)),
        "brainstorming": clip(base * 1.1),
        "general": clip(base * 1.0),
    }


def score_task(
    profile: Profile,
    task: Task,
    version_count: int,
    author_rounds: int = 0,
) -> float:
    dims = compute_dimension_scores(profile)

    tag_score = 0.0
    for tag in task.tags:
        tl = tag.lower()
        if tl in dims:
            tag_score += dims[tl]
        else:
            tag_score += 5.0

    avg_tag_score = tag_score / max(len(task.tags), 1)

    model_score = 0.0
    for model in profile.ollama_models:
        ml = model.lower()
        for kw, features in MODEL_FEATURES.items():
            if kw in ml:
                for dim, bonus in features.items():
                    for tag in task.tags:
                        if dim == tag.lower():
                            model_score += bonus

    model_score = min(model_score, 30.0)

    version_bonus = 15.0 * math.exp(-version_count / 3.0)
    author_penalty = float(author_rounds) * 2.0

    complexity = min(len(task.prompt) + len(task.description), 2000) / 2000.0
    complexity_factor = 1.0 + (1.0 - complexity) * 0.15

    total = (avg_tag_score + model_score + version_bonus - author_penalty) * complexity_factor
    return round(max(total, 0.0), 1)


def find_best_task(
    profile: Profile,
    tasks: list[Task],
    solution_counts: dict[str, int],
    author_solution_counts: Optional[dict[str, int]] = None,
) -> Optional[Task]:
    if not tasks:
        return None

    if author_solution_counts is None:
        author_solution_counts = {}

    best_task: Optional[Task] = None
    best_score = -1.0

    for task in tasks:
        version_count = solution_counts.get(task.id, 0)
        author_rounds = author_solution_counts.get(task.id, 0)
        score = score_task(profile, task, version_count, author_rounds)
        if score > best_score:
            best_score = score
            best_task = task

    return best_task
