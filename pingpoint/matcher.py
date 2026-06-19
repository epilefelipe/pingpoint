from pingpoint.models import Profile, Task


TASK_WEIGHTS = {
    "vision": {"high": 10, "medium": 5, "low": 0},
    "code": {"high": 10, "medium": 7, "low": 2},
    "reasoning": {"high": 10, "medium": 8, "low": 3},
    "creative": {"high": 5, "medium": 8, "low": 10},
    "translation": {"high": 3, "medium": 7, "low": 10},
    "analysis": {"high": 8, "medium": 10, "low": 7},
    "writing": {"high": 3, "medium": 7, "low": 10},
    "brainstorming": {"high": 5, "medium": 8, "low": 10},
    "general": {"high": 8, "medium": 10, "low": 8},
}


def score_task(profile: Profile, task: Task, version_count: int) -> float:
    capability = profile.capability

    tag_score = 0.0
    for tag in task.tags:
        tag_lower = tag.lower()
        if tag_lower in TASK_WEIGHTS:
            tag_score += TASK_WEIGHTS[tag_lower].get(capability, 5)
        else:
            tag_score += 5

    avg_tag_score = tag_score / max(len(task.tags), 1)

    model_score = 0.0
    for tag in task.tags:
        for model in profile.ollama_models:
            model_lower = model.lower()
            tag_lower = tag.lower()
            if tag_lower in model_lower or model_lower in tag_lower:
                model_score += 10.0
            if any(kw in model_lower for kw in ("vision", "llava")):
                if tag_lower == "vision":
                    model_score += 20.0

    model_score = min(model_score, 30.0)

    existing_solutions_score = 0.0
    if version_count == 0:
        existing_solutions_score = 15.0
    elif version_count < 3:
        existing_solutions_score = 10.0
    else:
        existing_solutions_score = 5.0

    total = avg_tag_score + model_score + existing_solutions_score
    return round(total, 1)


def find_best_task(
    profile: Profile,
    tasks: list[Task],
    solution_counts: dict[str, int],
) -> Task | None:
    if not tasks:
        return None

    best_task = None
    best_score = -1.0

    for task in tasks:
        version_count = solution_counts.get(task.id, 0)
        score = score_task(profile, task, version_count)
        if score > best_score:
            best_score = score
            best_task = task

    return best_task
