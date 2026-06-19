from pingpoint.matcher import score_task, find_best_task
from pingpoint.models import Profile, Task


class TestScoreTask:
    def test_score_with_known_tags(self, medium_profile, sample_task):
        score = score_task(medium_profile, sample_task, version_count=0)
        # code=7 + reasoning=8 => avg=7.5, plus model_score (llama in "llama3.2" matching "code"/"reasoning"? let's compute)
        # model_score: for tag "code", model "llama3.2": "code" not in "llama3.2", "llama3.2" not in "code" -> 0
        #             tag "reasoning", model "llama3.2": "reasoning" not in "llama3.2", "llama3.2" not in "reasoning" -> 0
        #             same for "mistral" -> 0
        # existing: 15.0 (version_count=0)
        # total: 7.5 + 0 + 15.0 = 22.5
        assert score == 22.5

    def test_score_vision_model_bonus(self):
        profile = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["NVIDIA RTX 4090"], vram_gb=24.0,
            ollama_models=["llava"], ollama_running=True, score=90.0,
        )
        task = Task(
            id="vision-test", title="Vision Task", description="d",
            prompt="p", test_prompt="tp", tags=["vision"],
        )
        score = score_task(profile, task, version_count=0)
        # tag_score: vision+high=10 => avg=10
        # model_score: "vision" not in "llava", "llava" not in "vision" => 0 basic match
        #   any(kw in model_lower ...) -> "llava" in "llava" -> True, tag_lower == "vision" -> +20
        #   capped at 30 => 20
        # existing: 15.0
        # total: 10 + 20 + 15 = 45.0
        assert score == 45.0

    def test_score_no_solutions_bonus(self, medium_profile, sample_task):
        score_0 = score_task(medium_profile, sample_task, version_count=0)
        score_1 = score_task(medium_profile, sample_task, version_count=1)
        score_3 = score_task(medium_profile, sample_task, version_count=3)
        assert score_0 > score_1 > score_3

    def test_score_unknown_tag(self, medium_profile):
        task = Task(
            id="unknown", title="Unknown", description="d",
            prompt="p", test_prompt="tp", tags=["some_unknown_tag"],
        )
        score = score_task(medium_profile, task, version_count=0)
        # unknown tag defaults to 5, avg=5
        # model_score: 0 (no match)
        # existing: 15
        # total: 20.0
        assert score == 20.0

    def test_score_model_tag_match(self):
        profile = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["NVIDIA RTX 4090"], vram_gb=24.0,
            ollama_models=["llama3.2-vision"], ollama_running=True, score=90.0,
        )
        task = Task(
            id="vision-test", title="Vision Task", description="d",
            prompt="p", test_prompt="tp", tags=["vision"],
        )
        score = score_task(profile, task, version_count=0)
        # tag_score: vision+high=10, avg=10
        # model_score: "vision" in "llama3.2-vision" -> +10, vision kw + llava kw check: "llama3.2-vision" not in ("vision","llava") tags
        #   wait: any(kw in model_lower ...) -> "vision" in "llama3.2-vision" -> True, tag_lower == "vision" -> +20
        #   model_score = 10 + 20 = 30, capped at 30
        # existing: 15
        # total: 10 + 30 + 15 = 55.0
        assert score == 55.0

    def test_score_low_capability(self, low_profile, sample_task):
        score = score_task(low_profile, sample_task, version_count=0)
        # code=2, reasoning=3 => avg=2.5
        # model_score: 0
        # existing: 15
        # total: 17.5
        assert score == 17.5


class TestFindBestTask:
    def test_find_best_returns_highest_score(self, medium_profile, sample_task):
        lower_task = Task(
            id="lower", title="Lower", description="d",
            prompt="p", test_prompt="tp", tags=["writing"],
        )
        tasks = [lower_task, sample_task]
        counts = {"lower": 0, "test-task": 0}
        best = find_best_task(medium_profile, tasks, counts)
        assert best is not None
        assert best.id == "test-task"

    def test_find_best_with_empty_tasks(self, medium_profile):
        best = find_best_task(medium_profile, [], {})
        assert best is None

    def test_find_best_considers_version_count(self, medium_profile, sample_task):
        task_a = Task(
            id="a", title="A", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        task_b = Task(
            id="b", title="B", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        tasks = [task_a, task_b]
        # task_a has 0 solutions (bonus 15), task_b has 2 solutions (bonus 10)
        counts = {"a": 0, "b": 2}
        best = find_best_task(medium_profile, tasks, counts)
        assert best is not None
        assert best.id == "a"
