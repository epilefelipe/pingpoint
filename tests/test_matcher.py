import math

from pingpoint.matcher import score_task, find_best_task, compute_dimension_scores
from pingpoint.models import Profile, Task


class TestComputeDimensionScores:
    def test_low_profile_dims(self, low_profile):
        dims = compute_dimension_scores(low_profile)
        for d in ["vision", "code", "reasoning", "creative", "translation",
                   "analysis", "writing", "brainstorming", "general"]:
            assert 0.0 <= dims[d] <= 10.0
        assert dims["code"] < dims["creative"]

    def test_medium_profile_dims(self, medium_profile):
        dims = compute_dimension_scores(medium_profile)
        for d in ["vision", "code", "reasoning", "creative", "translation",
                   "analysis", "writing", "brainstorming", "general"]:
            assert 0.0 <= dims[d] <= 10.0

    def test_high_profile_dims(self, high_profile):
        dims = compute_dimension_scores(high_profile)
        assert dims["vision"] == 10.0
        assert dims["code"] == 10.0
        assert dims["reasoning"] >= 9.0


class TestScoreTask:
    def test_known_tags_medium(self, medium_profile, sample_task):
        score = score_task(medium_profile, sample_task, version_count=0)
        # Should be positive and reasonable
        assert 30.0 <= score <= 50.0

    def test_vision_model_bonus(self):
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
        assert score >= 40.0

    def test_no_solutions_bonus(self, medium_profile, sample_task):
        score_0 = score_task(medium_profile, sample_task, version_count=0)
        score_1 = score_task(medium_profile, sample_task, version_count=1)
        score_3 = score_task(medium_profile, sample_task, version_count=3)
        assert score_0 > score_1 > score_3

    def test_version_decay_smooth(self, medium_profile, sample_task):
        """Version bonus should decay exponentially, not jump."""
        diff_0_1 = score_task(medium_profile, sample_task, 0) - score_task(medium_profile, sample_task, 1)
        diff_4_5 = score_task(medium_profile, sample_task, 4) - score_task(medium_profile, sample_task, 5)
        assert diff_0_1 > diff_4_5 > 0

    def test_unknown_tag(self, medium_profile):
        task = Task(
            id="unknown", title="Unknown", description="d",
            prompt="p", test_prompt="tp", tags=["some_unknown_tag"],
        )
        score = score_task(medium_profile, task, version_count=0)
        assert 20.0 <= score <= 40.0

    def test_model_tag_match(self):
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
        assert score >= 40.0

    def test_low_capability(self, low_profile, sample_task):
        score = score_task(low_profile, sample_task, version_count=0)
        assert 15.0 <= score <= 25.0

    def test_author_penalty(self, medium_profile, sample_task):
        score_no_rounds = score_task(medium_profile, sample_task, 0, author_rounds=0)
        score_with_rounds = score_task(medium_profile, sample_task, 0, author_rounds=3)
        assert score_no_rounds > score_with_rounds

    def test_complexity_factor(self, medium_profile):
        simple = Task(id="s", title="S", description="x", prompt="x", test_prompt="tp", tags=["code"])
        complex_t = Task(id="c", title="C", description="x" * 2000, prompt="x" * 2000, test_prompt="tp", tags=["code"])
        score_simple = score_task(medium_profile, simple, 0)
        score_complex = score_task(medium_profile, complex_t, 0)
        # Simple tasks get a slight bonus over very complex ones
        assert score_simple > score_complex


class TestFindBestTask:
    def test_returns_highest_score(self, medium_profile, sample_task):
        lower_task = Task(
            id="lower", title="Lower", description="d",
            prompt="p", test_prompt="tp", tags=["writing"],
        )
        tasks = [lower_task, sample_task]
        counts = {"lower": 0, "test-task": 0}
        best = find_best_task(medium_profile, tasks, counts)
        assert best is not None
        assert best.id == "test-task"

    def test_empty_tasks(self, medium_profile):
        best = find_best_task(medium_profile, [], {})
        assert best is None

    def test_considers_version_count(self, medium_profile, sample_task):
        task_a = Task(
            id="a", title="A", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        task_b = Task(
            id="b", title="B", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        tasks = [task_a, task_b]
        counts = {"a": 0, "b": 2}
        best = find_best_task(medium_profile, tasks, counts)
        assert best is not None
        assert best.id == "a"

    def test_author_preference(self, medium_profile, sample_task):
        task_a = Task(
            id="a", title="A", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        task_b = Task(
            id="b", title="B", description="d",
            prompt="p", test_prompt="tp", tags=["code"],
        )
        tasks = [task_a, task_b]
        counts = {"a": 0, "b": 0}
        # Author already has 3 rounds on task_a, prefers task_b
        author_counts = {"a": 3, "b": 0}
        best = find_best_task(medium_profile, tasks, counts, author_counts)
        assert best is not None
        assert best.id == "b"
