from pingpoint.models import Profile, Task, Solution, SolutionMetadata, TestResult


class TestProfile:
    def test_has_gpu_true(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["NVIDIA RTX 4090"], vram_gb=24.0,
            ollama_models=[], ollama_running=True, score=50.0,
        )
        assert p.has_gpu is True

    def test_has_gpu_unknown(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["Unknown"], vram_gb=None,
            ollama_models=[], ollama_running=True, score=0.0,
        )
        assert p.has_gpu is False

    def test_has_gpu_empty(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=[], vram_gb=None,
            ollama_models=[], ollama_running=True, score=0.0,
        )
        assert p.has_gpu is False

    def test_capability_high(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["NVIDIA RTX 4090"], vram_gb=24.0,
            ollama_models=[], ollama_running=True, score=50.0,
        )
        assert p.capability == "high"

    def test_capability_medium(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["NVIDIA GTX 1060"], vram_gb=6.0,
            ollama_models=[], ollama_running=True, score=50.0,
        )
        assert p.capability == "medium"

    def test_capability_low(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=["Unknown"], vram_gb=None,
            ollama_models=[], ollama_running=True, score=0.0,
        )
        assert p.capability == "low"

    def test_capability_low_no_gpu(self):
        p = Profile(
            platform="Linux", cpu="x86", cpu_cores=4, ram_gb=16.0,
            gpu=[], vram_gb=None,
            ollama_models=[], ollama_running=True, score=0.0,
        )
        assert p.capability == "low"


class TestTask:
    def test_to_dict(self, sample_task):
        d = sample_task.to_dict()
        assert d["id"] == "test-task"
        assert d["title"] == "Test Task"
        assert d["tags"] == ["code", "reasoning"]
        assert d["issue_url"] == "https://github.com/test/repo/issues/1"
        assert d["issue_number"] == 1

    def test_default_tags(self):
        t = Task(id="x", title="x", description="x", prompt="x", test_prompt="x")
        assert t.tags == []
        assert t.issue_url is None
        assert t.issue_number is None

    def test_to_dict_structure(self, sample_task):
        d = sample_task.to_dict()
        expected_keys = {"id", "title", "description", "prompt",
                         "test_prompt", "tags", "created_at",
                         "issue_url", "issue_number"}
        assert set(d.keys()) == expected_keys


class TestSolution:
    def test_id_property(self, sample_solution):
        assert sample_solution.id == "test-task/v1"

    def test_compute_hash_consistency(self, sample_solution):
        h1 = sample_solution.compute_hash()
        h2 = sample_solution.compute_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_changes_with_output(self, sample_solution):
        h1 = sample_solution.compute_hash()
        sample_solution.output = "different output"
        h2 = sample_solution.compute_hash()
        assert h1 != h2

    def test_compute_hash_changes_with_version(self, sample_solution):
        h1 = sample_solution.compute_hash()
        sample_solution.version = 2
        h2 = sample_solution.compute_hash()
        assert h1 != h2

    def test_compute_hash_changes_with_run_number(self, sample_solution):
        h1 = sample_solution.compute_hash()
        sample_solution.run_number = 2
        h2 = sample_solution.compute_hash()
        assert h1 != h2

    def test_default_round_and_run_number(self, sample_metadata):
        sol = Solution(
            task_id="t", version=1,
            prompt_used="p", output="o", metadata=sample_metadata,
        )
        assert sol.run_number == 1
        assert sol.round == 1


class TestSolutionMetadata:
    def test_fields(self, sample_metadata):
        assert sample_metadata.model == "llama3.2"
        assert sample_metadata.temperature == 0.7
        assert sample_metadata.max_tokens == 2048
        assert sample_metadata.execution_time_s == 10.5


class TestTestResult:
    def test_id_property(self, sample_test_result):
        assert sample_test_result.id == "test-task/v1/test"

    def test_default_created_at(self, sample_test_result):
        assert sample_test_result.created_at is not None

    def test_default_improvement_found(self):
        from pingpoint.models import TestResult
        r = TestResult(
            task_id="t", version=1, passed=True, score=50.0,
            summary="s", details="d", improvement_found=True,
        )
        assert r.improvement_found is True
