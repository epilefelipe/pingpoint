import subprocess
from unittest.mock import patch, MagicMock

from pingpoint.runner import clean_ansi, call_ollama, get_ollama_version, run_solution


class TestCleanAnsi:
    def test_clean_ansi_removes_codes(self):
        result = clean_ansi("\x1b[32mhello\x1b[0m")
        assert result == "hello"

    def test_clean_ansi_no_ansi(self):
        result = clean_ansi("hello world")
        assert result == "hello world"

    def test_clean_ansi_empty(self):
        assert clean_ansi("") == ""

    def test_clean_ansi_multiple_codes(self):
        result = clean_ansi("\x1b[1m\x1b[31mbold red\x1b[0m")
        assert result == "bold red"

    def test_clean_ansi_complex_sequences(self):
        result = clean_ansi("\x1b[38;2;255;255;255mwhite\x1b[0m")
        assert result == "white"


class TestCallOllama:
    @patch("pingpoint.runner.subprocess.run")
    def test_successful_call(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\x1b[32mHello response\x1b[0m",
            stderr="",
        )
        result = call_ollama("llama3.2", "test prompt", timeout=120)
        assert result is not None
        output, elapsed = result
        assert output == "Hello response"
        assert elapsed >= 0
        mock_run.assert_called_once()

    @patch("pingpoint.runner.subprocess.run")
    def test_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = call_ollama("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="  ", stderr="")
        result = call_ollama("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.subprocess.run")
    def test_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = call_ollama("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=1)
        result = call_ollama("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.subprocess.run")
    def test_passthrough_params(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        call_ollama("mistral", "some prompt", temperature=0.5, max_tokens=1024, timeout=60)
        args, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 60


class TestGetOllamaVersion:
    @patch("pingpoint.runner.subprocess.run")
    def test_version_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.32\n", stderr="")
        assert get_ollama_version() == "ollama version 0.1.32"

    @patch("pingpoint.runner.subprocess.run")
    def test_version_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        assert get_ollama_version() == "unknown"

    @patch("pingpoint.runner.subprocess.run")
    def test_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert get_ollama_version() == "not installed"


class TestRunSolution:
    @patch("pingpoint.runner.call_ollama")
    @patch("pingpoint.runner.get_ollama_version")
    def test_run_solution_success(self, mock_version, mock_call):
        mock_call.return_value = ("print('hello')", 5.0)
        mock_version.return_value = "ollama 0.1.0"

        sol = run_solution("llama3.2", "task prompt", hardware_summary="CPU only")
        assert sol is not None
        assert sol.output == "print('hello')"
        assert sol.metadata.model == "llama3.2"
        assert sol.metadata.execution_time_s == 5.0
        assert sol.metadata.ollama_version == "ollama 0.1.0"
        assert sol.prompt_used == "task prompt"

    @patch("pingpoint.runner.call_ollama")
    def test_run_solution_with_improvement(self, mock_call, sample_metadata):
        mock_call.return_value = ("improved output", 3.0)
        sol = run_solution("llama3.2", "task prompt", improvement_prompt="improvement prompt")
        assert sol is not None
        assert sol.prompt_used == "improvement prompt"
        assert sol.output == "improved output"

    @patch("pingpoint.runner.call_ollama")
    def test_run_solution_failure(self, mock_call):
        mock_call.return_value = None
        sol = run_solution("llama3.2", "task prompt")
        assert sol is None
