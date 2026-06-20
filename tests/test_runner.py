import json
from unittest.mock import patch, MagicMock

from pingpoint.runner import clean_ansi, call_ollama_api, get_ollama_version
from pingpoint import runner as runner_module


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


class TestCallOllamaApi:
    @patch.object(runner_module, "_get_backend")
    def test_successful_call(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = ("Hello response", 0.5)
        mock_get_backend.return_value = mock_backend
        result = call_ollama_api("llama3.2", "test prompt", timeout=120)
        assert result is not None
        output, elapsed = result
        assert output == "Hello response"
        assert elapsed >= 0

    @patch.object(runner_module, "_get_backend")
    def test_backend_none(self, mock_get_backend):
        mock_get_backend.return_value = None
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch.object(runner_module, "_get_backend")
    def test_empty_output(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = None
        mock_get_backend.return_value = mock_backend
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch.object(runner_module, "_get_backend")
    def test_generate_exception(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = Exception("fail")
        mock_get_backend.return_value = mock_backend
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch.object(runner_module, "_get_backend")
    def test_passthrough_params(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = ("ok", 0.5)
        mock_get_backend.return_value = mock_backend
        call_ollama_api("mistral", "some prompt", temperature=0.5, max_tokens=1024, timeout=60)
        mock_backend.generate.assert_called_once()
        _, kwargs = mock_backend.generate.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 1024
        assert kwargs["timeout"] == 60


class TestGetOllamaVersion:
    @patch.object(runner_module, "_get_backend")
    def test_version_found(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.version.return_value = "0.1.32"
        mock_get_backend.return_value = mock_backend
        assert get_ollama_version() == "0.1.32"

    @patch.object(runner_module, "_get_backend")
    def test_version_not_found(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_backend.version.return_value = "unknown"
        mock_get_backend.return_value = mock_backend
        assert get_ollama_version() == "unknown"

    @patch.object(runner_module, "_get_backend")
    def test_server_error(self, mock_get_backend):
        mock_get_backend.return_value = None
        assert get_ollama_version() == "unknown"
