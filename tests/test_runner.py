import json
import subprocess
from unittest.mock import patch, MagicMock

from pingpoint.runner import clean_ansi, call_ollama_api, get_ollama_version


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
    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = json.dumps(
            {"response": "Hello response"}
        ).encode()
        mock_urlopen.return_value = mock_response
        result = call_ollama_api("llama3.2", "test prompt", timeout=120)
        assert result is not None
        output, elapsed = result
        assert output == "Hello response"
        assert elapsed >= 0
        mock_urlopen.assert_called_once()

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("error")
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_empty_output(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = json.dumps(
            {"response": "  "}
        ).encode()
        mock_urlopen.return_value = mock_response
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_os_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("No connection")
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_json_decode_error(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = b"not json"
        mock_urlopen.return_value = mock_response
        result = call_ollama_api("llama3.2", "test prompt")
        assert result is None

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_passthrough_params(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = json.dumps(
            {"response": "ok"}
        ).encode()
        mock_urlopen.return_value = mock_response
        call_ollama_api("mistral", "some prompt", temperature=0.5, max_tokens=1024, timeout=60)
        args, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 60


class TestGetOllamaVersion:
    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_version_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = json.dumps(
            {"version": "0.1.32"}
        ).encode()
        mock_urlopen.return_value = mock_response
        assert get_ollama_version() == "0.1.32"

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_version_not_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = json.dumps(
            {}
        ).encode()
        mock_urlopen.return_value = mock_response
        assert get_ollama_version() == "unknown"

    @patch("pingpoint.runner.urllib.request.urlopen")
    def test_server_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("error")
        assert get_ollama_version() == "unknown"
