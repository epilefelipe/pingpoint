import json
from unittest.mock import patch, MagicMock

from pingpoint.backend import OllamaBackend, OMLXBackend, detect_backend, get_backend


class TestOllamaBackend:
    def test_name(self):
        assert OllamaBackend().name == "ollama"

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_probe_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps(
            {"version": "0.1.0"}
        ).encode()
        mock_urlopen.return_value = mock_resp
        assert OllamaBackend.probe() is True

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_probe_fail(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert OllamaBackend.probe() is False

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps(
            {"response": "Hello world"}
        ).encode()
        mock_urlopen.return_value = mock_resp
        result = OllamaBackend().generate("llama3.2", "hi")
        assert result is not None
        output, elapsed = result
        assert output == "Hello world"
        assert elapsed >= 0

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_generate_empty(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps(
            {"response": "  "}
        ).encode()
        mock_urlopen.return_value = mock_resp
        assert OllamaBackend().generate("llama3.2", "hi") is None

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_list_models(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps({
            "models": [{"name": "llama3.2"}, {"name": "mistral"}]
        }).encode()
        mock_urlopen.return_value = mock_resp
        models = OllamaBackend().list_models()
        assert models == ["llama3.2", "mistral"]

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_list_models_empty(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert OllamaBackend().list_models() == []


class TestOMLXBackend:
    def test_name(self):
        assert OMLXBackend().name == "omlx"

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_probe_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps(
            {"data": [{"id": "llama3.2"}]}
        ).encode()
        mock_urlopen.return_value = mock_resp
        assert OMLXBackend.probe() is True

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_probe_fail(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert OMLXBackend.probe() is False

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Hello from oMLX"}}]
        }).encode()
        mock_urlopen.return_value = mock_resp
        result = OMLXBackend().generate("llama3.2", "hi")
        assert result is not None
        output, elapsed = result
        assert output == "Hello from oMLX"
        assert elapsed >= 0

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_generate_empty(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "  "}}]
        }).encode()
        mock_urlopen.return_value = mock_resp
        assert OMLXBackend().generate("llama3.2", "hi") is None

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_list_models(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps({
            "data": [{"id": "llama3.2"}, {"id": "mistral"}]
        }).encode()
        mock_urlopen.return_value = mock_resp
        models = OMLXBackend().list_models()
        assert models == ["llama3.2", "mistral"]

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_list_models_fail(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert OMLXBackend().list_models() == []

    @patch("pingpoint.backend.urllib.request.urlopen")
    def test_version(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps({
            "version": "0.4.4"
        }).encode()
        mock_urlopen.return_value = mock_resp
        assert OMLXBackend().version() == "0.4.4"


class TestDetectBackend:
    @patch("pingpoint.backend.OllamaBackend.probe")
    @patch("pingpoint.backend.OMLXBackend.probe")
    def test_detects_ollama_first(self, mock_omlx_probe, mock_ollama_probe):
        mock_ollama_probe.return_value = True
        mock_omlx_probe.return_value = False
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "ollama"

    @patch("pingpoint.backend.OllamaBackend.probe")
    @patch("pingpoint.backend.OMLXBackend.probe")
    def test_detects_omlx(self, mock_omlx_probe, mock_ollama_probe):
        mock_ollama_probe.return_value = False
        mock_omlx_probe.return_value = True
        backend = detect_backend()
        assert backend is not None
        assert backend.name == "omlx"

    @patch("pingpoint.backend.OllamaBackend.probe")
    @patch("pingpoint.backend.OMLXBackend.probe")
    def test_detects_none(self, mock_omlx_probe, mock_ollama_probe):
        mock_ollama_probe.return_value = False
        mock_omlx_probe.return_value = False
        assert detect_backend() is None

    @patch("pingpoint.backend.OllamaBackend.probe")
    @patch("pingpoint.backend.OMLXBackend.probe")
    def test_prefer_omlx(self, mock_omlx_probe, mock_ollama_probe):
        mock_ollama_probe.return_value = True
        mock_omlx_probe.return_value = True
        backend = detect_backend(prefer="omlx")
        assert backend is not None
        assert backend.name == "omlx"


class TestGetBackend:
    def test_get_ollama(self):
        backend = get_backend("ollama")
        assert backend is not None
        assert backend.name == "ollama"

    def test_get_omlx(self):
        backend = get_backend("omlx")
        assert backend is not None
        assert backend.name == "omlx"

    def test_get_unknown(self):
        assert get_backend("unknown") is None
