import platform
import subprocess
from unittest.mock import patch, MagicMock

from pingpoint.profiler import (
    get_ollama_models, is_ollama_running, get_ollama_version,
    get_gpu_info, get_cpu_info, get_ram_gb, profile,
)


class TestGetOllamaModels:
    @patch("pingpoint.profiler.OllamaBackend")
    def test_models_found(self, mock_backend_cls):
        mock_backend_cls.return_value.list_models.return_value = ["llama3.2", "mistral"]
        models = get_ollama_models()
        assert models == ["llama3.2", "mistral"]

    @patch("pingpoint.profiler.OllamaBackend")
    def test_no_models(self, mock_backend_cls):
        mock_backend_cls.return_value.list_models.return_value = []
        models = get_ollama_models()
        assert models == []

    @patch("pingpoint.profiler.OllamaBackend")
    def test_ollama_not_found(self, mock_backend_cls):
        mock_backend_cls.return_value.list_models.side_effect = Exception()
        models = get_ollama_models()
        assert models == []


class TestIsOllamaRunning:
    @patch("pingpoint.profiler.OllamaBackend.probe")
    def test_running(self, mock_probe):
        mock_probe.return_value = True
        assert is_ollama_running() is True

    @patch("pingpoint.profiler.OllamaBackend.probe")
    def test_not_running(self, mock_probe):
        mock_probe.return_value = False
        assert is_ollama_running() is False

    @patch("pingpoint.profiler.OllamaBackend.probe")
    def test_not_installed(self, mock_probe):
        mock_probe.return_value = False
        assert is_ollama_running() is False


class TestGetOllamaVersion:
    @patch("pingpoint.profiler.OllamaBackend")
    def test_version(self, mock_backend_cls):
        mock_backend_cls.return_value.version.return_value = "ollama version 0.1.32"
        assert get_ollama_version() == "ollama version 0.1.32"

    @patch("pingpoint.profiler.OllamaBackend")
    def test_error(self, mock_backend_cls):
        mock_backend_cls.return_value.version.return_value = "unknown"
        assert get_ollama_version() == "unknown"

    @patch("pingpoint.profiler.OllamaBackend")
    def test_not_installed(self, mock_backend_cls):
        mock_backend_cls.return_value.version.side_effect = Exception()
        assert get_ollama_version() != "not installed"  # now returns "unknown"


class TestGetGpuInfo:
    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_windows(self, mock_run, mock_system):
        mock_system.return_value = "Windows"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Name\nNVIDIA RTX 4090\n",
        )
        gpus, vram = get_gpu_info()
        assert gpus == ["NVIDIA RTX 4090"]
        assert vram is None

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_linux_nvidia(self, mock_run, mock_system):
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA RTX 4090, 24576 MiB\n",
        )
        gpus, vram = get_gpu_info()
        assert gpus == ["NVIDIA RTX 4090"]
        assert vram == 24.0

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_linux_no_nvidia(self, mock_run, mock_system):
        mock_system.return_value = "Linux"
        mock_run.side_effect = FileNotFoundError()
        gpus, vram = get_gpu_info()
        assert gpus == []
        assert vram is None

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_macos(self, mock_run, mock_system):
        mock_system.return_value = "Darwin"

        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "system_profiler" in str(cmd):
                return MagicMock(returncode=0, stdout="Chip: Apple M2 Pro\n")
            elif "sysctl" in str(cmd):
                return MagicMock(returncode=0, stdout="hw.memsize: 17179869184\n")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        gpus, vram = get_gpu_info()
        assert len(gpus) >= 1
        assert vram == 16.0


class TestGetCpuInfo:
    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_windows(self, mock_run, mock_system):
        mock_system.return_value = "Windows"

        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "NumberOfCores" in str(cmd):
                return MagicMock(returncode=0, stdout="NumberOfCores\n8\n")
            return MagicMock(returncode=0, stdout="Name\nIntel Core i7\n")

        mock_run.side_effect = side_effect
        name, cores = get_cpu_info()
        assert "Intel" in name
        assert cores == 8

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_linux(self, mock_run, mock_system):
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0, stdout="8\n")
        name, cores = get_cpu_info()
        assert cores == 8

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_fallback_cores(self, mock_run, mock_system):
        mock_system.return_value = "Linux"
        mock_run.side_effect = FileNotFoundError()
        name, cores = get_cpu_info()
        # fallback: platform.machine().count("64") + 2
        assert cores >= 2


class TestGetRamGb:
    @patch("pingpoint.profiler.get_ram_gb")
    def test_psutil(self, mock_self):
        pass

    @patch("pingpoint.profiler.platform.system")
    @patch("pingpoint.profiler.subprocess.run")
    def test_linux_free(self, mock_run, mock_system):
        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Mem:          16492674424  ...\n",
        )
        ram = get_ram_gb()
        import pytest
        assert ram == pytest.approx(15.36, rel=1e-3)

    @patch("pingpoint.profiler.platform.system")
    def test_fallback(self, mock_system):
        mock_system.return_value = "Unknown"
        ram = get_ram_gb()
        assert ram == 8.0


class TestProfile:
    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.detect_backend")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_low_end(
        self, mock_platform, mock_detect,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_backend = MagicMock()
        mock_backend.list_models.return_value = []
        mock_backend.name = "ollama"
        mock_detect.return_value = mock_backend
        mock_ram.return_value = 4.0
        mock_cpu.return_value = ("Unknown", 2)
        mock_gpu.return_value = (["Unknown"], None)

        p = profile()
        assert p.platform == "Linux"
        assert p.cpu_cores == 2
        assert p.ram_gb == 4.0
        assert p.gpu == ["Unknown"]
        assert p.vram_gb is None
        assert p.ollama_running is True
        assert p.ollama_models == []
        assert p.score == 6.0

    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.detect_backend")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_high_end(
        self, mock_platform, mock_detect,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_backend = MagicMock()
        mock_backend.list_models.return_value = ["llama3.2", "mistral", "llava"]
        mock_backend.name = "ollama"
        mock_detect.return_value = mock_backend
        mock_ram.return_value = 64.0
        mock_cpu.return_value = ("AMD Ryzen 9", 16)
        mock_gpu.return_value = (["NVIDIA RTX 4090"], 24.0)

        p = profile()
        assert p.capability == "high"
        assert p.has_gpu is True
        assert p.score == 100.0

    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.detect_backend")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_medium_gpu(
        self, mock_platform, mock_detect,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_backend = MagicMock()
        mock_backend.list_models.return_value = ["llama3.2"]
        mock_backend.name = "ollama"
        mock_detect.return_value = mock_backend
        mock_ram.return_value = 16.0
        mock_cpu.return_value = ("Intel i5", 4)
        mock_gpu.return_value = (["NVIDIA GTX 1060"], 6.0)

        p = profile()
        assert p.capability == "medium"
        assert p.score == 61.0
