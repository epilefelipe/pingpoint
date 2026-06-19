import platform
import subprocess
from unittest.mock import patch, MagicMock

from pingpoint.profiler import (
    get_ollama_models, is_ollama_running, get_ollama_version,
    get_gpu_info, get_cpu_info, get_ram_gb, profile,
)


class TestGetOllamaModels:
    @patch("pingpoint.profiler.subprocess.run")
    def test_models_found(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NAME\tSIZE\tMODIFIED\nllama3.2\t4.0GB\t2 days ago\nmistral\t4.0GB\t1 day ago\n",
        )
        models = get_ollama_models()
        assert models == ["llama3.2", "mistral"]

    @patch("pingpoint.profiler.subprocess.run")
    def test_no_models(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="NAME\tSIZE\tMODIFIED\n")
        models = get_ollama_models()
        assert models == []

    @patch("pingpoint.profiler.subprocess.run")
    def test_ollama_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        models = get_ollama_models()
        assert models == []


class TestIsOllamaRunning:
    @patch("pingpoint.profiler.subprocess.run")
    def test_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert is_ollama_running() is True

    @patch("pingpoint.profiler.subprocess.run")
    def test_not_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert is_ollama_running() is False

    @patch("pingpoint.profiler.subprocess.run")
    def test_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert is_ollama_running() is False


class TestGetOllamaVersion:
    @patch("pingpoint.profiler.subprocess.run")
    def test_version(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.32\n")
        assert get_ollama_version() == "ollama version 0.1.32"

    @patch("pingpoint.profiler.subprocess.run")
    def test_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert get_ollama_version() == "unknown"

    @patch("pingpoint.profiler.subprocess.run")
    def test_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert get_ollama_version() == "not installed"


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
        # Use the real function but mock psutil import
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
        assert ram == pytest.approx(15.36, rel=1e-3)  # 16492674424 / 1024^3

    @patch("pingpoint.profiler.platform.system")
    def test_fallback(self, mock_system):
        mock_system.return_value = "Unknown"
        ram = get_ram_gb()
        assert ram == 8.0


class TestProfile:
    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.get_ollama_models")
    @patch("pingpoint.profiler.is_ollama_running")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_low_end(
        self, mock_platform, mock_running, mock_models,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_running.return_value = True
        mock_models.return_value = []
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
        # score: no gpu (0), cores 2*2=4, ram 4/2=2, models 0
        assert p.score == 6.0  # 0 + 4 + 2 + 0

    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.get_ollama_models")
    @patch("pingpoint.profiler.is_ollama_running")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_high_end(
        self, mock_platform, mock_running, mock_models,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_running.return_value = True
        mock_models.return_value = ["llama3.2", "mistral", "llava"]
        mock_ram.return_value = 64.0
        mock_cpu.return_value = ("AMD Ryzen 9", 16)
        mock_gpu.return_value = (["NVIDIA RTX 4090"], 24.0)

        p = profile()
        assert p.capability == "high"
        assert p.has_gpu is True
        # score: gpu=40 + vram>=16 => +20 = 60, cores: min(16*2,20) = 20, ram: min(64/2,10)=10, models: min(3*5,10)=10
        # total: 60 + 20 + 10 + 10 = 100.0
        assert p.score == 100.0

    @patch("pingpoint.profiler.get_gpu_info")
    @patch("pingpoint.profiler.get_cpu_info")
    @patch("pingpoint.profiler.get_ram_gb")
    @patch("pingpoint.profiler.get_ollama_models")
    @patch("pingpoint.profiler.is_ollama_running")
    @patch("pingpoint.profiler.platform.system")
    def test_profile_medium_gpu(
        self, mock_platform, mock_running, mock_models,
        mock_ram, mock_cpu, mock_gpu,
    ):
        mock_platform.return_value = "Linux"
        mock_running.return_value = True
        mock_models.return_value = ["llama3.2"]
        mock_ram.return_value = 16.0
        mock_cpu.return_value = ("Intel i5", 4)
        mock_gpu.return_value = (["NVIDIA GTX 1060"], 6.0)

        p = profile()
        assert p.capability == "medium"
        # score: gpu=40 + vram>=8? no (6<8), so no extra = 40, cores: min(4*2,20)=8, ram: min(16/2,10)=8, models: min(1*5,10)=5
        # total: 40 + 8 + 8 + 5 = 61.0
        assert p.score == 61.0
